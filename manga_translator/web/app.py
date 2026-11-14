"""Self-hosted web UI: upload pages, translate them with your own API key, read the result.

    python -m manga_translator serve                 # http://127.0.0.1:8000
    python -m manga_translator serve --host 0.0.0.0 --token <secret>    # on a LAN / server

Chapters are plain folders under the library directory (default ``data/chapters``), the same
layout the CLI uses, so both can work on the same chapters.

API keys typed in the browser are kept in the browser and sent with each translate request.
The server holds them in memory for that job only and never writes them to disk.
"""

from __future__ import annotations

import hmac
import io
import json
import logging
import os
import re
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from ..schema import Chapter
from ..utils import FONTS_DIR, IMAGE_EXTENSIONS, list_page_images, natural_key
from .jobs import Job, JobRunner

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-()]{0,79}$")
MAX_UPLOAD = 60 * 1024 * 1024
MODELS = [
    {"id": "claude-opus-5-5", "label": "Claude Opus 5.5 (best)"},
    {"id": "claude-sonnet-5-5", "label": "Claude Sonnet 5.5"},
    {"id": "claude-haiku-5-5", "label": "Claude Haiku 5.5 (cheapest)"},
]
DEFAULT_SETTINGS = {"model": "claude-opus-5-5", "effort": "medium", "honorifics": "keep",
                    "inpaint": "lama", "uppercase": False, "sfx": False, "font": "ComicNeue-Bold.ttf"}
CATEGORIES = {"dialogue", "thought", "narration", "sign", "sfx", "noise"}


class NewChapter(BaseModel):
    name: str


class RegionEdit(BaseModel):
    translation: str | None = None
    short_translation: str | None = None
    category: str | None = None


class JobRequest(BaseModel):
    action: str = "translate"          # translate | render | redetect
    pages: list[str] | None = None
    retranslate: bool = False


class Login(BaseModel):
    token: str


def create_app(library: Path, token: str | None = None) -> FastAPI:
    library = library.resolve()
    library.mkdir(parents=True, exist_ok=True)
    runner = JobRunner(library)
    app = FastAPI(title="Manga Translator", docs_url=None, redoc_url=None)

    # ------------------------------------------------------------- access
    @app.middleware("http")
    async def require_token(request: Request, call_next):
        path = request.url.path
        if token and (path.startswith("/api/") or path.startswith("/img/")) and path != "/api/login":
            given = request.cookies.get("mt_token") or request.headers.get("x-access-token") or ""
            if not hmac.compare_digest(given.encode(), token.encode()):
                return JSONResponse({"detail": "login required"}, status_code=401)
        return await call_next(request)

    @app.post("/api/login")
    def login(body: Login):
        if not token or not hmac.compare_digest(body.token.encode(), token.encode()):
            raise HTTPException(401, "wrong token")
        resp = JSONResponse({"ok": True})
        resp.set_cookie("mt_token", token, httponly=True, samesite="strict", max_age=60 * 60 * 24 * 90)
        return resp

    # ------------------------------------------------------------ helpers
    def chapter_dir(name: str, must_exist: bool = True) -> Path:
        if not NAME_RE.match(name) or ".." in name:
            raise HTTPException(400, "invalid chapter name")
        d = library / name
        if must_exist and not d.is_dir():
            raise HTTPException(404, "chapter not found")
        return d

    def load_state(d: Path) -> Chapter:
        state = d / "work" / "chapter.json"
        return Chapter.load(state) if state.exists() else Chapter()

    def page_file(d: Path, page: str) -> Path:
        for p in list_page_images(d):
            if p.name == page:
                return p
        raise HTTPException(404, "page not found")

    def settings_of(d: Path) -> dict:
        f = d / "work" / "settings.json"
        saved = json.loads(f.read_text()) if f.exists() else {}
        return {**DEFAULT_SETTINGS, **{k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}}

    def busy(name: str) -> None:
        if runner.active_for(name):
            raise HTTPException(409, "a job is running on this chapter; wait for it or cancel it")

    def summary(d: Path) -> dict:
        state = {p.image: p for p in load_state(d).pages}
        pages = []
        for f in list_page_images(d):
            p = state.get(f.name)
            regions = p.regions if p else []
            rendered = d / "translated" / f.name
            pages.append({
                "name": f.name,
                "v": int(f.stat().st_mtime),
                "detected": p is not None,
                "regions": len(regions),
                "translated": sum(r.translated for r in regions),
                "rendered": int(rendered.stat().st_mtime) if rendered.exists() else 0,
            })
        return {"name": d.name, "pages": pages}

    # ------------------------------------------------------------- config
    @app.get("/api/config")
    def config():
        fonts = sorted(p.name for p in FONTS_DIR.glob("*") if p.suffix.lower() in (".ttf", ".otf"))
        return {"server_key": bool(os.environ.get("ANTHROPIC_API_KEY")), "models": MODELS, "fonts": fonts,
                "defaults": DEFAULT_SETTINGS, "auth": bool(token)}

    @app.post("/api/key/check")
    def check_key(x_anthropic_key: str | None = Header(default=None)):
        import anthropic

        try:
            client = anthropic.Anthropic(api_key=x_anthropic_key) if x_anthropic_key else anthropic.Anthropic()
            client.models.list(limit=1)
        except anthropic.AuthenticationError:
            return {"ok": False, "error": "The key was rejected (401)."}
        except Exception as exc:
            return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
        return {"ok": True}

    # ----------------------------------------------------------- chapters
    @app.get("/api/chapters")
    def chapters():
        out = []
        for d in sorted((p for p in library.iterdir() if p.is_dir() and NAME_RE.match(p.name)),
                        key=lambda p: natural_key(p.name)):
            s = summary(d)
            ps = s["pages"]
            out.append({"name": d.name, "pages": len(ps),
                        "translated": sum(1 for p in ps if p["regions"] and p["translated"] == p["regions"]),
                        "rendered": sum(1 for p in ps if p["rendered"]),
                        "cover": ps[0]["name"] if ps else None, "cover_v": ps[0]["v"] if ps else 0})
        return out

    @app.post("/api/chapters")
    def create_chapter(body: NewChapter):
        name = body.name.strip()
        d = chapter_dir(name, must_exist=False)
        if d.exists():
            raise HTTPException(409, "a chapter with that name already exists")
        d.mkdir(parents=True)
        return {"name": name}

    @app.delete("/api/chapters/{name}")
    def delete_chapter(name: str):
        d = chapter_dir(name)
        busy(name)
        shutil.rmtree(d)
        return {"ok": True}

    @app.get("/api/chapters/{name}")
    def chapter(name: str):
        d = chapter_dir(name)
        job = runner.active_for(name)
        return summary(d) | {"settings": settings_of(d), "job": job.public() if job else None}

    @app.put("/api/chapters/{name}/settings")
    def save_settings(name: str, body: dict):
        d = chapter_dir(name)
        merged = {**settings_of(d), **{k: v for k, v in body.items() if k in DEFAULT_SETTINGS}}
        (d / "work").mkdir(exist_ok=True)
        (d / "work" / "settings.json").write_text(json.dumps(merged, indent=2))
        return merged

    # -------------------------------------------------------------- pages
    @app.post("/api/chapters/{name}/pages")
    async def upload_pages(name: str, files: list[UploadFile] = File(...)):
        """Pages are appended in the order sent and numbered after the existing ones."""
        d = chapter_dir(name)
        busy(name)
        nums = [int(m.group(1)) for p in list_page_images(d) if (m := re.fullmatch(r"page(\d+)", p.stem))]
        n = max(nums) + 1 if nums else (1 if not list_page_images(d) else len(list_page_images(d)) + 1)
        saved = []
        for f in files:
            data = await f.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, f"{f.filename}: larger than 60 MB")
            try:
                with Image.open(io.BytesIO(data)) as im:
                    fmt = (im.format or "").upper()
                    im.verify()
            except Exception:
                raise HTTPException(400, f"{f.filename}: not an image")
            ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt)
            if ext is None:
                # Anything else Pillow can read (GIF, BMP, AVIF if supported...) is stored as PNG.
                with Image.open(io.BytesIO(data)) as im:
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, "PNG")
                data, ext = buf.getvalue(), ".png"
            assert ext in IMAGE_EXTENSIONS
            target = d / f"page{n}{ext}"
            target.write_bytes(data)
            saved.append(target.name)
            n += 1
        return {"saved": saved}

    @app.delete("/api/chapters/{name}/pages/{page}")
    def delete_page(name: str, page: str):
        d = chapter_dir(name)
        busy(name)
        f = page_file(d, page)
        state = load_state(d)
        state.pages = [p for p in state.pages if p.image != page]
        if (d / "work" / "chapter.json").exists():
            state.save(d / "work" / "chapter.json")
        for extra in (d / "translated" / page, d / "work" / "debug" / page, d / "work" / "thumbs" / f"{page}.jpg"):
            extra.unlink(missing_ok=True)
        f.unlink()
        return {"ok": True}

    @app.get("/api/chapters/{name}/pages/{page}")
    def page_detail(name: str, page: str):
        d = chapter_dir(name)
        page_file(d, page)
        p = load_state(d).page(page)
        if p is None:
            return {"name": page, "detected": False, "regions": []}
        regions = [{
            "id": r.id, "kind": r.kind, "box": r.text_box, "bubble_box": r.bubble_box,
            "original_text": r.original_text, "translation": r.translation,
            "short_translation": r.short_translation, "category": r.category,
        } for r in p.ordered_regions()]
        return {"name": page, "detected": True, "width": p.width, "height": p.height, "regions": regions}

    @app.patch("/api/chapters/{name}/pages/{page}/regions/{rid}")
    def edit_region(name: str, page: str, rid: str, body: RegionEdit):
        d = chapter_dir(name)
        busy(name)
        state = load_state(d)
        p = state.page(page)
        r = p.region(rid) if p else None
        if r is None:
            raise HTTPException(404, "region not found")
        if body.category is not None:
            if body.category not in CATEGORIES:
                raise HTTPException(400, "unknown category")
            r.category = body.category
        if body.translation is not None:
            r.translation = body.translation
        if body.short_translation is not None:
            r.short_translation = body.short_translation or None
        state.save(d / "work" / "chapter.json")
        return {"ok": True}

    # --------------------------------------------------------------- jobs
    @app.post("/api/chapters/{name}/jobs")
    def start_job(name: str, body: JobRequest, x_anthropic_key: str | None = Header(default=None)):
        d = chapter_dir(name)
        busy(name)
        if body.action not in ("translate", "render", "redetect"):
            raise HTTPException(400, "unknown action")
        if not list_page_images(d):
            raise HTTPException(400, "upload some pages first")
        if body.pages:
            for pg in body.pages:
                page_file(d, pg)
        opts = settings_of(d) | {"retranslate": body.retranslate}
        if body.action == "translate" and not x_anthropic_key and not os.environ.get("ANTHROPIC_API_KEY"):
            # Detection still makes sense without a key, but translation would fail on page 1.
            state = load_state(d)
            done = {p.image for p in state.pages if p.regions and all(r.translated for r in p.regions)}
            if body.retranslate or any(f.name not in done for f in list_page_images(d)
                                       if not body.pages or f.name in body.pages):
                raise HTTPException(400, "add an Anthropic API key in Settings first")
        job = runner.submit(Job(chapter=name, action=body.action, options=opts, pages=body.pages),
                            api_key=x_anthropic_key or None)
        return job.public()

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = runner.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job.public()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = runner.cancel(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job.public()

    # ------------------------------------------------------------- images
    @app.get("/img/{name}/{kind}/{page}")
    def image(name: str, kind: str, page: str):
        d = chapter_dir(name)
        src = page_file(d, page)
        if kind == "original":
            f = src
        elif kind == "translated":
            f = d / "translated" / page
        elif kind == "debug":
            f = d / "work" / "debug" / page
        elif kind == "thumb":
            f = d / "work" / "thumbs" / f"{page}.jpg"
            if not f.exists() or f.stat().st_mtime < src.stat().st_mtime:
                f.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    im.thumbnail((360, 540))
                    im.save(f, "JPEG", quality=82)
        elif kind == "thumb-translated":
            t = d / "translated" / page
            if not t.exists():
                raise HTTPException(404, "not rendered")
            f = d / "work" / "thumbs" / f"t_{page}.jpg"
            if not f.exists() or f.stat().st_mtime < t.stat().st_mtime:
                f.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(t) as im:
                    im = im.convert("RGB")
                    im.thumbnail((360, 540))
                    im.save(f, "JPEG", quality=82)
        else:
            raise HTTPException(404)
        if not f.exists():
            raise HTTPException(404, "not available yet")
        # URLs carry ?v=<mtime>, so a changed image gets a new URL and caching is safe.
        return FileResponse(f, headers={"Cache-Control": "private, max-age=31536000"})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(404)
    async def not_found(request: Request, exc):
        detail = getattr(exc, "detail", "not found")
        return JSONResponse({"detail": detail}, status_code=404)

    return app


def serve(library: Path, host: str = "127.0.0.1", port: int = 8000, token: str | None = None) -> None:
    import uvicorn

    token = token or os.environ.get("MANGA_TRANSLATOR_TOKEN") or None
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        log.warning("Serving on %s without --token: anyone who can reach this port can use the app "
                    "(and the server's API key, if one is set).", host)
    uvicorn.run(create_app(library, token), host=host, port=port, log_level="warning")

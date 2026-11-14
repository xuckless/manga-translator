"""Background jobs for the web UI.

One worker thread runs jobs in order: the detector and LaMa are large, and running two pages
through them at once would only fight over the same CPU/GPU. Models stay loaded between jobs.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..pipeline import ChapterPipeline
from ..render import PageRenderer, RenderSettings
from ..translation import FatalTranslationError
from ..utils import FONTS_DIR

log = logging.getLogger(__name__)


class Cancelled(Exception):
    pass


@dataclass
class Job:
    chapter: str
    action: str                       # "translate" (detect + translate + render) | "render" | "redetect"
    options: dict
    pages: list[str] | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"            # queued | running | done | failed | cancelled
    stage: str = ""
    done: int = 0
    total: int = 0
    page: str = ""
    message: str = ""
    log: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    finished: float | None = None
    cancel_requested: bool = False
    # Held in memory for the job only; never written to disk or returned by the API.
    _api_key: str | None = field(default=None, repr=False)

    def public(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "chapter", "action", "pages", "status", "stage", "done", "total", "page", "message",
            "created", "finished")} | {"log": self.log[-200:]}


class _JobLogHandler(logging.Handler):
    """Copies pipeline log lines into the running job so the UI can show them."""

    def __init__(self, runner: "JobRunner"):
        super().__init__(logging.INFO)
        self.runner = runner
        self.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        job = self.runner.current
        if job is not None and threading.current_thread() is self.runner.thread:
            job.log.append(self.format(record))


class JobRunner:
    def __init__(self, library: Path):
        self.library = library
        self.jobs: dict[str, Job] = {}
        self.queue: "queue.Queue[Job]" = queue.Queue()
        self.current: Job | None = None
        self._detector = None
        self._renderers: dict[tuple, PageRenderer] = {}
        self.thread = threading.Thread(target=self._loop, name="manga-jobs", daemon=True)
        logging.getLogger("manga_translator").addHandler(_JobLogHandler(self))
        self.thread.start()

    # ------------------------------------------------------------------ API
    def submit(self, job: Job, api_key: str | None = None) -> Job:
        job._api_key = api_key
        self.jobs[job.id] = job
        self.queue.put(job)
        return job

    def active_for(self, chapter: str) -> Job | None:
        for job in self.jobs.values():
            if job.chapter == chapter and job.status in ("queued", "running"):
                return job
        return None

    def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job and job.status in ("queued", "running"):
            job.cancel_requested = True
        return job

    # --------------------------------------------------------------- worker
    def _loop(self) -> None:
        while True:
            job = self.queue.get()
            if job.cancel_requested:
                job.status, job.finished = "cancelled", time.time()
                continue
            self.current = job
            job.status = "running"
            try:
                self._run(job)
                job.status = "done"
            except Cancelled:
                job.status = "cancelled"
                job.message = "Cancelled"
            except FatalTranslationError as exc:
                job.status = "failed"
                job.message = str(exc)
            except Exception as exc:  # surface to the UI, keep the worker alive
                job.status = "failed"
                job.message = str(exc) or exc.__class__.__name__
                job.log.append("ERROR " + traceback.format_exc(limit=3))
                log.exception("job %s failed", job.id)
            finally:
                job._api_key = None
                job.finished = time.time()
                self.current = None
            self._forget_old()

    def _forget_old(self, keep: int = 50) -> None:
        finished = sorted((j for j in self.jobs.values() if j.finished), key=lambda j: j.finished)
        for j in finished[:-keep]:
            self.jobs.pop(j.id, None)

    def _progress(self, job: Job, stage: str):
        def report(i: int, total: int, page: str) -> None:
            if job.cancel_requested:
                raise Cancelled()
            job.stage, job.done, job.total, job.page = stage, i, total, page
        return report

    def _run(self, job: Job) -> None:
        pipeline = ChapterPipeline(self.library / job.chapter)
        opts = job.options

        if job.action in ("translate", "redetect"):
            job.stage = "detect"
            pipeline.detect(force=job.action == "redetect", only=job.pages, detector=self.detector(),
                            progress=self._progress(job, "detect"))

        touched = job.pages
        if job.action == "translate":
            todo = pipeline.pages_to_translate(force=bool(opts.get("retranslate")), only=job.pages)
            touched = [p.image for p in todo] if not job.pages else job.pages
            if todo:
                from ..translation.claude import ClaudeTranslator

                translator = ClaudeTranslator(model=opts.get("model", "claude-opus-5-5"),
                                              effort=opts.get("effort", "medium"),
                                              honorifics=opts.get("honorifics", "keep"),
                                              api_key=job._api_key)
                job.stage = "translate"
                pending = pipeline.translate(translator, force=bool(opts.get("retranslate")), only=touched,
                                             progress=self._progress(job, "translate"))
                if pending == len(todo):
                    raise RuntimeError("No page could be translated; see the log for the reason.")
                if pending:
                    job.message = f"{pending} page(s) failed to translate; see the log. Run again to retry."
            # Re-letter what was just translated, plus any page that was never rendered.
            out = pipeline.dir / "translated"
            never = [p.image for p in pipeline.chapter.pages if not (out / p.image).exists()]
            touched = sorted(set(touched or []) | set(never))
            if not touched:
                job.done = job.total = 0
                job.stage = "done"
                return

        settings = render_settings(opts)
        job.stage = "render"
        pipeline.render(settings, only=touched, renderer=self.renderer(settings),
                        progress=self._progress(job, "render"))
        job.stage = "done"
        job.done = job.total

    # --------------------------------------------------------------- models
    def detector(self):
        if self._detector is None:
            from ..detection import TextDetector
            self._detector = TextDetector()
        return self._detector

    def renderer(self, settings: RenderSettings) -> PageRenderer:
        key = (settings.inpaint,)
        if key not in self._renderers:
            self._renderers[key] = PageRenderer(settings)
        r = self._renderers[key]
        r.s = settings          # font/uppercase/sfx can change per job; the inpainter is reused
        return r


def render_settings(opts: dict) -> RenderSettings:
    font = opts.get("font")
    kwargs = {}
    if font and (FONTS_DIR / font).is_file():
        kwargs["font_path"] = FONTS_DIR / font
    return RenderSettings(inpaint=opts.get("inpaint", "lama"), uppercase=bool(opts.get("uppercase")),
                          render_sfx=bool(opts.get("sfx")), debug=bool(opts.get("debug")), **kwargs)

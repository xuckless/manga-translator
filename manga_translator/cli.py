"""Command line entry point: ``python -m manga_translator <command> ...``"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import ChapterPipeline
from .render import RenderSettings

log = logging.getLogger("manga_translator")


def _translator(args, pipeline: ChapterPipeline):
    if args.backend == "claude":
        from .translation.claude import ClaudeTranslator
        return ClaudeTranslator(model=args.model, effort=args.effort, honorifics=args.honorifics)
    from .translation.manual import ManualTranslator
    return ManualTranslator(pipeline.work / "manual", honorifics=args.honorifics)


def _render_settings(args) -> RenderSettings:
    return RenderSettings(inpaint=args.inpaint, uppercase=args.uppercase, render_sfx=args.sfx, debug=args.debug,
                          **({"font_path": Path(args.font)} if args.font else {}))


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="manga_translator", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sv = sub.add_parser("serve", help="start the web UI")
    sv.add_argument("--library", default="data/chapters", help="folder holding one sub-folder per chapter")
    sv.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to serve on your network")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--token", help="access token required to use the UI (or MANGA_TRANSLATOR_TOKEN)")

    def chapter_cmd(name, help_):
        c = sub.add_parser(name, help=help_)
        c.add_argument("chapter", help="chapter directory containing the page images")
        c.add_argument("--pages", nargs="*", help="only these page files (e.g. page5.jpg)")
        return c

    d = chapter_cmd("detect", "find text regions and speech bubbles")
    d.add_argument("--force", action="store_true")
    d.add_argument("--threshold", type=float, default=0.35)

    def add_translate(c):
        c.add_argument("--backend", choices=["claude", "manual"], default="claude")
        c.add_argument("--model", default="claude-opus-5-5")
        c.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
        c.add_argument("--honorifics", choices=["keep", "localize"], default="keep")
        c.add_argument("--retranslate", action="store_true", help="re-translate pages that are already done")

    def add_render(c):
        c.add_argument("--inpaint", choices=["lama", "opencv"], default="lama")
        c.add_argument("--font", help="path to a .ttf/.otf font")
        c.add_argument("--uppercase", action="store_true")
        c.add_argument("--sfx", action="store_true", help="also replace sound effects (default: keep the art)")
        c.add_argument("--debug", action="store_true", help="write mask overlays to work/debug/")

    add_translate(chapter_cmd("translate", "OCR + translate each page with a vision model"))
    add_render(chapter_cmd("render", "clean the pages and letter the translation"))
    r = chapter_cmd("run", "detect + translate + render")
    add_translate(r)
    add_render(r)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpx2", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cmd == "serve":
        from .web.app import serve
        print(f"Manga Translator on http://{args.host}:{args.port}  (library: {Path(args.library).resolve()})")
        serve(Path(args.library), host=args.host, port=args.port, token=args.token)
        return 0

    pipeline = ChapterPipeline(args.chapter)
    if args.cmd in ("detect", "run"):
        if args.cmd == "detect" or not pipeline.chapter.pages:
            pipeline.detect(force=getattr(args, "force", False), threshold=getattr(args, "threshold", 0.35),
                            only=args.pages)
    if args.cmd in ("translate", "run"):
        pending = pipeline.translate(_translator(args, pipeline), force=args.retranslate, only=args.pages)
        if pending:
            where = pipeline.work / "manual" if args.backend == "manual" else "the log above"
            log.warning("%d page(s) still need translation - see %s", pending, where)
    if args.cmd in ("render", "run"):
        pipeline.render(_render_settings(args), only=args.pages)
    return 0


if __name__ == "__main__":
    sys.exit(main())

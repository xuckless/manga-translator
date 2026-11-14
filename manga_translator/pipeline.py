"""Chapter pipeline. Every stage is resumable and saves after each page.

    <chapter>/                 source pages (page1.jpg, page2.jpg, ...)
    <chapter>/work/chapter.json   detections + translations (the single source of truth)
    <chapter>/work/manual/     request/response files for the manual backend
    <chapter>/work/debug/      mask overlays (render --debug)
    <chapter>/translated/      final pages
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from .render import PageRenderer, RenderSettings
from .schema import Chapter, Page
from .translation import FatalTranslationError, apply_translation, build_request
from .utils import list_page_images, natural_key

log = logging.getLogger(__name__)

# Called before each page is processed: (index, total, page file). May raise to cancel.
ProgressFn = Optional[Callable[[int, int, str], None]]


def _carry_over_translations(old_page: Page, new: list) -> list[str]:
    """Keep translations when re-detecting: copy them onto the new region covering the same text.
    Returns the new reading order (new ids ranked by their matched old region's position)."""
    from .utils import box_intersection, box_area

    rank = {rid: i for i, rid in enumerate(old_page.reading_order)}
    order: list[tuple[float, str]] = []
    for n in new:
        # Every translated old region that sits mostly inside this new one (a re-detect can fold
        # two neighbouring boxes into one), real text before noise.
        matches = []
        for o in old_page.regions:
            if not o.translated:
                continue
            inter = box_intersection(o.text_box, n.text_box)
            if inter >= 0.6 * max(1.0, min(box_area(o.text_box), box_area(n.text_box))):
                matches.append(o)
        real = [o for o in matches if o.category != "noise"] or matches
        real.sort(key=lambda o: rank.get(o.id, len(rank)))
        if real:
            n.original_text = "".join(o.original_text or "" for o in real)
            n.translation = " ".join(o.translation for o in real if o.translation).strip()
            n.short_translation = " ".join((o.short_translation or o.translation) for o in real if o.translation).strip()
            n.category = real[0].category
        order.append((rank.get(real[0].id, len(rank)) if real else len(rank) + int(n.id), n.id))
    return [rid for _, rid in sorted(order)] if old_page.reading_order else []


class ChapterPipeline:
    def __init__(self, chapter_dir: str | Path):
        self.dir = Path(chapter_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"Chapter directory not found: {self.dir}")
        self.work = self.dir / "work"
        self.state_path = self.work / "chapter.json"
        self.chapter = Chapter.load(self.state_path) if self.state_path.exists() else Chapter()

    def save(self) -> None:
        self.chapter.save(self.state_path)

    # ------------------------------------------------------------------ detect
    def detect(self, force: bool = False, threshold: float = 0.35, only: list[str] | None = None,
               detector=None, progress: ProgressFn = None) -> None:
        if detector is None:
            from .detection import TextDetector
            detector = TextDetector(threshold=threshold)
        existing = {p.image: p for p in self.chapter.pages}
        paths = list_page_images(self.dir)
        todo = [p.name for p in paths
                if p.name not in existing or (force and (not only or p.name in only))]
        pages: list[Page] = []
        for path in paths:
            page = existing.get(path.name)
            if path.name in todo:
                if progress:
                    progress(todo.index(path.name), len(todo), path.name)
                with Image.open(path) as im:
                    regions = detector.detect(im)
                order = _carry_over_translations(page, regions) if page is not None else []
                page = Page(image=path.name, width=im.width, height=im.height, regions=regions,
                            reading_order=order, summary=page.summary if page else None)
                log.info("detect %s: %d regions", path.name, len(regions))
            pages.append(page)
        self.chapter.pages = sorted(pages, key=lambda p: natural_key(p.image))
        self.save()

    # --------------------------------------------------------------- translate
    def pages_to_translate(self, force: bool = False, only: list[str] | None = None) -> list[Page]:
        return [p for p in self.chapter.pages
                if (not only or p.image in only) and p.regions
                and (force or not all(r.translated for r in p.regions))]

    def translate(self, translator, force: bool = False, only: list[str] | None = None,
                  progress: ProgressFn = None) -> int:
        """Translate pages in reading order. Returns the number of pages still pending."""
        if not self.chapter.pages:
            raise RuntimeError("Run detection first.")
        pending = 0
        todo = self.pages_to_translate(force, only)
        number_of = {p.image: i for i, p in enumerate(self.chapter.pages, 1)}
        for k, page in enumerate(todo):
            if progress:
                progress(k, len(todo), page.image)
            number = number_of[page.image]
            with Image.open(self.dir / page.image) as im:
                request = build_request(self.chapter, page, im, number)
            try:
                result = translator.translate_page(request)
            except FatalTranslationError:
                raise
            except Exception as exc:  # keep going; a re-run picks the page up again
                log.error("translate %s failed: %s", page.image, exc)
                pending += 1
                continue
            if result is None:          # manual backend: request written, awaiting response
                pending += 1
                continue
            missing = apply_translation(self.chapter, page, result)
            if missing:
                log.warning("translate %s: no answer for regions %s", page.image, ", ".join(missing))
            log.info("translate %s: %d regions", page.image, len(page.regions) - len(missing))
            self.save()
        return pending

    # ------------------------------------------------------------------ render
    def render(self, settings: RenderSettings, only: list[str] | None = None,
               renderer: PageRenderer | None = None, progress: ProgressFn = None) -> None:
        renderer = renderer or PageRenderer(settings)
        out_dir = self.dir / "translated"
        todo = [p for p in self.chapter.pages if not only or p.image in only]
        for k, page in enumerate(todo):
            if progress:
                progress(k, len(todo), page.image)
            if not any(r.translated for r in page.regions):
                log.info("render %s: nothing translated, copying as-is", page.image)
            debug = self.work / "debug" / page.image if settings.debug else None
            stats = renderer.render(page, self.dir / page.image, out_dir / page.image, debug)
            notes = ", ".join(f"{k}={v}" for k, v in stats.items() if k != "regions" and v)
            log.info("render %s: %d regions%s", page.image, stats["regions"], f" ({notes})" if notes else "")

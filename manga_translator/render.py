"""Clean the Japanese text off a page and letter the English in."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import Image

from .inpaint import Inpainter
from .masks import RegionMasks, build_masks
from .schema import Page, Region  # noqa: F401
from .typeset import TextLayout, Typesetter
from .utils import FONTS_DIR

log = logging.getLogger(__name__)


@dataclass
class RenderSettings:
    font_path: Path = FONTS_DIR / "ComicNeue-Bold.ttf"
    inpaint: str = "lama"            # "lama" | "opencv"
    uppercase: bool = False
    render_sfx: bool = False         # SFX are drawn art; by default they are left untouched
    min_font_frac: float = 0.0095    # font size bounds as a fraction of page height
    max_font_frac: float = 0.026
    size_spread: float = 1.3         # no balloon's text may exceed page median size by more than this
    debug: bool = False


@dataclass
class _Job:
    region: Region
    masks: RegionMasks
    layout: TextLayout | None = None


class PageRenderer:
    def __init__(self, settings: RenderSettings):
        self.s = settings
        self.inpainter = Inpainter(settings.inpaint)

    def _wanted(self, r: Region) -> bool:
        if not r.should_render:
            return False
        return r.category != "sfx" or self.s.render_sfx

    def render(self, page: Page, src: Path, out: Path, debug_out: Path | None = None) -> dict:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        jobs = [_Job(r, build_masks(img, r)) for r in page.ordered_regions() if self._wanted(r)]
        _split_shared_balloons(jobs)

        # 1) Clean. Flat balloon paper: paint the glyphs with the paper colour. Everything else
        #    (text over artwork, screentone-filled balloons) goes to the inpainter in one batch.
        cleaned = img.copy()
        art_mask = np.zeros((H, W), bool)
        for j in jobs:
            if j.region.kind == "bubble" and j.masks.flat:
                r, g, b = j.masks.bg_color
                cleaned[j.masks.erase] = (b, g, r)
            else:
                art_mask |= j.masks.erase
        cleaned = self.inpainter.inpaint(cleaned, art_mask)

        # 2) Lay out. First pass finds each balloon's best size; second pass caps outliers so a
        #    tiny "…" balloon doesn't end up in giant letters next to normal dialogue.
        ts = Typesetter(self.s.font_path, H * self.s.min_font_frac, H * self.s.max_font_frac,
                        uppercase=self.s.uppercase)
        for j in jobs:
            j.layout = self._fit(ts, j)
        sizes = [j.layout.font_size for j in jobs if len(j.region.translation or "") > 12]
        if sizes:
            cap = int(median(sizes) * self.s.size_spread)
            for j in jobs:
                if j.layout.font_size > cap:
                    j.layout = self._fit(ts, j, max_px=cap)

        # 3) Draw.
        canvas = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
        stats = {"regions": len(jobs), "overflow": [], "short": [], "untraced": []}
        for j in jobs:
            ys, xs = np.nonzero(j.masks.layout)
            origin = (int(xs.min()), int(ys.min()))
            if j.region.kind == "bubble":
                fill = _ink(j.masks.text_color, j.masks.bg_color)
                ts.draw(canvas, j.layout, origin, fill=fill)
                if not j.masks.traced:
                    stats["untraced"].append(j.region.id)
            else:
                ts.draw(canvas, j.layout, origin, fill=j.masks.text_color, stroke=j.masks.bg_color)
            if j.layout.overflow:
                stats["overflow"].append(j.region.id)
            if j.layout.used_short:
                stats["short"].append(j.region.id)

        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out, quality=95)
        if debug_out:
            self._debug(img, jobs, debug_out)
        return stats

    def _fit(self, ts: Typesetter, j: _Job, max_px: int | None = None) -> TextLayout:
        ys, xs = np.nonzero(j.masks.layout)
        crop = j.masks.layout[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return ts.fit(j.region.translation, crop, j.region.short_translation, max_px=max_px)

    @staticmethod
    def _debug(img: np.ndarray, jobs: list[_Job], out: Path) -> None:
        vis = img.copy()
        for j in jobs:
            vis[j.masks.erase] = (0.5 * vis[j.masks.erase] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            contours, _ = cv2.findContours(j.masks.layout.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (0, 200, 0) if j.masks.traced or j.region.kind == "free" else (0, 165, 255)
            cv2.drawContours(vis, contours, -1, color, 2)
            x1, y1 = j.region.text_box[:2]
            cv2.putText(vis, j.region.id, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), vis)


def _split_shared_balloons(jobs: list[_Job]) -> None:
    """Conjoined balloons (two lobes, one white shape) and balloons the detector split into two
    boxes make neighbouring traces overlap. Letterers keep each line in its own lobe, so the
    combined shape is divided: every pixel goes to the region whose text box is nearer."""
    bubbles = [j for j in jobs if j.region.kind == "bubble"]
    for i, a in enumerate(bubbles):
        for b in bubbles[i + 1:]:
            shared = a.masks.layout & b.masks.layout
            if not shared.any():
                continue
            union = a.masks.layout | b.masks.layout
            ys, xs = np.nonzero(union)
            da = _dist_to_box(xs, ys, a.region.text_box)
            db = _dist_to_box(xs, ys, b.region.text_box)
            to_a = da <= db
            a.masks.layout = np.zeros_like(union)
            b.masks.layout = np.zeros_like(union)
            a.masks.layout[ys[to_a], xs[to_a]] = True
            b.masks.layout[ys[~to_a], xs[~to_a]] = True
            # Keep only each side's main piece, so stray fragments don't drag the layout around.
            a.masks.layout = _largest_component(a.masks.layout)
            b.masks.layout = _largest_component(b.masks.layout)


def _dist_to_box(xs: np.ndarray, ys: np.ndarray, box) -> np.ndarray:
    dx = np.maximum(np.maximum(box[0] - xs, 0), xs - box[2])
    dy = np.maximum(np.maximum(box[1] - ys, 0), ys - box[3])
    return np.hypot(dx, dy)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    if n <= 2:
        return mask
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def _ink(text_color, bg_color) -> tuple[int, int, int]:
    """Use the original ink colour, unless it's too close to the paper to read."""
    contrast = abs(sum(text_color) - sum(bg_color)) / 3
    if contrast >= 90:
        return tuple(text_color)
    return (0, 0, 0) if sum(bg_color) / 3 > 127 else (255, 255, 255)

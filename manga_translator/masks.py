"""Pixel masks from detector boxes.

Two masks matter per region:

* **layout mask** - where translated text may go. For bubbles this is the balloon's inner
  area (traced from the image, so ovals, spikes and clouds keep their shape). For free text
  it is the text box, widened when the Japanese was a tall narrow column.
* **erase mask** - the Japanese glyph pixels to remove. Inside balloons these are filled with
  the balloon colour; over artwork they are inpainted.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .schema import Region
from .utils import clamp_box, expand_box


@dataclass
class RegionMasks:
    layout: np.ndarray          # bool, full page size
    erase: np.ndarray           # bool, full page size
    bg_color: tuple[int, int, int]
    text_color: tuple[int, int, int]
    traced: bool                # True if the balloon outline was traced (not the ellipse fallback)
    flat: bool = True           # paper is a flat colour (fill it) rather than textured (inpaint it)


def build_masks(img_bgr: np.ndarray, region: Region) -> RegionMasks:
    if region.kind == "bubble":
        return _bubble_masks(img_bgr, region)
    return _free_text_masks(img_bgr, region)


# ---------------------------------------------------------------------------
# Speech bubbles
# ---------------------------------------------------------------------------

def _bubble_masks(img: np.ndarray, region: Region) -> RegionMasks:
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tb = clamp_box(region.text_box, W, H)
    bb = clamp_box(region.bubble_box or expand_box(tb, 20, W, H), W, H)
    bb = clamp_box([min(bb[0], tb[0]), min(bb[1], tb[1]), max(bb[2], tb[2]), max(bb[3], tb[3])], W, H)
    x1, y1, x2, y2 = expand_box(bb, 3, W, H)
    crop = gray[y1:y2, x1:x2]
    t = [tb[0] - x1, tb[1] - y1, tb[2] - x1, tb[3] - y1]

    # Paper tone candidates: the dominant smoothed tones of the text box, centre first (the centre
    # stays inside the balloon even when the box pokes over its rim). Big bold lettering can
    # out-vote the paper, so the second peak is tried too; the first that traces a plausible
    # balloon wins.
    blurred = cv2.medianBlur(crop, 5)
    qw, qh = (t[2] - t[0]) // 4, (t[3] - t[1]) // 4
    centre = np.s_[t[1] + qh:t[3] - qh, t[0] + qw:t[2] - qw]
    whole = np.s_[t[1]:t[3], t[0]:t[2]]
    candidates: list[float] = []
    for sl in (centre, whole):
        area = blurred[sl]
        if area.size < 25:
            continue
        hist = np.bincount((area // 8).ravel(), minlength=32).astype(float)
        for _ in range(2):
            peak = int(np.argmax(hist))
            if hist[peak] <= 0.08 * area.size:
                break
            level = peak * 8 + 4.0
            if all(abs(level - c) > 24 for c in candidates):
                candidates.append(level)
            hist[max(0, peak - 3):peak + 4] = 0
    if not candidates:
        candidates = [255.0]

    # Raw image first (keeps thin outlines closed); blurred second (merges screentone dots).
    traced, interior, bg_level = False, None, candidates[0]
    for level in candidates:
        for src in (crop, blurred):
            found = _trace_interior(src, level, t)
            if found is not None and _plausible_interior(found, t, crop.shape):
                traced, interior, bg_level = True, found, level
                break
        if traced:
            break
    textured = _is_textured(crop, blurred, centre if blurred[centre].size >= 25 else whole, bg_level)

    if not traced:
        # Ellipse inscribed in the detected balloon box - safe for borderless/odd balloons.
        interior = np.zeros(crop.shape, np.uint8)
        cx, cy = (bb[0] + bb[2]) / 2 - x1, (bb[1] + bb[3]) / 2 - y1
        cv2.ellipse(interior, (int(cx), int(cy)), (int((bb[2] - bb[0]) * 0.47), int((bb[3] - bb[1]) * 0.47)),
                    0, 0, 360, 1, thickness=cv2.FILLED)
        interior = interior.astype(bool) | _box_mask(crop.shape, t)

    # Glyph pixels: anything inside the balloon that is not paper colour, grown slightly to catch
    # anti-aliasing. Eroding the interior keeps the balloon's own outline untouched.
    inner = cv2.erode(interior.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    # Glyphs differ from the *smoothed* paper tone; screentone dots alone don't (they blur away).
    if textured:   # isolated dots vanish under the median blur; letter strokes and halos don't
        ink = (np.abs(blurred.astype(np.int16) - bg_level) >= 40) & inner
    else:
        ink = (np.abs(crop.astype(np.int16) - bg_level) >= 45) & inner
    erase = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool) & inner

    bgr_crop = img[y1:y2, x1:x2]
    bg_color = _median_color(bgr_crop, interior & ~erase, default=(255, 255, 255))
    # Lettering colour = the core of the strokes, not their anti-aliased edges.
    if ink.sum() >= 10:
        dist = np.abs(crop.astype(np.int16) - bg_level)
        core = ink & (dist >= np.percentile(dist[ink], 70))
        text_color = _median_color(bgr_crop, core, default=(0, 0, 0))
    else:
        text_color = (0, 0, 0)

    paper = crop[interior & ~erase]
    flat = paper.size < 50 or float(np.std(paper)) < 12

    return RegionMasks(
        layout=_to_page(interior, (x1, y1), (H, W)),
        erase=_to_page(erase, (x1, y1), (H, W)),
        bg_color=bg_color,
        text_color=text_color,
        traced=traced,
        flat=flat,
    )


def _is_textured(crop: np.ndarray, blurred: np.ndarray, sl, level: float) -> bool:
    """Screentone paper: pixels whose smoothed tone is the paper still vary a lot individually."""
    area, raw = blurred[sl], crop[sl]
    near = raw[np.abs(area.astype(np.int16) - level) < 12]
    return near.size > 20 and float(np.std(near)) > 14


def _trace_interior(img: np.ndarray, level: float, t) -> np.ndarray | None:
    """The balloon interior: the paper-toned component overlapping the text most, holes filled."""
    light = (np.abs(img.astype(np.int16) - level) < 45).astype(np.uint8)
    n, labels, _, _ = cv2.connectedComponentsWithStats(light, connectivity=4)
    if n <= 1:
        return None
    counts = np.bincount(labels[t[1]:t[3], t[0]:t[2]].ravel(), minlength=n)
    counts[0] = 0  # label 0 = non-paper pixels (ink, outlines)
    comp = (labels == int(np.argmax(counts))).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def _plausible_interior(interior: np.ndarray, t, shape) -> bool:
    """Reject traces that leaked into the page background or only caught a sliver."""
    area = interior.sum()
    text_area = max(1, (t[2] - t[0]) * (t[3] - t[1]))
    if area < 0.6 * text_area:
        return False
    h, w = shape
    border = np.concatenate([interior[0], interior[-1], interior[:, 0], interior[:, -1]])
    # A closed balloon only touches the crop edge at a few points (tails, panel borders).
    return border.mean() < 0.35 and area < 0.97 * h * w


# ---------------------------------------------------------------------------
# Free text over artwork
# ---------------------------------------------------------------------------

def _free_text_masks(img: np.ndarray, region: Region) -> RegionMasks:
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tb = clamp_box(region.text_box, W, H)
    # Detector boxes often clip the first/last glyph of a column and the furigana beside it,
    # so search a wider window and keep whole glyphs that touch the box.
    reach = max(10, int(0.012 * H))
    x1, y1, x2, y2 = expand_box(tb, reach, W, H)
    crop = gray[y1:y2, x1:x2]
    inside = np.zeros(crop.shape, bool)
    inside[tb[1] - y1:tb[3] - y1, tb[0] - x1:tb[2] - x1] = True

    # Manga narration is usually dark glyphs with a white halo; occasionally the inverse.
    dark_text = np.median(crop[inside]) > 110
    ink = crop < 100 if dark_text else crop > 160
    glyph = _glyphs_touching(ink, inside, max_side=int(0.08 * H))
    glyph = _drop_specks(glyph, min_area=max(4, int(0.00002 * W * H)))

    # Swallow the outline/halo drawn around the letters: the paper-coloured pixels hugging them.
    halo = max(3, int(round(H * 0.004)))
    disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    near = cv2.dilate(glyph.astype(np.uint8), disk(3 * halo)).astype(bool)
    # Only count a halo where it stands out from the art behind the text; otherwise bright (or
    # dark) background next to the letters gets swallowed and inpainted for no reason.
    ring = ~cv2.dilate(inside.astype(np.uint8), disk(halo)).astype(bool)
    backdrop = float(np.median(crop[ring])) if ring.any() else (255.0 if dark_text else 0.0)
    if dark_text:
        outline = near & (crop > max(185.0, backdrop + 30))
    else:
        outline = near & (crop < min(70.0, backdrop - 30))
    erase = cv2.dilate((glyph | outline).astype(np.uint8), disk(max(2, halo // 2))).astype(bool)
    erase |= cv2.dilate(glyph.astype(np.uint8), disk(halo)).astype(bool)

    text_color = (20, 20, 20) if dark_text else (245, 245, 245)
    bg_color = (255, 255, 255) if dark_text else (0, 0, 0)

    layout = np.zeros((H, W), bool)
    lx1, ly1, lx2, ly2 = _widen_for_english(region.text_box, gray)
    layout[ly1:ly2, lx1:lx2] = True

    return RegionMasks(
        layout=layout,
        erase=_to_page(erase, (x1, y1), (H, W)),
        bg_color=bg_color,
        text_color=text_color,
        traced=False,
        flat=False,
    )


def _widen_for_english(box, gray: np.ndarray) -> list[int]:
    """Japanese runs in tall columns; English needs width. Widen narrow boxes around their centre,
    but never across a panel border or into the gutter between panels."""
    H, W = gray.shape
    x1, y1, x2, y2 = clamp_box(box, W, H)
    w, h = x2 - x1, y2 - y1
    target_w = max(w, int(h * 0.75))
    if target_w <= w:
        return [x1, y1, x2, y2]
    cy = (y1 + y2) / 2
    new_h = max(h * 0.7, w * 1.2)          # shrink height a bit: area stays comparable
    ny1, ny2 = int(max(0, cy - new_h / 2)), int(min(H, cy + new_h / 2))
    band = gray[ny1:ny2]

    def is_wall(x: int) -> bool:
        col = band[:, x]
        return (col < 60).mean() > 0.8 or (col > 245).mean() > 0.97   # border line / white gutter

    grow = (target_w - w) // 2
    left = x1
    while left > 0 and x1 - left < grow and not is_wall(left - 1):
        left -= 1
    right = x2
    while right < W and right - x2 < grow and not is_wall(right):
        right += 1
    # If one side is blocked, give the remaining width to the other side.
    extra = grow * 2 - ((x1 - left) + (right - x2))
    while extra > 0 and right < W and not is_wall(right):
        right += 1; extra -= 1
    while extra > 0 and left > 0 and not is_wall(left - 1):
        left -= 1; extra -= 1
    return [left, ny1, right, ny2]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _glyphs_touching(ink: np.ndarray, inside: np.ndarray, max_side: int) -> np.ndarray:
    """Ink inside the box, plus whole glyph-sized components that poke out of it.

    Large components (panel borders, art strokes crossing the box) are only kept inside the box.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    touches = np.zeros(n, bool)
    touches[np.unique(labels[inside & ink])] = True
    small = (stats[:, cv2.CC_STAT_WIDTH] <= max_side) & (stats[:, cv2.CC_STAT_HEIGHT] <= max_side)
    keep = touches & small
    keep[0] = False
    return keep[labels] | (ink & inside)


def _drop_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def _box_mask(shape, t) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[max(0, t[1]):t[3], max(0, t[0]):t[2]] = True
    return m


def _median_color(bgr: np.ndarray, mask: np.ndarray, default) -> tuple[int, int, int]:
    if mask.sum() < 10:
        return default
    b, g, r = (int(np.median(bgr[..., c][mask])) for c in range(3))
    return (r, g, b)


def _to_page(mask: np.ndarray, origin, page_shape) -> np.ndarray:
    out = np.zeros(page_shape, bool)
    x, y = origin
    out[y:y + mask.shape[0], x:x + mask.shape[1]] = mask
    return out

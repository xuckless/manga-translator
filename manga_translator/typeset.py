"""Fit translated text into balloon shapes and draw it.

Layout works on the balloon's actual mask, not its bounding box: each candidate line gets the
width the balloon has *at that height*, so text naturally forms the diamond/lozenge shape
letterers use in round balloons. Line breaks are chosen with a small dynamic program that
balances line lengths and avoids ugly breaks (ending a line on "the", a lone last word, ...).

Fallback chain when space is tight:
    full translation at the largest size that fits
 -> condensed ``short_translation`` (written by the translator for exactly this case)
 -> smallest size with overflow (flagged so it can be reviewed)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Words a line should not end with: breaking after them separates them from what they modify.
WEAK_LINE_END = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "as",
    "and", "or", "but", "nor", "so", "if", "than", "that", "my", "your", "his", "her", "its",
    "our", "their", "this", "these", "those", "i", "i'm", "i'll", "we", "you", "is", "are", "was",
    "be", "not", "no", "mr.", "ms.", "mrs.", "dr.",
}


@dataclass
class Line:
    text: str
    x: float        # left edge, relative to the layout crop
    y: float        # top of the line box
    width: float


@dataclass
class TextLayout:
    font_size: int
    lines: list[Line] = field(default_factory=list)
    used_short: bool = False
    overflow: bool = False


class Typesetter:
    def __init__(self, font_path: str | Path, min_px: int, max_px: int,
                 line_spacing: float = 1.08, uppercase: bool = False):
        self.font_path = str(font_path)
        self.min_px = max(6, int(min_px))
        self.max_px = max(self.min_px, int(max_px))
        self.line_spacing = line_spacing
        self.uppercase = uppercase

    # ------------------------------------------------------------------ fonts
    @lru_cache(maxsize=256)
    def font(self, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(self.font_path, size)

    # ------------------------------------------------------------------ layout
    def fit(self, text: str, mask: np.ndarray, short_text: str | None = None,
            max_px: int | None = None) -> TextLayout:
        """Lay ``text`` out inside ``mask`` (bool array, the crop's own coordinates)."""
        text = _normalise(text, self.uppercase)
        short = _normalise(short_text, self.uppercase) if short_text else None
        hi = min(self.max_px, max_px or self.max_px)
        shape = _Shape(mask)

        for candidate, is_short in ((text, False), (short, True)):
            if not candidate or (is_short and candidate == text):
                continue
            # Splitting a word with a hyphen is a last resort: first look for any size that fits
            # whole words, and only then allow hyphenation.
            for hyphenate in (False, True):
                layout = self._largest_fit(candidate, shape, self.min_px, hi, hyphenate)
                if layout:
                    layout.used_short = is_short
                    return layout

        # Nothing fits: smallest size, greedy wrap at the widest row, allowed to spill.
        best = short or text
        return self._overflow_layout(best, shape, self.min_px, used_short=bool(short))

    def _largest_fit(self, text: str, shape: "_Shape", lo: int, hi: int,
                     hyphenate: bool = True) -> TextLayout | None:
        # Feasibility is (almost) monotonic in font size: binary search for the largest that fits.
        if self._try_size(text, shape, lo, hyphenate) is None:
            return None
        a, b = lo, hi
        while a < b:
            mid = (a + b + 1) // 2
            if self._try_size(text, shape, mid, hyphenate) is not None:
                a = mid
            else:
                b = mid - 1
        return self._try_size(text, shape, a, hyphenate)

    @lru_cache(maxsize=4096)
    def _metrics(self, size: int) -> tuple[int, int, float]:
        ascent, descent = self.font(size).getmetrics()
        return ascent, descent, self.font(size).getlength(" ")

    def _try_size(self, text: str, shape: "_Shape", size: int, hyphenate: bool = True) -> TextLayout | None:
        font = self.font(size)
        ascent, descent, space = self._metrics(size)
        glyph_h = ascent + descent
        line_h = max(glyph_h, int(round(size * self.line_spacing)))
        if glyph_h > shape.height:
            return None

        words = _hyphenate(text.split(), font, shape.max_width) if hyphenate else text.split()
        widths = [font.getlength(w) for w in words]
        max_lines = min(len(words), 1 + (shape.height - glyph_h) // line_h)

        best: tuple[float, TextLayout] | None = None
        for n in range(1, max_lines + 1):
            block_h = (n - 1) * line_h + glyph_h
            for nudge in (0.0, -0.25, 0.25, -0.5, 0.5):
                top = shape.cy - block_h / 2 + nudge * line_h
                slots = [shape.interval(top + i * line_h, top + i * line_h + glyph_h) for i in range(n)]
                if any(s is None for s in slots):
                    continue
                result = _break_lines(words, widths, space, [s[1] - s[0] for s in slots])
                if result is None:
                    continue
                cost, groups = result
                # Prefer layouts that fill the balloon vertically too (avoids one long thin line).
                cost += 0.15 * abs(nudge)
                if best is None or cost < best[0]:
                    lines = []
                    for i, (g, (l, r)) in enumerate(zip(groups, slots)):
                        s = " ".join(words[g[0]:g[1]])
                        w = font.getlength(s)
                        lines.append(Line(s, l + (r - l - w) / 2, top + i * line_h, w))
                    best = (cost, TextLayout(font_size=size, lines=lines))
                break  # first vertical position that works for this line count is good enough
        return best[1] if best else None

    def _overflow_layout(self, text: str, shape: "_Shape", size: int, used_short: bool) -> TextLayout:
        font = self.font(size)
        ascent, descent, _ = self._metrics(size)
        line_h = max(ascent + descent, int(round(size * self.line_spacing)))
        width = max(shape.max_width, size * 4)
        lines_txt, cur = [], ""
        for word in _hyphenate(text.split(), font, width):
            trial = f"{cur} {word}".strip()
            if cur and font.getlength(trial) > width:
                lines_txt.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            lines_txt.append(cur)
        top = shape.cy - (len(lines_txt) * line_h) / 2
        lines = []
        for i, s in enumerate(lines_txt):
            w = font.getlength(s)
            lines.append(Line(s, shape.cx - w / 2, top + i * line_h, w))
        return TextLayout(font_size=size, lines=lines, used_short=used_short, overflow=True)

    # ------------------------------------------------------------------ drawing
    def draw(self, image: Image.Image, layout: TextLayout, origin: tuple[int, int],
             fill: tuple[int, int, int], stroke: tuple[int, int, int] | None = None) -> None:
        draw = ImageDraw.Draw(image)
        font = self.font(layout.font_size)
        sw = max(2, int(round(layout.font_size * 0.14))) if stroke else 0
        ox, oy = origin
        for line in layout.lines:
            draw.text((ox + line.x, oy + line.y), line.text, font=font, fill=fill,
                      stroke_width=sw, stroke_fill=stroke)


# ---------------------------------------------------------------------------
# Balloon shape profile
# ---------------------------------------------------------------------------

class _Shape:
    """Row-by-row free horizontal span of a (padded) balloon mask."""

    def __init__(self, mask: np.ndarray):
        ys, xs = np.nonzero(mask)
        h = int(ys.max() - ys.min() + 1) if ys.size else 1
        w = int(xs.max() - xs.min() + 1) if xs.size else 1
        # Inner padding so letters never kiss the balloon outline.
        pad = max(2, int(round(min(w, h) * 0.08)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        inner = cv2.erode(mask.astype(np.uint8), k).astype(bool)
        if inner.sum() < 0.3 * mask.sum():
            inner = mask.astype(bool)

        ys, xs = np.nonzero(inner)
        self.cx = float(xs.mean()) if xs.size else mask.shape[1] / 2
        self.cy = float(ys.mean()) if ys.size else mask.shape[0] / 2
        self.top = int(ys.min()) if ys.size else 0
        self.bottom = int(ys.max()) + 1 if ys.size else mask.shape[0]
        self.height = self.bottom - self.top

        # For every row: the contiguous run containing the centre column (so text never jumps
        # across a gap such as a tail or an art element poking into the balloon).
        cxi = int(round(self.cx))
        self.left = np.full(mask.shape[0], np.nan)
        self.right = np.full(mask.shape[0], np.nan)
        for y in range(self.top, self.bottom):
            row = inner[y]
            if not row[cxi]:
                continue
            l = cxi
            while l > 0 and row[l - 1]:
                l -= 1
            r = cxi
            while r < len(row) - 1 and row[r + 1]:
                r += 1
            self.left[y], self.right[y] = l, r + 1
        spans = self.right - self.left
        self.max_width = float(np.nanmax(spans)) if np.isfinite(spans).any() else float(w)

    def interval(self, y0: float, y1: float) -> tuple[float, float] | None:
        a, b = int(math.floor(y0)), int(math.ceil(y1))
        if a < self.top or b > self.bottom:
            return None
        l, r = self.left[a:b], self.right[a:b]
        if np.isnan(l).any():
            return None
        lo, hi = float(np.max(l)), float(np.min(r))
        return (lo, hi) if hi - lo > 4 else None


# ---------------------------------------------------------------------------
# Line breaking
# ---------------------------------------------------------------------------

def _break_lines(words, widths, space, avail) -> tuple[float, list[tuple[int, int]]] | None:
    """Split ``words`` into exactly ``len(avail)`` lines, line i no wider than avail[i].

    Minimises squared unused width (balanced, centred look) plus penalties for weak breaks.
    """
    n, W = len(avail), len(words)
    if W < n:
        return None
    prefix = [0.0]
    for w in widths:
        prefix.append(prefix[-1] + w)

    def line_width(j, i):
        return prefix[i] - prefix[j] + space * (i - j - 1)

    INF = float("inf")
    cost = [[INF] * (W + 1) for _ in range(n + 1)]
    back = [[-1] * (W + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for k in range(1, n + 1):
        cap = avail[k - 1]
        last = k == n
        for i in range(k, W + 1):
            if last and i != W:
                continue
            for j in range(k - 1, i):
                if cost[k - 1][j] == INF:
                    continue
                lw = line_width(j, i)
                if lw > cap:
                    continue
                slack = (cap - lw) / cap
                c = slack * slack
                end_word = words[i - 1].lower().strip("\"'“”‘’(")
                if not last:
                    if end_word in WEAK_LINE_END:
                        c += 0.6
                    if re.search(r"[.!?…,;:—]$", words[i - 1]):
                        c -= 0.15  # breaking at punctuation reads naturally
                elif n > 1 and i - j == 1 and len(words[i - 1]) <= 4:
                    c += 0.4  # lonely short last word
                total = cost[k - 1][j] + c
                if total < cost[k][i]:
                    cost[k][i], back[k][i] = total, j
    if cost[n][W] == INF:
        return None
    groups, i = [], W
    for k in range(n, 0, -1):
        j = back[k][i]
        groups.append((j, i))
        i = j
    return cost[n][W] / n, groups[::-1]


def _hyphenate(words: list[str], font, max_width: float) -> list[str]:
    """Split words that cannot fit on any line, so they don't make the whole layout infeasible."""
    out = []
    for w in words:
        if font.getlength(w) <= max_width or len(w) < 4:
            out.append(w)
            continue
        chunk = ""
        for ch in w:
            if chunk and font.getlength(chunk + ch + "-") > max_width:
                out.append(chunk + "-")
                chunk = ch
            else:
                chunk += ch
        if chunk:
            out.append(chunk)
    return out


def _normalise(text: str | None, uppercase: bool) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text.upper() if uppercase else text

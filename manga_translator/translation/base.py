"""Shared pieces for every translation backend.

The detector already knows *where* the text is. The vision model is only asked *what it
says* and *how it reads in English* - it never has to produce coordinates, which is where
LLMs are weakest. Each detected region is drawn on the page with a numbered label
("set-of-marks" prompting) and the model answers per label.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Literal, Protocol

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from ..schema import Chapter, GlossaryEntry, Page
from ..utils import FONTS_DIR


# ---------------------------------------------------------------------------
# Response schema (used for structured outputs and for validating manual responses)
# ---------------------------------------------------------------------------

class FatalTranslationError(RuntimeError):
    """An error no other page would avoid (bad API key, unknown model): stop instead of retrying."""


class RegionTranslation(BaseModel):
    id: str = Field(description="The region label number as a string, e.g. \"3\".")
    original_text: str = Field(description="Exact Japanese transcription. Omit furigana (small ruby kana).")
    category: Literal["dialogue", "thought", "narration", "sfx", "sign", "noise"]
    translation: str = Field(description="Natural English for this region only. Empty string for noise.")
    short_translation: str = Field(
        description="Condensed version (about 60% of the length) that keeps the meaning, for when "
                    "the balloon is too small. Same as translation if it is already short."
    )


class GlossaryItem(BaseModel):
    source: str = Field(description="Japanese term or name as written.")
    target: str = Field(description="English rendering to use consistently.")
    note: str = Field(description="Very short context, e.g. 'character, female' or 'spell name'.")


class PageTranslation(BaseModel):
    regions: list[RegionTranslation]
    reading_order: list[str] = Field(description="All region ids in the order a Japanese reader reads them.")
    new_glossary: list[GlossaryItem] = Field(description="Names/terms first seen on this page. May be empty.")
    page_summary: str = Field(description="One or two sentences on what happens on this page, for continuity.")


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

@dataclass
class TranslationRequest:
    page: Page
    marked_image: Image.Image
    user_text: str


class Translator(Protocol):
    def translate_page(self, request: TranslationRequest) -> PageTranslation: ...


def build_request(chapter: Chapter, page: Page, source_image: Image.Image, page_number: int,
                  context_lines: int = 14) -> TranslationRequest:
    marked = draw_marks(source_image, page)
    regions_desc = "\n".join(
        f"- [{r.id}] " + ("text inside a speech balloon" if r.kind == "bubble" else "text over artwork (not in a balloon)")
        for r in page.regions
    )

    glossary = "\n".join(f"- {g.source} → {g.target}" + (f" ({g.note})" if g.note else "") for g in chapter.glossary)

    recent: list[str] = []
    summaries: list[str] = []
    for prev in chapter.pages:
        if prev.image == page.image:
            break
        if prev.summary:
            summaries.append(prev.summary)
        for r in prev.ordered_regions():
            if r.translation and r.category in ("dialogue", "thought", "narration"):
                recent.append(f"{r.original_text} → {r.translation}")
    recent = recent[-context_lines:]

    parts = [f"Page {page_number}. The marked regions on this page are:\n{regions_desc}"]
    if summaries:
        parts.append("Story so far (previous pages):\n" + "\n".join(f"- {s}" for s in summaries[-4:]))
    if recent:
        parts.append("Most recent lines from previous pages (for continuity of voice and terms):\n"
                     + "\n".join(recent))
    if glossary:
        parts.append("Glossary - keep these renderings consistent:\n" + glossary)
    parts.append("Return one entry in `regions` for every marked id.")
    return TranslationRequest(page=page, marked_image=marked, user_text="\n\n".join(parts))


def draw_marks(image: Image.Image, page: Page) -> Image.Image:
    """Draw each region's text box with a numbered badge so the model can refer to it."""
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    size = max(18, img.height // 55)
    try:
        font = ImageFont.truetype(str(FONTS_DIR / "ComicNeue-Bold.ttf"), size)
    except OSError:
        font = ImageFont.load_default()
    placed: list[tuple[float, float, float, float]] = []
    for r in page.regions:
        color = (230, 30, 30) if r.kind == "bubble" else (20, 140, 60)
        x1, y1, x2, y2 = r.text_box
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
    for r in page.regions:
        color = (230, 30, 30) if r.kind == "bubble" else (20, 140, 60)
        x1, y1 = r.text_box[:2]
        label = r.id
        bw, bh = d.textlength(label, font=font) + 8, size + 6
        bx, by = max(0, x1 - bw - 2), max(0, y1 - bh)
        # Every badge must stay visible: a hidden label means the model can't find that region
        # (and may invent text for it). Slide down / left until it doesn't overlap another badge.
        for _ in range(60):
            if not any(bx < px2 and bx + bw > px1 and by < py2 and by + bh > py1 for px1, py1, px2, py2 in placed):
                break
            by += bh + 2
            if by + bh > img.height:
                by, bx = max(0, y1 - bh), max(0, bx - bw - 2)
        placed.append((bx, by, bx + bw, by + bh))
        d.rectangle([bx, by, bx + bw, by + bh], fill=color, outline=(255, 255, 255))
        d.text((bx + 4, by + 2), label, font=font, fill=(255, 255, 255))
    return img


def encode_jpeg(image: Image.Image, max_side: int = 2000, quality: int = 90) -> str:
    img = image.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def apply_translation(chapter: Chapter, page: Page, result: PageTranslation) -> list[str]:
    """Copy a backend result into the chapter. Returns ids the model skipped (left untranslated)."""
    by_id = {r.id.strip().lstrip("[").rstrip("]"): r for r in result.regions}
    missing = []
    for region in page.regions:
        t = by_id.get(region.id)
        if t is None:
            missing.append(region.id)
            continue
        region.original_text = t.original_text
        region.category = t.category
        region.translation = t.translation.strip()
        region.short_translation = (t.short_translation or t.translation).strip()
    known = {r.id for r in page.regions}
    page.reading_order = [i for i in result.reading_order if i in known]
    page.summary = result.page_summary
    chapter.add_glossary([GlossaryEntry(g.source, g.target, g.note) for g in result.new_glossary])
    return missing

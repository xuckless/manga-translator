"""Chapter state, persisted as ``<chapter>/work/chapter.json``.

All coordinates are absolute pixels ``[x1, y1, x2, y2]`` in the source image.
Each stage fills in more fields, so any stage can be re-run on its own.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

Box = list[int]  # [x1, y1, x2, y2]

RegionKind = Literal["bubble", "free"]
Category = Literal["dialogue", "thought", "narration", "sfx", "sign", "noise"]


@dataclass
class Region:
    id: str
    kind: RegionKind                # "bubble" = text inside a speech bubble, "free" = text over artwork
    text_box: Box                   # tight box around the Japanese text
    bubble_box: Optional[Box] = None
    score: float = 0.0

    # Filled by the translation stage
    original_text: Optional[str] = None
    translation: Optional[str] = None
    short_translation: Optional[str] = None
    category: Optional[Category] = None

    @property
    def translated(self) -> bool:
        return self.translation is not None

    @property
    def should_render(self) -> bool:
        """Noise is never touched; SFX are left as drawn art by default (see render settings)."""
        return bool(self.translation and self.translation.strip()) and self.category != "noise"


@dataclass
class Page:
    image: str                      # file name relative to the chapter directory
    width: int
    height: int
    regions: list[Region] = field(default_factory=list)
    reading_order: list[str] = field(default_factory=list)
    summary: Optional[str] = None

    def region(self, rid: str) -> Optional[Region]:
        return next((r for r in self.regions if r.id == rid), None)

    def ordered_regions(self) -> list[Region]:
        if not self.reading_order:
            return list(self.regions)
        rank = {rid: i for i, rid in enumerate(self.reading_order)}
        return sorted(self.regions, key=lambda r: rank.get(r.id, len(rank)))


@dataclass
class GlossaryEntry:
    source: str
    target: str
    note: str = ""


@dataclass
class Chapter:
    pages: list[Page] = field(default_factory=list)
    glossary: list[GlossaryEntry] = field(default_factory=list)

    def page(self, image: str) -> Optional[Page]:
        return next((p for p in self.pages if p.image == image), None)

    def add_glossary(self, entries: list[GlossaryEntry]) -> None:
        known = {g.source for g in self.glossary}
        for e in entries:
            if e.source and e.target and e.source not in known:
                self.glossary.append(e)
                known.add(e.source)

    # ---- persistence -------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic: a crash mid-write never corrupts progress

    @classmethod
    def load(cls, path: Path) -> "Chapter":
        raw = json.loads(path.read_text(encoding="utf-8"))
        pages = [
            Page(**{**p, "regions": [Region(**r) for r in p.get("regions", [])]})
            for p in raw.get("pages", [])
        ]
        glossary = [GlossaryEntry(**g) for g in raw.get("glossary", [])]
        return cls(pages=pages, glossary=glossary)

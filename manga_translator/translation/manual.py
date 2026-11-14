"""File-based backend: hand each page to any model or person, then read the answers back.

``translate --backend manual`` writes, per page, into ``<chapter>/work/manual/``:

    <page>.marked.jpg     the page with numbered region boxes
    <page>.request.md     system prompt + page instructions + the JSON schema to answer with

and reads ``<page>.response.json`` (a ``PageTranslation``) wherever one exists. Useful for
running without an API key, for testing prompts with another model, or for human correction.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import PageTranslation, TranslationRequest
from .prompts import system_prompt


class ManualTranslator:
    def __init__(self, folder: Path, honorifics: str = "keep"):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.honorifics = honorifics

    def paths(self, image_name: str) -> tuple[Path, Path, Path]:
        stem = Path(image_name).stem
        return (self.folder / f"{stem}.marked.jpg", self.folder / f"{stem}.request.md",
                self.folder / f"{stem}.response.json")

    def translate_page(self, request: TranslationRequest) -> PageTranslation | None:
        marked, req, resp = self.paths(request.page.image)
        if resp.exists():
            return PageTranslation.model_validate_json(resp.read_text(encoding="utf-8"))
        request.marked_image.save(marked, quality=92)
        schema = json.dumps(PageTranslation.model_json_schema(), indent=2)
        req.write_text(
            "# System prompt\n\n" + system_prompt(self.honorifics)
            + f"\n\n# Page request\n\nImage: `{marked.name}`\n\n" + request.user_text
            + f"\n\n# Answer format\n\nWrite `{resp.name}` containing only JSON matching this schema:\n\n"
            + f"```json\n{schema}\n```\n",
            encoding="utf-8",
        )
        return None

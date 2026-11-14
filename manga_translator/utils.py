from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FONTS_DIR = PROJECT_ROOT / "assets" / "fonts"


def natural_key(s: str) -> list:
    """Sort 'page2' before 'page10' (plain sorting puts page10 first and scrambles chapter order)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_page_images(chapter_dir: Path) -> list[Path]:
    """Source pages live directly in the chapter dir; generated files live in subfolders."""
    files = [
        p for p in chapter_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(("debug_", "rendered_"))
    ]
    return sorted(files, key=lambda p: natural_key(p.name))


def box_area(b) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def box_intersection(a, b) -> float:
    return box_area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def box_iou(a, b) -> float:
    inter = box_intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union else 0.0


def box_union(a, b) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def clamp_box(b, w: int, h: int) -> list[int]:
    return [max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3]))]


def expand_box(b, px: float, w: int, h: int) -> list[int]:
    return clamp_box([b[0] - px, b[1] - px, b[2] + px, b[3] + px], w, h)

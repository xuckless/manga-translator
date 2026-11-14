"""Text + speech-bubble detection.

Uses ``ogkalu/comic-text-and-bubble-detector`` (RT-DETR-v2, Apache-2.0), trained on ~11k
manga/webtoon/comic pages. It predicts three classes:

    bubble       - the speech balloon itself
    text_bubble  - the text inside a balloon
    text_free    - text drawn directly over artwork (narration, signs, SFX)

Pixel-accurate masks are derived from these boxes in ``masks.py``.
"""

from __future__ import annotations

import logging
from functools import cached_property

from PIL import Image

from .schema import Region
from .utils import box_area, box_intersection, box_iou, box_union, expand_box

log = logging.getLogger(__name__)

DETECTOR_REPO = "ogkalu/comic-text-and-bubble-detector"


class TextDetector:
    def __init__(self, threshold: float = 0.35, device: str | None = None):
        self.threshold = threshold
        self.device = device or self._pick_device()

    @staticmethod
    def _pick_device() -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @cached_property
    def _model(self):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        processor = AutoImageProcessor.from_pretrained(DETECTOR_REPO)
        model = AutoModelForObjectDetection.from_pretrained(DETECTOR_REPO).eval().to(self.device)
        return processor, model

    def _raw_detections(self, image: Image.Image) -> list[tuple[str, float, list[int]]]:
        import torch

        processor, model = self._model
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=[image.size[::-1]]
        )[0]
        dets = []
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            name = model.config.id2label[int(label)]
            dets.append((name, float(score), [int(round(v)) for v in box.tolist()]))
        return dets

    def detect(self, image: Image.Image) -> list[Region]:
        w, h = image.size
        dets = _nms(self._raw_detections(image.convert("RGB")))

        bubbles = [d for d in dets if d[0] == "bubble"]
        texts = [d for d in dets if d[0] in ("text_bubble", "text_free")]

        # Attach each text box to the bubble that contains most of it.
        grouped: dict[int, list] = {}
        free: list = []
        for t in texts:
            best, best_cov = None, 0.0
            for i, b in enumerate(bubbles):
                cov = box_intersection(t[2], b[2]) / max(1.0, box_area(t[2]))
                if cov > best_cov:
                    best, best_cov = i, cov
            if best is not None and best_cov >= 0.5:
                grouped.setdefault(best, []).append(t)
            elif t[0] == "text_bubble":
                # Text the model thinks is in a balloon, but no balloon was found: treat the
                # surrounding area as the balloon and let masks.py find the real outline.
                bbox = expand_box(t[2], 0.25 * max(t[2][2] - t[2][0], t[2][3] - t[2][1]), w, h)
                grouped.setdefault(len(bubbles), []).append(t)
                bubbles.append(("bubble", t[1], bbox))
            else:
                free.append(t)

        regions: list[Region] = []
        for bi, members in grouped.items():
            tb = members[0][2]
            for m in members[1:]:
                tb = box_union(tb, m[2])
            regions.append(Region(id="", kind="bubble", text_box=tb, bubble_box=bubbles[bi][2],
                                  score=round(max(m[1] for m in members), 3)))
        for t in free:
            regions.append(Region(id="", kind="free", text_box=t[2], score=round(t[1], 3)))

        # Provisional Japanese reading order (top-to-bottom bands, right-to-left within a band).
        # The translator returns the true order; this just keeps IDs stable and readable.
        band = max(1, int(h * 0.12))
        regions.sort(key=lambda r: ((r.text_box[1] + r.text_box[3]) // 2 // band, -(r.text_box[0] + r.text_box[2])))
        for i, r in enumerate(regions, 1):
            r.id = str(i)
        return regions


def _nms(dets, iou_thr: float = 0.5, contain_thr: float = 0.8):
    """Drop duplicate boxes: same-class overlaps, and text boxes nested inside another text box
    (the detector sometimes returns a caption plus fragments of it). Nested text is absorbed by
    growing the outer box, so nothing gets lost."""
    kept: list = []
    for d in sorted(dets, key=lambda d: -d[1]):
        dup = None
        for i, k in enumerate(kept):
            same = k[0] == d[0]
            both_text = k[0] != "bubble" and d[0] != "bubble"
            if same and box_iou(k[2], d[2]) >= iou_thr:
                dup = i
                break
            if both_text:
                inter = box_intersection(k[2], d[2])
                if inter >= contain_thr * min(box_area(k[2]), box_area(d[2])):
                    dup = i
                    break
        if dup is None:
            kept.append(d)
        else:
            k = kept[dup]
            kept[dup] = (k[0], k[1], box_union(k[2], d[2]) if k[0] != "bubble" else k[2])
    return kept

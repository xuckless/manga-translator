"""Remove text from artwork.

* ``lama``   - LaMa (big-lama, ONNX export ``Carve/LaMa-ONNX``, Apache-2.0). Rebuilds screentone,
               hatching and backgrounds convincingly. Runs on CPU via onnxruntime.
* ``opencv`` - ``cv2.inpaint`` (Telea). No download, but smears textured backgrounds.

Balloon interiors never need this: they are flat paper and are just filled with their colour.
"""

from __future__ import annotations

import logging
from functools import cached_property

import cv2
import numpy as np

log = logging.getLogger(__name__)

LAMA_REPO = "Carve/LaMa-ONNX"
LAMA_FILE = "lama_fp32.onnx"
LAMA_SIZE = 512


class Inpainter:
    def __init__(self, method: str = "lama"):
        if method not in ("lama", "opencv"):
            raise ValueError(f"unknown inpaint method {method!r}")
        self.method = method

    @cached_property
    def _session(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(LAMA_REPO, LAMA_FILE)
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def inpaint(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return a copy of ``img_bgr`` with ``mask`` (bool HxW) filled in."""
        if not mask.any():
            return img_bgr.copy()
        if self.method == "opencv":
            return cv2.inpaint(img_bgr, mask.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)

        out = img_bgr.copy()
        # Inpaint each separate blob in its own context window so LaMa sees it at a useful scale.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)), connectivity=8
        )
        for i in range(1, n):
            x, y, w, h = stats[i, :4]
            blob = mask & (labels == i)
            out = self._lama_window(out, blob, (x, y, w, h))
        return out

    def _lama_window(self, img: np.ndarray, mask: np.ndarray, rect) -> np.ndarray:
        H, W = img.shape[:2]
        x, y, w, h = rect
        side = int(max(w, h) * 1.8) + 32           # give the model surrounding context
        side = max(side, 256)
        cx, cy = x + w // 2, y + h // 2
        x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
        x2, y2 = min(W, x1 + side), min(H, y1 + side)
        x1, y1 = max(0, x2 - side), max(0, y2 - side)

        crop = img[y1:y2, x1:x2]
        mcrop = mask[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_AREA)
        m = cv2.resize(mcrop.astype(np.uint8), (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_NEAREST)

        image_in = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        mask_in = (m > 0).astype(np.float32)[None, None]
        result = self._session.run(None, {"image": image_in, "mask": mask_in})[0][0]
        result = result.transpose(1, 2, 0)
        if result.max() <= 1.5:                    # some exports return [0,1], others [0,255]
            result = result * 255.0
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = cv2.resize(result, (cw, ch), interpolation=cv2.INTER_CUBIC)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Only replace masked pixels (feathered) so untouched art stays bit-identical.
        soft = cv2.GaussianBlur(mcrop.astype(np.float32), (5, 5), 0)
        soft = np.maximum(soft, mcrop.astype(np.float32))[..., None]
        blended = (result * soft + crop * (1 - soft)).astype(np.uint8)

        out = img.copy()
        out[y1:y2, x1:x2] = blended
        return out

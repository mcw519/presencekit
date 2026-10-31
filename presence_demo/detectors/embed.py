"""Face embedders: SFace on ONNX Runtime (default), SFace on OpenCV DNN, and the HSV-histogram fallback.

The SFace backends (``OrtSFace`` fp32/int8, ``OpenCVSFace``, ``TFLiteSFace`` fp16/int8/NPU) live in
``presence_demo.embed_backends`` and are re-exported here.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..embed_backends import OpenCVSFace, OrtSFace, TFLiteSFace  # noqa: F401  (re-exported)


class SFaceEmbedder:
    """v0.12 default: SFace on OpenCV DNN, raw (unnormalised) features. Kept for compatibility; the tracker
    normalises embeddings on intake, so it is interchangeable with :class:`OrtSFace`."""

    name = "SFace"
    accelerator = "cpu (OpenCV DNN)"

    def __init__(self, path):
        self.rec = cv2.FaceRecognizerSF.create(str(path), "")

    def embed(self, img, raw):
        return self.rec.feature(self.rec.alignCrop(img, raw)).ravel()

    def feature_aligned(self, crop):
        return self.rec.feature(crop).ravel()


class HistEmbedder:
    """Weak fallback: HSV histogram of face + upper body region. Tune cos thresholds higher."""

    name = "HSV-hist(fallback)"
    accelerator = "cpu (OpenCV)"

    def embed(self, img, raw):
        x, y, w, h = [int(v) for v in raw[:4]]
        H, W = img.shape[:2]
        crop = img[max(0, y) : min(H, y + int(h * 2.2)), max(0, x - w // 4) : min(W, x + w + w // 4)]
        if crop.size == 0:
            return np.zeros(96, np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 6], [0, 180, 0, 256]).ravel()
        return hist - hist.mean()

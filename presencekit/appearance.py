"""Lightweight, weight-free appearance descriptor: HSV histogram of the torso region.

Computed in memory on the decoded frame (PRIVACY: nothing is written to disk). No trained model, so
there is no model/data license to worry about. Similarity lives in :mod:`presencekit.kalman`.
"""

from __future__ import annotations

import cv2
import numpy as np

from .body import _kp

H_BINS, S_BINS = 12, 4


def torso_box(face_bbox=None, body=None) -> tuple | None:
    """Torso region (normalized): from shoulder/hip keypoints, else the body box, else below the face."""
    if body is not None:
        ls, rs = _kp(body, "l_shoulder", 0.3), _kp(body, "r_shoulder", 0.3)
        lh, rh = _kp(body, "l_hip", 0.3), _kp(body, "r_hip", 0.3)
        if ls and rs:
            x0, x1 = sorted((ls[0], rs[0]))
            pad = 0.1 * (x1 - x0)
            y0 = (ls[1] + rs[1]) / 2
            hips = [p[1] for p in (lh, rh) if p]
            y1 = sum(hips) / len(hips) if hips else y0 + 1.2 * (x1 - x0)
            if y1 - y0 > 0.01 and x1 - x0 > 0.01:
                return (x0 + pad, y0, x1 - pad, y1)
        bx0, by0, bx1, by1 = body.bbox
        bh, bw = by1 - by0, bx1 - bx0
        return (bx0 + 0.2 * bw, by0 + 0.25 * bh, bx1 - 0.2 * bw, by0 + 0.6 * bh)
    if face_bbox is not None:
        x0, y0, x1, y1 = face_bbox
        w, h = x1 - x0, y1 - y0
        cx = (x0 + x1) / 2
        return (cx - 0.8 * w, y1 + 0.3 * h, cx + 0.8 * w, y1 + 1.8 * h)
    return None


def descriptor(img: np.ndarray, box) -> np.ndarray | None:
    """L1-normalized H x S histogram of ``box`` (normalized coords) in ``img`` (BGR); None if too small."""
    if box is None:
        return None
    h, w = img.shape[:2]
    x0, y0 = int(max(0.0, box[0]) * w), int(max(0.0, box[1]) * h)
    x1, y1 = int(min(1.0, box[2]) * w), int(min(1.0, box[3]) * h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256]).ravel().astype(np.float32)
    s = float(hist.sum())
    return hist / s if s > 0 else None

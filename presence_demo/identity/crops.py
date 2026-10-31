"""In-memory face crops for re-ID (PRIVACY: decoded and re-encoded in memory only, never written to disk)."""

from __future__ import annotations

import base64

import cv2
import numpy as np

CROP_MAX = 320
THUMB_PX = 112


def face_crop(jpeg: bytes, bbox, margin: float = 0.15) -> bytes | None:
    """Decode (in memory), crop a normalized bbox with margin, pad to square, scale to CROP_MAX, re-encode JPEG."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox[:4]
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    a, b = int(max(0, (x0 - mx) * w)), int(max(0, (y0 - my) * h))
    c, d = int(min(w, (x1 + mx) * w)), int(min(h, (y1 + my) * h))
    if c - a < 8 or d - b < 8:
        return None
    crop = img[b:d, a:c]
    # square crop scaled to exactly CROP_MAX (the API rejects small images: "Invalid image dimensions")
    side = max(crop.shape[:2])
    pad = np.zeros((side, side, 3), np.uint8)
    pad[(side - crop.shape[0]) // 2 :][: crop.shape[0], (side - crop.shape[1]) // 2 :][:, : crop.shape[1]] = crop
    interp = cv2.INTER_AREA if side > CROP_MAX else cv2.INTER_CUBIC
    crop = cv2.resize(pad, (CROP_MAX, CROP_MAX), interpolation=interp)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else None


def thumb(jpeg: bytes | None, px: int = THUMB_PX) -> str | None:
    """Small display thumbnail (base64 JPEG) for this connection's browser only (``reid_show_crops``).
    Never written to disk, logs or diagnostic events."""
    if not jpeg:
        return None
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def face_quality(t: dict) -> float:
    """Quality of a track's face for re-ID (0 = unusable). Uses the tracker snapshot only (no pixels)."""
    if not t.get("visible") or t.get("has_face") is False or t.get("visible_kind") == "predicted":
        return 0.0
    b = t.get("bbox") or [0, 0, 0, 0]
    w = b[2] - b[0]
    q = w * float(t.get("conf") or 0)
    if t.get("facing") == "away":
        return 0.0
    if t.get("facing") == "side":
        q *= 0.5
    return q

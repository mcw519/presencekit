"""CC0 face crops for embedding calibration/evaluation — fetched and kept in memory only.

Source: Cleveland Museum of Art Open Access API (all results filtered with ``cc0=1``; the museum
releases these images under CC0 1.0: https://www.clevelandart.org/open-access). 19th/early-20th
century photographic portraits. Nothing is written to disk: images are downloaded into memory,
faces are detected and aligned (YuNet + SFace alignCrop) and only the 112x112 aligned crops are
returned (as numpy arrays) to the caller.
"""

from __future__ import annotations

import json
import urllib.request

import cv2
import numpy as np

API = (
    "https://openaccess-api.clevelandart.org/api/artworks/"
    "?q={q}&type=Photograph&cc0=1&has_image=1&limit={n}&skip={skip}"
)


def _get(url: str, timeout: float = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "presencekit-eval/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_aligned_faces(
    yunet_path, sface_path, n: int = 40, query: str = "portrait", min_face: int = 60, skip: int = 0
):
    """Return [(artwork_id, title, aligned_bgr_112x112)] for images that contain exactly one clear face."""
    det = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), 0.8, 0.3, 50)
    rec = cv2.FaceRecognizerSF.create(str(sface_path), "")
    items = json.loads(_get(API.format(q=query, n=n * 3, skip=skip)))["data"]
    out = []
    for it in items:
        url = ((it.get("images") or {}).get("web") or {}).get("url")
        if not url or it.get("share_license_status", "CC0") != "CC0":
            continue
        try:
            img = cv2.imdecode(np.frombuffer(_get(url), np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            continue
        if img is None:
            continue
        h, w = img.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        if faces is None or len(faces) != 1 or faces[0][2] < min_face:
            continue
        out.append((it["id"], it.get("title", ""), rec.alignCrop(img, faces[0])))
        del img
        if len(out) >= n:
            break
    return out


def augment(crop: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Photometric/geometric variants of one aligned crop (same identity)."""
    out = []
    out.append(cv2.convertScaleAbs(crop, alpha=1.0, beta=35))  # brighter
    out.append(cv2.convertScaleAbs(crop, alpha=0.75, beta=-10))  # darker / lower contrast
    out.append(cv2.GaussianBlur(crop, (5, 5), 1.2))
    m = cv2.getRotationMatrix2D((56, 56), float(rng.uniform(-8, 8)), float(rng.uniform(0.93, 1.07)))
    out.append(cv2.warpAffine(crop, m, (112, 112), borderMode=cv2.BORDER_REFLECT))
    ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 35])  # in-memory JPEG artefacts
    out.append(cv2.imdecode(enc, cv2.IMREAD_COLOR))
    noisy = crop.astype(np.int16) + rng.normal(0, 6, crop.shape).astype(np.int16)
    out.append(np.clip(noisy, 0, 255).astype(np.uint8))
    return out

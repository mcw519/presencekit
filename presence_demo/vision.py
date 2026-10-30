"""Pluggable Detector / Embedder. Default: OpenCV YuNet + SFace (ONNX, CPU).
Fallback (no model files): Haar cascade (bundled with OpenCV) + HSV color-histogram embedding."""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .core.tracks import Detection
from .detectors.base import Embedder, FaceDetector, PoseDetector  # noqa: F401  (re-exported)
from .detectors.embed import HistEmbedder, SFaceEmbedder
from .detectors.face import HaarDetector, YuNetDetector
from .detectors.pose import COCO_KP, COCO_KP_IDX, MoveNetMultiPoseDetector  # noqa: F401  (re-exported)
from .intent import frontal_from_landmarks

log = logging.getLogger(__name__)
MODEL_DIR = Path(os.environ.get("PRESENCE_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
YUNET = ("face_detection_yunet_2023mar.onnx", ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE = (
    "face_recognition_sface_2021dec.onnx",
    ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)
# official opencv_zoo INT8 quantization of the same model (sha256 = git-LFS oid in opencv_zoo)
SFACE_INT8 = (
    "face_recognition_sface_2021dec_int8.onnx",
    ZOO + "face_recognition_sface/face_recognition_sface_2021dec_int8.onnx",
    "2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a",
)
# produced locally by scripts/export_sface_tflite.py (not downloadable; not committed)
SFACE_TFLITE = {
    "int8": "face_recognition_sface_2021dec_int8.tflite",
    "fp16": "face_recognition_sface_2021dec_fp16.tflite",
}


# MoveNet.MultiPose Lightning (Google, Apache-2.0), TFLite float16, run with ai-edge-litert (LiteRT).
# Source / license: https://www.kaggle.com/models/google/movenet  (variation tfLite/multipose-lightning-tflite-float16)
MOVENET = (
    "movenet_multipose_lightning_fp16.tflite",
    "https://www.kaggle.com/api/v1/models/google/movenet/tfLite/multipose-lightning-tflite-float16/1/download",
)

# v0.12 name of the face detector protocol
Detector = FaceDetector


def ensure_movenet() -> Path | None:
    """Download the Kaggle tarball (contains a single .tflite) once, extracting in memory."""
    import io
    import tarfile

    fname, url = MOVENET
    p = MODEL_DIR / fname
    if p.exists() and p.stat().st_size > 1000:
        return p
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            m = next(m for m in tf.getmembers() if m.name.endswith(".tflite"))
            p.write_bytes(tf.extractfile(m).read())
        return p
    except Exception as e:  # noqa: BLE001
        log.warning("pose model download failed: %s", e)
        return None


def load_pose_detector(allow_download: bool = True) -> PoseDetector | None:
    """Load the default pose detector, downloading the model on first use. None if unavailable."""
    p = ensure_movenet() if allow_download else (MODEL_DIR / MOVENET[0])
    if not p or not Path(p).exists():
        return None
    try:
        return MoveNetMultiPoseDetector(p)
    except Exception as e:  # noqa: BLE001
        log.warning("pose model failed to load: %s", e)
        return None


def _sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_verified: set = set()


def _local(spec) -> Path | None:
    """Already-downloaded model file (no network)."""
    p = MODEL_DIR / spec[0]
    return p if p.exists() and p.stat().st_size > 1000 else None


def ensure_model(fname, url, sha256: str | None = None) -> Path | None:
    """Return the model path, downloading it on first use; verify sha256 when given."""
    p = MODEL_DIR / fname
    if p.exists() and p.stat().st_size > 1000:
        if sha256 and p not in _verified:
            if _sha256(p) != sha256:
                log.warning("checksum mismatch for %s, re-downloading", fname)
                p.unlink()
                return ensure_model(fname, url, sha256)
            _verified.add(p)
        return p
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        tmp = p.with_suffix(p.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        if sha256 and _sha256(tmp) != sha256:
            tmp.unlink()
            raise ValueError(f"sha256 mismatch for {fname}")
        tmp.replace(p)
        _verified.add(p)
        return p
    except Exception as e:  # noqa: BLE001
        log.warning("model download failed (%s): %s", fname, e)
        if p.exists():
            p.unlink()
        return None


def decode_jpeg(data: bytes) -> tuple[np.ndarray | None, float]:
    """Decode JPEG bytes in memory (PRIVACY: never written to disk). Returns (image, ms)."""
    t0 = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img, (time.perf_counter() - t0) * 1000


class VisionPipeline:
    def __init__(self, detector: Detector | None = None, embedder: Embedder | None = None, allow_download: bool = True):
        if detector is None:
            p = ensure_model(*YUNET) if allow_download else _local(YUNET)
            detector = YuNetDetector(p) if p else HaarDetector()
        if embedder is None:
            # PRESENCE_FACE_EMBEDDER=hist disables SFace (see docs/licenses.md for the training-data caveat).
            want_sface = os.environ.get("PRESENCE_FACE_EMBEDDER", "sface").lower() != "hist"
            p = (ensure_model(*SFACE) if allow_download else _local(SFACE)) if want_sface else None
            embedder = SFaceEmbedder(p) if p and isinstance(detector, YuNetDetector) else HistEmbedder()
        self.detector, self.embedder = detector, embedder
        self.allow_download = allow_download
        self._embedders: dict = {}
        self._backend_key = ("fp32", "", "int8")
        self.embed_status = {
            "requested": "fp32",
            "active": "fp32" if isinstance(embedder, SFaceEmbedder) else "hist",
            "model": embedder.name,
            "accelerator": "cpu (OpenCV DNN)",
            "note": "",
        }

    def select_embedder(self, backend: str = "fp32", delegate_path: str = "", precision: str = "int8") -> dict:
        """Switch the SFace backend (fp32 | int8 | npu). Unavailable backends fall back to fp32 with a note.

        Only applies when the default SFace embedder is in use (not the HSV-histogram fallback)."""
        key = (backend, delegate_path or "", precision)
        if key == self._backend_key or not isinstance(self._embedders.get("fp32", self.embedder), SFaceEmbedder):
            return self.embed_status
        if "fp32" not in self._embedders:
            if not isinstance(self.embedder, SFaceEmbedder):
                return self.embed_status
            self._embedders["fp32"] = self.embedder
        from .embed_backends import OpenCVSFace, TFLiteSFace

        note, emb, accel = "", None, "cpu (OpenCV DNN)"
        try:
            if backend == "int8":
                emb = self._embedders.get("int8")
                if emb is None:
                    p = ensure_model(*SFACE_INT8) if self.allow_download else _local(SFACE_INT8)
                    if not p or not Path(p).exists():
                        raise FileNotFoundError(SFACE_INT8[0])
                    emb = self._embedders["int8"] = OpenCVSFace(p, "int8")
            elif backend == "npu":
                emb = self._embedders.get(key)
                if emb is None:
                    p = MODEL_DIR / SFACE_TFLITE.get(precision, SFACE_TFLITE["int8"])
                    if not p.exists():
                        raise FileNotFoundError(f"{p.name} (run scripts/export_sface_tflite.py)")
                    emb = TFLiteSFace(p, delegate_path or None)
                    if emb.delegate_error or not delegate_path:
                        why = emb.delegate_error or "no npu_delegate_path set"
                        note = f"NPU delegate not active ({why}); running on CPU"
                        fp16 = MODEL_DIR / SFACE_TFLITE["fp16"]
                        if precision == "int8" and fp16.exists():
                            # LiteRT's CPU kernels are much faster for fp16 than for this int8 graph on x86
                            emb = TFLiteSFace(fp16)
                            note += " with the fp16 model (faster than int8 on CPU)"
                    emb.note = note
                    self._embedders[key] = emb
                note = getattr(emb, "note", "")
                accel = emb.accelerator
            else:
                emb = self._embedders["fp32"]
        except Exception as e:  # noqa: BLE001
            log.warning("embedding backend %s unavailable, using fp32: %s", backend, e)
            emb, note = self._embedders["fp32"], f"{backend} unavailable ({e}); using fp32"
        if emb is self._embedders["fp32"]:
            accel = "cpu (OpenCV DNN)"
        self.embedder = emb
        self._backend_key = key
        active = "fp32" if emb is self._embedders["fp32"] else backend
        self.embed_status = {
            "requested": backend,
            "active": active,
            "model": emb.name,
            "accelerator": accel,
            "note": note,
        }
        if note:
            log.info("embedding backend: %s", note)
        return self.embed_status

    @property
    def name(self):
        return f"{self.detector.name} + {self.embedder.name}"

    def process(self, img: np.ndarray) -> list[Detection]:
        return self.process_timed(img)[0]

    def process_timed(self, img: np.ndarray) -> tuple[list[Detection], dict]:
        """Returns detections and timings in ms: decode/detect/embed."""
        h, w = img.shape[:2]
        out = []
        t0 = time.perf_counter()
        raws = self.detector.detect(img)
        t1 = time.perf_counter()
        for raw, conf in raws:
            x, y, bw, bh = [float(v) for v in raw[:4]]
            bbox = (max(0, x / w), max(0, y / h), min(1, (x + bw) / w), min(1, (y + bh) / h))
            try:
                e = self.embedder.embed(img, raw)
            except Exception:  # noqa: BLE001
                e = None
            out.append(Detection(bbox, conf, e, frontal=frontal_from_landmarks(raw)))
        t2 = time.perf_counter()
        return out, {"detect": (t1 - t0) * 1000, "embed": (t2 - t1) * 1000}

    def process_jpeg_timed(self, data: bytes) -> tuple[list[Detection], dict]:
        """Decode + process a JPEG entirely in memory.

        PRIVACY GUARD: frames must never touch the disk. Do not add cv2.imwrite, debug dumps,
        temp files or logging of `data` here; the image buffer is dropped when this returns.
        tests/test_privacy.py enforces this.
        """
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        dec = (time.perf_counter() - t0) * 1000
        if img is None:
            return [], {"decode": dec, "detect": 0.0, "embed": 0.0, "w": 0, "h": 0}
        dets, t = self.process_timed(img)
        return dets, {"decode": dec, **t, "w": img.shape[1], "h": img.shape[0]}

    def info(self) -> dict:
        return {
            "pipeline": self.name,
            "opencv": cv2.__version__,
            "cv_threads": cv2.getNumThreads(),
            "dnn_backend": "OpenCV (CPU)",
            "embed_backend": dict(self.embed_status),
            "dnn_target": "CPU",
            "cpu_count": os.cpu_count(),
        }

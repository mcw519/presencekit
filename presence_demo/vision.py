"""Vision pipeline: face detector + embedder, the pose loader and in-memory JPEG decode.

Default (v0.14): YuNet + SFace on ONNX Runtime (:mod:`presence_demo.inference`). Fallbacks, in order: YuNet on
OpenCV DNN when only the fixed-size opencv_zoo file is present, then the Haar cascade + HSV histogram (no model
files). Model files come from the model store (``presence_demo.inference.ModelStore``; ``PRESENCE_MODEL_DIR``).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .core.tracks import Detection
from .detectors.base import Embedder, FaceDetector, PoseDetector  # noqa: F401  (re-exported)
from .detectors.embed import HistEmbedder, SFaceEmbedder
from .detectors.face import HaarDetector, OrtYuNetDetector, YuNetDetector
from .detectors.pose import COCO_KP, COCO_KP_IDX, MoveNetMultiPoseDetector, OrtMoveNetDetector  # noqa: F401
from .embed_backends import OrtSFace
from .inference import CATALOG, ModelSpec, OrtOptions, default_root, default_store
from .intent import frontal_from_landmarks

log = logging.getLogger(__name__)
MODEL_DIR = default_root()


def _spec(key: str) -> tuple:
    s = CATALOG[key]
    return (s.filename, s.url, s.sha256) if s.sha256 else (s.filename, s.url)


# v0.12/v0.13 tuples (filename, url[, sha256]); the catalogue in presence_demo.inference.models is the source.
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
YUNET_ORT = _spec("yunet")  # dynamic input size, for ONNX Runtime
YUNET = _spec("yunet-opencv")  # fixed 640x640 export, OpenCV DNN fallback
SFACE = _spec("sface")
SFACE_INT8 = _spec("sface-int8")
SFACE_TFLITE = {"int8": CATALOG["sface-tflite-int8"].filename, "fp16": CATALOG["sface-tflite-fp16"].filename}
MOVENET_ONNX = CATALOG["movenet"].filename  # produced by scripts/export_movenet_onnx.py
MOVENET = _spec("movenet-tflite")

# v0.12 name of the face detector protocol
Detector = FaceDetector
SFACE_EMBEDDERS = (OrtSFace, SFaceEmbedder)  # embedders that select_embedder may swap


def _catalog_spec(fname: str, url: str | None = None, sha256: str | None = None) -> ModelSpec:
    """The catalogue entry for ``fname``, or an ad-hoc spec for a file the catalogue does not know."""
    known = next((s for s in CATALOG.values() if s.filename == fname), None)
    return known or ModelSpec(fname, fname, url, sha256)


def _local(spec) -> Path | None:
    """Already-present model file (no network)."""
    return default_store().local(_catalog_spec(*spec))


def ensure_model(fname, url=None, sha256: str | None = None) -> Path | None:
    """Return the model path, downloading it on first use (sha256-verified). None when unavailable."""
    return default_store().ensure(_catalog_spec(fname, url, sha256))


def ensure_movenet() -> Path | None:
    """The MoveNet TFLite model (Kaggle tarball, extracted in memory), downloaded once."""
    return default_store().ensure("movenet-tflite")


def load_pose_detector(allow_download: bool = True, options: OrtOptions | None = None) -> PoseDetector | None:
    """MoveNet on ONNX Runtime when the exported ONNX file is present, else the TFLite model on LiteRT (downloaded
    on first use; needs ``ai-edge-litert``). None if neither is available."""
    store = default_store()
    p = store.local("movenet")
    if p:
        try:
            return OrtMoveNetDetector(p, options=options or OrtOptions.from_env())
        except Exception as e:  # noqa: BLE001
            log.warning("MoveNet ONNX failed to load, trying LiteRT: %s", e)
    try:
        import ai_edge_litert  # noqa: F401
    except ImportError:
        log.warning("pose unavailable: run scripts/export_movenet_onnx.py, or install the litert extra")
        return None
    p = ensure_movenet() if allow_download else store.local("movenet-tflite")
    if not p:
        return None
    try:
        return MoveNetMultiPoseDetector(p)
    except Exception as e:  # noqa: BLE001
        log.warning("pose model failed to load: %s", e)
        return None


def decode_jpeg(data: bytes) -> tuple[np.ndarray | None, float]:
    """Decode JPEG bytes in memory (PRIVACY: never written to disk). Returns (image, ms)."""
    t0 = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img, (time.perf_counter() - t0) * 1000


class VisionPipeline:
    """Face detector + embedder (+ SFace backend switching). ``options`` decides how the ONNX models run
    (default: ``OrtOptions.from_env()``)."""

    def __init__(
        self,
        detector: Detector | None = None,
        embedder: Embedder | None = None,
        allow_download: bool = True,
        options: OrtOptions | None = None,
    ):
        self.options = options or OrtOptions.from_env()
        self.allow_download = allow_download
        notes = []
        if detector is None:
            detector = self._default_detector(notes)
        if embedder is None:
            embedder = self._default_embedder(detector, notes)
        self.detector, self.embedder = detector, embedder
        self._embedders: dict = {}
        self._backend_key = ("fp32", "", "int8")
        self.embed_status = {
            "requested": "fp32",
            "active": "fp32" if isinstance(embedder, SFACE_EMBEDDERS) else "hist",
            "model": embedder.name,
            "accelerator": getattr(embedder, "accelerator", "cpu"),
            "note": "; ".join(notes),
        }

    def _model(self, spec) -> Path | None:
        return ensure_model(*spec) if self.allow_download else _local(spec)

    def _default_detector(self, notes: list) -> Detector:
        p = self._model(YUNET_ORT)
        if p:
            try:
                return OrtYuNetDetector(p, options=self.options)
            except Exception as e:  # noqa: BLE001
                log.warning("YuNet on onnxruntime failed: %s", e)
                notes.append(f"YuNet ONNX Runtime unavailable ({type(e).__name__})")
        p = _local(YUNET)
        return YuNetDetector(p) if p else HaarDetector()

    def _default_embedder(self, detector: Detector, notes: list) -> Embedder:
        # PRESENCE_FACE_EMBEDDER=hist disables SFace (see docs/licenses.md for the training-data caveat).
        if os.environ.get("PRESENCE_FACE_EMBEDDER", "sface").lower() == "hist":
            return HistEmbedder()
        p = self._model(SFACE)
        if not p or not getattr(detector, "landmarks", False):
            return HistEmbedder()
        try:
            return OrtSFace(p, "fp32", self.options)
        except Exception as e:  # noqa: BLE001
            log.warning("SFace on onnxruntime failed, using OpenCV DNN: %s", e)
            notes.append(f"SFace ONNX Runtime unavailable ({type(e).__name__}); OpenCV DNN")
            return SFaceEmbedder(p)

    def prepare_backend(self, backend: str) -> None:
        """Download the model file ``backend`` needs (no switching; safe without the vision lock)."""
        if backend == "int8" and self.allow_download and isinstance(self.embedder, SFACE_EMBEDDERS):
            ensure_model(*SFACE_INT8)

    def select_embedder(self, backend: str = "fp32", delegate_path: str = "", precision: str = "int8") -> dict:
        """Switch the SFace backend (fp32 | int8 | npu). Unavailable backends fall back to fp32 with a note.

        Only applies when the default SFace embedder is in use (not the HSV-histogram fallback)."""
        key = (backend, delegate_path or "", precision)
        if key == self._backend_key or not isinstance(self._embedders.get("fp32", self.embedder), SFACE_EMBEDDERS):
            return self.embed_status
        if "fp32" not in self._embedders:
            self._embedders["fp32"] = self.embedder
        from .embed_backends import TFLiteSFace

        note, emb = "", None
        try:
            if backend == "int8":
                emb = self._embedders.get("int8")
                if emb is None:
                    p = self._model(SFACE_INT8)
                    if not p or not Path(p).exists():
                        raise FileNotFoundError(SFACE_INT8[0])
                    emb = self._embedders["int8"] = OrtSFace(p, "int8", self.options)
            elif backend == "npu":
                emb = self._embedders.get(key)
                if emb is None:
                    p = default_store().root / SFACE_TFLITE.get(precision, SFACE_TFLITE["int8"])
                    if not p.exists():
                        raise FileNotFoundError(f"{p.name} (run scripts/export_sface_tflite.py)")
                    emb = TFLiteSFace(p, delegate_path or None)
                    if emb.delegate_error or not delegate_path:
                        why = emb.delegate_error or "no npu_delegate_path set"
                        note = f"NPU delegate not active ({why}); running on CPU"
                        fp16 = p.with_name(SFACE_TFLITE["fp16"])
                        if precision == "int8" and fp16.exists():
                            # LiteRT's CPU kernels are much faster for fp16 than for this int8 graph on x86
                            emb = TFLiteSFace(fp16)
                            note += " with the fp16 model (faster than int8 on CPU)"
                    emb.note = note
                    self._embedders[key] = emb
                note = getattr(emb, "note", "")
            else:
                emb = self._embedders["fp32"]
        except Exception as e:  # noqa: BLE001
            log.warning("embedding backend %s unavailable, using fp32: %s", backend, e)
            emb, note = self._embedders["fp32"], f"{backend} unavailable ({e}); using fp32"
        self.embedder = emb
        self._backend_key = key
        self.embed_status = {
            "requested": backend,
            "active": "fp32" if emb is self._embedders["fp32"] else backend,
            "model": emb.name,
            "accelerator": getattr(emb, "accelerator", "cpu"),
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
        img, dec = decode_jpeg(data)
        if img is None:
            return [], {"decode": dec, "detect": 0.0, "embed": 0.0, "w": 0, "h": 0}
        dets, t = self.process_timed(img)
        return dets, {"decode": dec, **t, "w": img.shape[1], "h": img.shape[0]}

    def info(self) -> dict:
        import onnxruntime as ort

        accel = getattr(self.detector, "accelerator", "cpu")
        return {
            "pipeline": self.name,
            "opencv": cv2.__version__,
            "cv_threads": cv2.getNumThreads(),
            "onnxruntime": ort.__version__,
            "ort_providers": list(self.options.providers),
            "dnn_backend": accel,
            "embed_backend": dict(self.embed_status),
            "dnn_target": accel.split(" ")[0].upper(),
            "cpu_count": os.cpu_count(),
        }

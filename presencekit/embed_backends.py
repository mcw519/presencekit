"""SFace embedding backends (selected with ``Config.embed_backend``).

* ``fp32`` – opencv_zoo ``face_recognition_sface_2021dec.onnx`` on ONNX Runtime (:class:`OrtSFace`, default).
* ``int8`` – opencv_zoo's official ``face_recognition_sface_2021dec_int8.onnx`` on ONNX Runtime.
* ``npu``  – SFace exported to TFLite (``scripts/export_sface_tflite.py``; full-INT8 or fp16), run with
  LiteRT (ai-edge-litert). An optional external delegate (``npu_delegate_path``: NNAPI / MediaTek
  NeuroPilot shared library on Android) is tried first; otherwise LiteRT's default CPU path (XNNPACK).
  The accelerator that is *actually* active is reported in :attr:`TFLiteSFace.accelerator`.

All backends take the full frame plus a YuNet row (5 landmarks) and return an L2-normalised 128-d
float32 vector. :class:`OpenCVSFace` (OpenCV DNN) is kept as a fallback and for comparisons. Frames and crops stay in
memory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .inference import OrtOptions, load_model

log = logging.getLogger(__name__)

# SFace / ArcFace 112x112 5-point template (same as OpenCV FaceRecognizerSF::alignCrop)
TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    np.float64,
)


def similarity_transform(src: np.ndarray, dst: np.ndarray = TEMPLATE) -> np.ndarray:
    """Umeyama least-squares similarity transform (2x3) mapping ``src`` points onto ``dst``."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    cov = d0.T @ s0 / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    r = u @ np.diag(d) @ vt
    var = (s0**2).sum() / len(src)
    scale = (s * d).sum() / var if var > 0 else 1.0
    m = np.zeros((2, 3))
    m[:, :2] = scale * r
    m[:, 2] = md - scale * r @ ms
    return m


def align_crop(img: np.ndarray, raw) -> np.ndarray:
    """112x112 aligned face crop from a YuNet row (x, y, w, h, 5 x landmark xy, score)."""
    pts = np.asarray(raw[4:14], np.float64).reshape(5, 2)
    return cv2.warpAffine(img, similarity_transform(pts), (112, 112))


def l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def sface_blob(crop_bgr: np.ndarray) -> np.ndarray:
    """OpenCV ``FaceRecognizerSF.feature`` input: aligned 112x112 crop -> RGB float 0..255, NCHW."""
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None]


class OrtSFace:
    """SFace (fp32, or the official int8 quantization) on ONNX Runtime with OpenCV ``FaceRecognizerSF``'s input
    blob. Stateless per call, so it is safe to share between threads."""

    def __init__(self, path, label: str = "fp32", options: OrtOptions | None = None):
        self.model = load_model(path, options)
        self.label = label
        self.name = f"SFace {label}"
        self._input = self.model.input_names[0]

    @property
    def accelerator(self) -> str:
        return self.model.accelerator

    def feature_aligned(self, crop: np.ndarray) -> np.ndarray:
        return l2(self.model.run({self._input: sface_blob(crop)})[0])

    def embed(self, img, raw):
        return self.feature_aligned(align_crop(img, raw))


class OpenCVSFace:
    """SFace ONNX (fp32 or the official int8 quantization) on OpenCV DNN, CPU."""

    def __init__(self, path, label: str = "fp32"):
        self.rec = cv2.FaceRecognizerSF.create(str(path), "")
        self.label = label
        self.name = f"SFace {label}"
        self.accelerator = "cpu (OpenCV DNN)"

    def embed(self, img, raw):
        return l2(self.rec.feature(self.rec.alignCrop(img, raw)))

    def feature_aligned(self, crop):
        return l2(self.rec.feature(crop))


class TFLiteSFace:
    """SFace TFLite (int8 or fp16) via LiteRT; optional external (NPU) delegate with CPU fallback."""

    def __init__(self, path, delegate_path: str | None = None, threads: int = 4, delegate_options: dict | None = None):
        from ai_edge_litert.interpreter import Interpreter, load_delegate

        self.path = Path(path)
        self.delegate_error = None
        delegates = []
        if delegate_path:
            try:
                delegates = [load_delegate(str(delegate_path), delegate_options or {})]
            except Exception as e:  # noqa: BLE001 - missing lib, wrong ABI, unsupported device...
                self.delegate_error = f"{type(e).__name__}: {e}"
                log.warning("NPU delegate %s unavailable, falling back to CPU: %s", delegate_path, e)
        try:
            self.it = Interpreter(str(path), num_threads=threads, experimental_delegates=delegates or None)
            self.it.allocate_tensors()
        except Exception as e:  # noqa: BLE001 - delegate accepted but graph failed to prepare
            if not delegates:
                raise
            self.delegate_error = f"{type(e).__name__}: {e}"
            log.warning("NPU delegate failed to prepare the graph, falling back to CPU: %s", e)
            delegates = []
            self.it = Interpreter(str(path), num_threads=threads)
            self.it.allocate_tensors()
        self.accelerator = f"delegate:{Path(delegate_path).name}" if delegates else "cpu (LiteRT XNNPACK)"
        i, o = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.inp, self.out = i["index"], o["index"]
        self.in_dtype, self.out_dtype = i["dtype"], o["dtype"]
        self.in_q, self.out_q = i["quantization"], o["quantization"]
        self.input_shape = tuple(i["shape"])
        self.precision = "int8" if self.in_dtype == np.int8 else ("fp16" if "fp16" in self.path.name else "fp32")
        self.name = f"SFace TFLite {self.precision}"

    def feature_aligned(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)[None]  # NHWC RGB 0..255
        if self.input_shape[1] == 3:  # NCHW export
            x = x.transpose(0, 3, 1, 2)
        if self.in_dtype == np.int8:
            s, z = self.in_q
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        self.it.set_tensor(self.inp, x)
        self.it.invoke()
        y = self.it.get_tensor(self.out)
        if self.out_dtype == np.int8:
            s, z = self.out_q
            y = (y.astype(np.float32) - z) * s
        return l2(y)

    def embed(self, img, raw):
        return self.feature_aligned(align_crop(img, raw))

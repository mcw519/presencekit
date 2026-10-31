"""Reproducible export of SFace (opencv_zoo, ONNX) to TFLite for NPU/NNAPI delegates.

Produces in models/:
  * face_recognition_sface_2021dec_fp16.tflite  – float16 weights, float32 IO (GPU/NNAPI fp16 path)
  * face_recognition_sface_2021dec_int8.tflite  – full-integer INT8 (int8 input/output), static shape
    [1, 112, 112, 3] NHWC (RGB, 0..255 mapped to int8) – the NPU-friendly variant (NeuroPilot / NNAPI)

Toolchain (all permissive licences; see docs/licenses.md): onnx2tf (MIT), TensorFlow (Apache-2.0),
onnx / onnxruntime (Apache-2.0 / MIT), onnx-graphsurgeon & onnxsim (Apache-2.0), sng4onnx (MIT).
TensorFlow is heavy and not a runtime dependency, so use a separate environment:

    uv venv -p 3.12 /tmp/.venv-sface-export
    VIRTUAL_ENV=/tmp/.venv-sface-export uv pip install onnx2tf tensorflow tf-keras onnx onnxruntime \
        onnx-graphsurgeon onnxsim sng4onnx psutil ai-edge-litert opencv-python-headless
    /tmp/.venv-sface-export/bin/python scripts/export_sface_tflite.py

INT8 calibration uses CC0 face crops fetched into memory (scripts/faces_cc0.py) plus in-memory
augmentations; no image is ever written to disk. The script validates both outputs against the
ONNX fp32 model (OpenCV DNN) with the LiteRT CPU interpreter and prints cosine similarities.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from faces_cc0 import augment, fetch_aligned_faces  # noqa: E402

MODELS = ROOT / "models"
ONNX = MODELS / "face_recognition_sface_2021dec.onnx"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"
OUT_FP16 = MODELS / "face_recognition_sface_2021dec_fp16.tflite"
OUT_INT8 = MODELS / "face_recognition_sface_2021dec_int8.tflite"
ONNX_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def to_input(crop_bgr: np.ndarray) -> np.ndarray:
    """Aligned BGR crop -> NHWC float32 RGB 0..255 (same as OpenCV FaceRecognizerSF's blob, NHWC)."""
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)[None]


def run_onnx2tf(src: Path, out: Path) -> None:
    """onnx2tf in-process, with PReLU emitted as relu(x) + (-a)*relu(-x).

    onnx2tf falls back to an ABS-based PReLU decomposition when the legacy Keras PReLU layer is not
    importable; that pattern is not fused by the TFLite converter and runs as slow, NPU-unfriendly
    element-wise ops. The relu form below is the canonical pattern that the converter legalises to a
    single TFL PRELU op (supported by NNAPI and NeuroPilot).
    """
    import onnx2tf
    import tensorflow as tf
    from onnx2tf.ops import PRelu as prelu_mod

    class _PReLU:
        def __init__(self, weights, shared_axes=None):
            self.alpha = tf.constant(weights) if not isinstance(weights, tf.Tensor) else weights

        def __call__(self, x):
            return tf.nn.relu(x) + (-self.alpha) * tf.nn.relu(-x)  # Keras PReLU form

    prelu_mod.PReLU = _PReLU
    onnx2tf.convert(
        input_onnx_file_path=str(src),
        output_folder_path=str(out),
        batch_size=1,
        output_signaturedefs=True,
        tflite_backend="tf_converter",
        non_verbose=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=30, help="number of CC0 portraits for calibration")
    a = ap.parse_args()
    if not ONNX.exists() or sha256(ONNX) != ONNX_SHA256:
        sys.exit(f"{ONNX} missing or checksum mismatch (expected {ONNX_SHA256}); run the app once to download it")

    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmp:
        # 1) ONNX (NCHW) -> TF SavedModel (NHWC) with onnx2tf; static batch 1.
        #    onnx2tf simplifies its input file in place, so convert a temporary copy.
        src = Path(tmp) / "sface.onnx"
        src.write_bytes(ONNX.read_bytes())
        run_onnx2tf(src, Path(tmp) / "sm")
        # 2) calibration crops (memory only)
        faces = fetch_aligned_faces(YUNET, ONNX, n=a.calib)
        rng = np.random.default_rng(0)
        calib = [c for _, _, f in faces for c in [f, *augment(f, rng)[:3]]]
        print(f"calibration: {len(faces)} CC0 portraits -> {len(calib)} crops (in memory)")

        def rep():
            for c in calib:
                yield [to_input(c)]

        conv = tf.lite.TFLiteConverter.from_saved_model(str(Path(tmp) / "sm"))
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
        OUT_FP16.write_bytes(conv.convert())

        conv = tf.lite.TFLiteConverter.from_saved_model(str(Path(tmp) / "sm"))
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        OUT_INT8.write_bytes(conv.convert())

    for p in (OUT_FP16, OUT_INT8):
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB sha256={sha256(p)}")
    validate([f for _, _, f in faces])


def validate(crops) -> None:
    sys.path.insert(0, str(ROOT))
    from presencekit.embed_backends import TFLiteSFace

    ref = cv2.FaceRecognizerSF.create(str(ONNX), "")
    for p in (OUT_FP16, OUT_INT8):
        m = TFLiteSFace(p)
        cs = []
        for c in crops:
            e0 = ref.feature(c).ravel()
            e1 = m.feature_aligned(c)
            cs.append(float(e0 @ e1 / np.linalg.norm(e0) / np.linalg.norm(e1)))
        print(f"{p.name}: cosine vs ONNX fp32 on {len(cs)} faces: mean {np.mean(cs):.4f} min {np.min(cs):.4f}")


if __name__ == "__main__":
    main()

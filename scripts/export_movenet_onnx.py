"""Reproducible export of MoveNet MultiPose Lightning (TFLite, float16 weights) to ONNX for ONNX Runtime.

Produces in the model store (models/ or $PRESENCE_MODEL_DIR):
  * movenet_multipose_lightning.onnx – same network and weights, opset 17, dynamic input [1, H, W, 3] uint8 RGB
    (H, W multiples of 32), output [1, 6, 56]. presence_demo uses it instead of LiteRT when it is present.

Toolchain (permissive licences; see docs/licenses.md): tf2onnx (Apache-2.0), TensorFlow (Apache-2.0), onnx (Apache-2.0),
onnxruntime (MIT), ai-edge-litert (Apache-2.0). TensorFlow is heavy and not a runtime dependency, so use a separate
environment (tf2onnx 1.16 needs protobuf 3.20 and numpy < 2):

    uv venv -p 3.12 /tmp/.venv-movenet-export
    VIRTUAL_ENV=/tmp/.venv-movenet-export uv pip install "tensorflow==2.17.*" "tf2onnx==1.16.1" "onnx==1.16.2" \
        "protobuf==3.20.3" "numpy<2" onnxruntime ai-edge-litert "opencv-python-headless<4.11"
    /tmp/.venv-movenet-export/bin/python scripts/export_movenet_onnx.py

The source TFLite file is downloaded by the app on first use of pose (or by this script). The script validates
the ONNX model against LiteRT on the same inputs at several input sizes and prints the largest difference.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from presence_demo.inference import CATALOG, ModelStore  # noqa: E402

SHAPES = [(192, 256), (256, 192), (256, 256), (160, 256), (224, 256)]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def convert(src: Path, dst: Path, opset: int) -> None:
    import tf2onnx

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / dst.name
        tf2onnx.convert.from_tflite(str(src), opset=opset, output_path=str(out))
        out.replace(dst)


def inputs() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    xs = [rng.integers(0, 256, (1, h, w, 3), dtype=np.uint8) for h, w in SHAPES]
    face = ROOT / "tests" / "data" / "face.jpg"
    try:
        import cv2

        from presence_demo.detectors.pose import letterbox

        img = cv2.imread(str(face))
        xs += [letterbox(img, 256)[0][None], letterbox(cv2.resize(img, (640, 480)), 256)[0][None]]
    except ImportError:
        pass
    return xs


def validate(tflite: Path, onnx_path: Path) -> float:
    import onnxruntime as ort
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(str(tflite), num_threads=4)
    i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    worst = 0.0
    for x in inputs():
        it.resize_tensor_input(i_in, list(x.shape))
        it.allocate_tensors()
        it.set_tensor(i_in, x)
        it.invoke()
        a = it.get_tensor(i_out)
        b = s.run(None, {name: x})[0]
        d = float(np.abs(a - b).max())
        worst = max(worst, d)
        print(f"  input {x.shape[1]}x{x.shape[2]}: max |LiteRT - ORT| = {d:.2e}, top score {a[0, :, 55].max():.4f}")
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tolerance", type=float, default=1e-4, help="max abs output difference vs LiteRT")
    a = ap.parse_args()
    store = ModelStore()
    src = store.ensure("movenet-tflite")
    if not src:
        sys.exit("MoveNet TFLite model unavailable (download failed); see docs/detection.md")
    dst = store.path("movenet")
    convert(src, dst, a.opset)
    print(f"{dst.name}: {dst.stat().st_size / 1e6:.1f} MB sha256={sha256(dst)} ({CATALOG['movenet'].license})")
    worst = validate(src, dst)
    if worst > a.tolerance:
        dst.unlink()
        sys.exit(f"validation failed: max difference {worst:.2e} > {a.tolerance:.0e}; removed {dst.name}")
    print(f"ok: max difference {worst:.2e}")


if __name__ == "__main__":
    main()

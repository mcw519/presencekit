"""SFace embedding backends: fp32 / int8 (ONNX Runtime, OpenCV DNN) / npu (TFLite + optional delegate, CPU fallback)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from presence_demo import vision
from presence_demo.embed_backends import OpenCVSFace, TFLiteSFace, align_crop, similarity_transform
from presence_demo.tracker import Config

FACE = Path(__file__).parent / "data" / "face.jpg"
M = vision.MODEL_DIR
FP32, INT8 = M / vision.SFACE[0], M / vision.SFACE_INT8[0]
TFL_INT8, TFL_FP16 = M / vision.SFACE_TFLITE["int8"], M / vision.SFACE_TFLITE["fp16"]
need = pytest.mark.skipif
# LiteRT's Delegate.__del__ raises (at GC time) for a delegate whose library failed to load: harmless upstream quirk
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")


@pytest.fixture(scope="module")
def face():
    p = vision.VisionPipeline(allow_download=False)
    if not getattr(p.detector, "landmarks", False):  # YuNet on ONNX Runtime or OpenCV DNN
        pytest.skip("YuNet model not available")
    img = cv2.imread(str(FACE))
    return img, p.detector.detect(img)[0][0]


def variants(img):
    return [
        img,
        cv2.convertScaleAbs(img, alpha=1.0, beta=30),
        cv2.convertScaleAbs(img, alpha=0.8, beta=-10),
        cv2.GaussianBlur(img, (5, 5), 1.0),
    ]


def test_similarity_transform_recovers_known_transform():
    a, s, t = 0.2, 1.7, np.array([5.0, -3.0])
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    src = np.random.default_rng(0).uniform(0, 100, (5, 2))
    dst = (s * (r @ src.T)).T + t
    m = similarity_transform(src, dst)
    assert np.allclose(m[:, :2], s * r) and np.allclose(m[:, 2], t)


@need(not FP32.exists(), reason="SFace fp32 model not downloaded")
def test_align_crop_matches_opencv(face):
    img, raw = face
    ref = cv2.FaceRecognizerSF.create(str(FP32), "").alignCrop(img, raw)
    assert np.abs(ref.astype(int) - align_crop(img, raw)).max() <= 2


@need(not (FP32.exists() and INT8.exists()), reason="SFace fp32/int8 models not downloaded")
def test_int8_close_to_fp32_and_normalised(face):
    img, raw = face
    a, b = OpenCVSFace(FP32), OpenCVSFace(INT8, "int8")
    for v in variants(img):
        ea, eb = a.embed(v, raw), b.embed(v, raw)
        assert ea.shape == eb.shape == (128,)
        assert abs(np.linalg.norm(eb) - 1) < 1e-5
        assert float(ea @ eb) > 0.95


@need(not (FP32.exists() and TFL_INT8.exists()), reason="run scripts/export_sface_tflite.py first")
@pytest.mark.parametrize("model", [TFL_INT8, TFL_FP16])
def test_tflite_close_to_fp32(face, model):
    if not model.exists():
        pytest.skip("model not exported")
    img, raw = face
    a, t = OpenCVSFace(FP32), TFLiteSFace(model)
    assert t.accelerator.startswith("cpu") and t.delegate_error is None
    e = t.embed(img, raw)
    assert e.shape == (128,) and abs(np.linalg.norm(e) - 1) < 1e-5
    assert float(a.embed(img, raw) @ e) > (0.97 if t.precision == "int8" else 0.999)


@need(not TFL_INT8.exists(), reason="run scripts/export_sface_tflite.py first")
def test_bad_delegate_falls_back_to_cpu(face, tmp_path):
    t = TFLiteSFace(TFL_INT8, delegate_path=str(tmp_path / "libneuron_delegate_missing.so"))
    assert t.delegate_error and t.accelerator.startswith("cpu")
    img, raw = face
    assert t.embed(img, raw).shape == (128,)


@need(not FP32.exists(), reason="SFace fp32 model not downloaded")
def test_pipeline_backend_selection_and_fallback(face, monkeypatch):
    p = vision.VisionPipeline(allow_download=False)
    assert p.embed_status["active"] == "fp32"
    if INT8.exists():
        st = p.select_embedder("int8")
        assert st["active"] == "int8" and p.embedder.name == "SFace int8"
    monkeypatch.setitem(vision.SFACE_TFLITE, "int8", "does_not_exist.tflite")
    st = p.select_embedder("npu", "", "int8")
    assert st["active"] == "fp32" and "unavailable" in st["note"]
    st = p.select_embedder("fp32")
    assert st == {**st, "active": "fp32", "requested": "fp32", "note": ""}
    assert p.info()["embed_backend"]["active"] == "fp32"


@need(not TFL_INT8.exists(), reason="run scripts/export_sface_tflite.py first")
def test_pipeline_npu_without_delegate_reports_cpu():
    p = vision.VisionPipeline(allow_download=False)
    st = p.select_embedder("npu", "", "int8")
    assert st["active"] == "npu" and st["accelerator"].startswith("cpu") and "not active" in st["note"]


def test_quantized_threshold_selection():
    c = Config()
    c.update({"cos_match": 0.4, "cos_match_int8": 0.45})
    assert c.match_cos == 0.4
    c.update({"embed_backend": "int8"})
    assert c.match_cos == 0.45
    c.update({"embed_backend": "npu", "npu_precision": "fp16"})
    assert c.match_cos == 0.4
    assert c.update({"embed_backend": "gpu"}) and c.embed_backend == "npu"

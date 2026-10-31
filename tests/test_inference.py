"""Inference layer (ONNX Runtime): providers, options, graph sanitising, session cache, model store, and parity of
the ORT detectors with the OpenCV DNN / LiteRT implementations they replace."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from presence_demo import inference, vision
from presence_demo.detectors import face as face_mod
from presence_demo.inference import ModelSpec, ModelStore, OrtOptions, load_model, strip_initializer_inputs
from presence_demo.inference import providers as prov

FACE = Path(__file__).parent / "data" / "face.jpg"
STORE = ModelStore(allow_download=False)


def need(key: str):
    p = STORE.local(key)
    if p is None:
        pytest.skip(f"model {key} not present ({inference.CATALOG[key].produce or 'downloaded on first run'})")
    return p


# --- providers / options -------------------------------------------------------------------------------------------


def test_resolve_keeps_order_skips_unavailable_and_always_ends_with_cpu():
    have = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    names, opts, notes = prov.resolve("coreml, qnn, cpu, cuda", {"cuda": {"device_id": 1}}, have)
    assert names == ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert opts == [{}, {"device_id": 1}, {}]
    assert notes == ["QNNExecutionProvider not available in this onnxruntime build"]
    assert prov.resolve("auto", avail=have)[0] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert prov.resolve("auto", avail=["CoreMLExecutionProvider", "CPUExecutionProvider"])[0] == [prov.CPU]
    assert prov.resolve((), avail=have)[0] == [prov.CPU]
    assert prov.short("DmlExecutionProvider") == "directml" and prov.short("FooExecutionProvider") == "foo"


def test_options_from_env_and_hashable_key():
    o = OrtOptions.from_env(
        {"PRESENCE_ORT_PROVIDERS": "coreml,cpu", "PRESENCE_ORT_THREADS": "2", "PRESENCE_ORT_GRAPH_OPT": "extended"}
    )
    assert o.providers == ("coreml", "cpu") and o.intra_op_threads == 2 and o.graph_optimization == "extended"
    assert OrtOptions.from_env({}) == OrtOptions()
    assert OrtOptions(providers="cuda, cpu").providers == ("cuda", "cpu")
    a = OrtOptions(provider_options={"cuda": {"device_id": 0}})
    assert hash(a) == hash(OrtOptions(provider_options={"CUDAExecutionProvider": {"device_id": "0"}}))
    with pytest.raises(ValueError):
        OrtOptions(graph_optimization="max")


# --- graph sanitising ----------------------------------------------------------------------------------------------


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(num: int, payload: bytes | int) -> bytes:
    if isinstance(payload, int):
        return _varint(num << 3) + _varint(payload)
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def _value_info(name: str, dims: tuple[int, ...]) -> bytes:
    shape = b"".join(_field(1, _field(1, d)) for d in dims)
    tensor_type = _field(1, 1) + _field(2, shape)  # elem_type FLOAT, shape
    return _field(1, name.encode()) + _field(2, _field(1, tensor_type))


def tiny_model(ir_version: int = 7) -> bytes:
    """y = x + w, with the initializer w also listed as a graph input (the pattern SFace ships with)."""
    node = _field(1, b"x") + _field(1, b"w") + _field(2, b"y") + _field(4, b"Add")
    w = _field(1, 3) + _field(2, 1) + _field(8, b"w") + _field(9, np.array([1, 2, 3], np.float32).tobytes())
    graph = (
        _field(1, node)
        + _field(2, b"g")
        + _field(5, w)
        + _field(11, _value_info("x", (3,)))
        + _field(11, _value_info("w", (3,)))
        + _field(12, _value_info("y", (3,)))
    )
    return _field(1, ir_version) + _field(8, _field(2, 13)) + _field(7, graph)


def test_strip_initializer_inputs_on_a_handmade_model():
    raw = tiny_model()
    out, n = strip_initializer_inputs(raw)
    assert n == 1 and len(out) < len(raw)
    for blob in (raw, out):
        s = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
        assert [i.name for i in s.get_inputs()] == ["x"]
        assert s.run(None, {"x": np.ones(3, np.float32)})[0].tolist() == [2, 3, 4]
    assert strip_initializer_inputs(out) == (out, 0)  # idempotent
    assert strip_initializer_inputs(tiny_model(ir_version=3)) == (tiny_model(ir_version=3), 0)  # pre-IR-4: required
    with pytest.raises(ValueError):
        strip_initializer_inputs(b"\x08\x07")  # no graph


def test_strip_matches_the_onnx_package():
    onnx = pytest.importorskip("onnx")
    p = need("sface")
    m = onnx.load(str(p))
    inits = {i.name for i in m.graph.initializer}
    keep = [i for i in m.graph.input if i.name not in inits]
    del m.graph.input[:]
    m.graph.input.extend(keep)
    out, n = strip_initializer_inputs(p.read_bytes())
    assert n == 174 and out == m.SerializeToString()


# --- sessions ------------------------------------------------------------------------------------------------------


def test_load_model_shares_sessions_and_falls_back_to_cpu(tmp_path, monkeypatch):
    f = tmp_path / "add.onnx"
    f.write_bytes(tiny_model())
    a, b = load_model(f), load_model(f)
    assert a is b and a.sanitized == 1 and a.providers == [prov.CPU] and a.accelerator == "cpu (onnxruntime)"
    assert load_model(f, OrtOptions(intra_op_threads=1)) is not a
    assert load_model(f, cache=False) is not a
    assert a.info()["sanitized_inputs"] == 1 and a.info()["model"] == "add.onnx"

    real = ort.InferenceSession
    monkeypatch.setattr(prov, "available", lambda: ["CUDAExecutionProvider", prov.CPU])

    def flaky(src, so=None, providers=None, provider_options=None):
        if providers and providers[0] != prov.CPU:
            raise RuntimeError("libcudart.so not found")
        return real(src, so, providers=providers)

    monkeypatch.setattr(ort, "InferenceSession", flaky)
    m = load_model(f, OrtOptions(providers=("cuda",)), cache=False)
    assert m.providers == [prov.CPU] and "CUDAExecutionProvider failed" in m.notes[0]


# --- model store ---------------------------------------------------------------------------------------------------


def test_store_verifies_checksums_and_never_downloads_when_disallowed(tmp_path, monkeypatch):
    data = b"x" * 2048
    spec = ModelSpec("demo", "demo.onnx", "https://example.invalid/demo.onnx", hashlib.sha256(data).hexdigest())
    store = ModelStore(tmp_path, allow_download=False)
    calls = []

    def urlopen(*a, **k):
        calls.append(a)
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert store.ensure(spec) is None and not calls
    assert store.ensure(spec, allow_download=True) is None and len(calls) == 1  # failed download -> None, no file
    assert not list(tmp_path.iterdir())
    (tmp_path / "demo.onnx").write_bytes(data)
    assert store.local(spec) == tmp_path / "demo.onnx"
    (tmp_path / "bad.onnx").write_bytes(data + b"!")
    assert store.local(ModelSpec("bad", "bad.onnx", sha256=spec.sha256)) is None
    (tmp_path / "pointer.onnx").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    assert store.local(ModelSpec("p", "pointer.onnx")) is None  # git-LFS pointer, not a model
    assert {r["key"] for r in store.status()} == set(inference.CATALOG)


def test_store_root_follows_presence_model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PRESENCE_MODEL_DIR", str(tmp_path))
    assert inference.default_root() == tmp_path and inference.default_store().root == tmp_path


# --- parity with the implementations ORT replaces --------------------------------------------------------------------


def collages():
    face = cv2.imread(str(FACE))
    rng = np.random.default_rng(0)
    out = [face]
    for w, h in [(640, 480), (512, 480), (333, 251), (1280, 720), (96, 120)]:
        canvas = (rng.random((h, w, 3)) * 60).astype(np.uint8)
        for _ in range(4):
            f = cv2.resize(
                face, None, fx=rng.uniform(0.15, 0.6) * w / face.shape[1], fy=rng.uniform(0.15, 0.6) * w / face.shape[1]
            )
            fh, fw = f.shape[:2]
            if fh < h and fw < w:
                y, x = rng.integers(0, h - fh), rng.integers(0, w - fw)
                canvas[y : y + fh, x : x + fw] = f
        out.append(canvas)
    return out


def test_nms_matches_opencv_nmsboxes():
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = int(rng.integers(2, 40))
        rows = np.zeros((n, 15), np.float32)
        rows[:, :2] = rng.uniform(-10, 200, (n, 2))
        rows[:, 2:4] = rng.uniform(0, 80, (n, 2))
        rows[:, 14] = rng.choice([0.5, 0.6, 0.7, 0.9], n) if rng.random() < 0.3 else rng.uniform(0.4, 1.0, n)
        top_k = int(rng.choice([0, 5, 50]))
        want = cv2.dnn.NMSBoxes(
            [tuple(int(v) for v in r[:4]) for r in rows], rows[:, 14].tolist(), 0.5, 0.3, 1.0, top_k
        )
        got = face_mod.nms_int_boxes(rows, 0.5, 0.3, top_k)
        assert np.array_equal(got, rows[np.asarray(want, int).ravel()])


def test_ort_yunet_matches_opencv_facedetectoryn():
    ort_det = vision.OrtYuNetDetector(need("yunet"))
    cv_det = vision.YuNetDetector(need("yunet-opencv"))
    found = 0
    for img in collages():
        a = np.array([r for r, _ in cv_det.detect(img)]).reshape(-1, 15)
        b = np.array([r for r, _ in ort_det.detect(img)]).reshape(-1, 15)
        assert a.shape == b.shape
        assert np.abs(a - b).max(initial=0.0) < 1e-3
        found += len(a)
    assert found >= 5


def test_ort_sface_matches_opencv_featurizer():
    from presence_demo.embed_backends import OpenCVSFace, OrtSFace

    img = cv2.imread(str(FACE))
    raw = vision.YuNetDetector(need("yunet-opencv")).detect(img)[0][0]
    a, b = OpenCVSFace(need("sface")), OrtSFace(need("sface"))
    assert b.model.sanitized > 0 and b.accelerator == "cpu (onnxruntime)"
    ea, eb = a.embed(img, raw), b.embed(img, raw)
    assert eb.shape == (128,) and abs(np.linalg.norm(eb) - 1) < 1e-5 and float(ea @ eb) > 0.9999
    if STORE.local("sface-int8"):  # quantized kernels differ between runtimes; same bar as fp32 vs int8
        e8 = OrtSFace(STORE.local("sface-int8"), "int8").embed(img, raw)
        assert float(e8 @ OpenCVSFace(STORE.local("sface-int8"), "int8").embed(img, raw)) > 0.99
        assert float(e8 @ eb) > 0.95


def test_ort_movenet_matches_litert():
    pytest.importorskip("ai_edge_litert")
    a = vision.MoveNetMultiPoseDetector(need("movenet-tflite"))
    b = vision.OrtMoveNetDetector(need("movenet"))
    img = cv2.imread(str(FACE))
    for im in (img, cv2.resize(img, (640, 480)), cv2.resize(img, (333, 251))):
        ra, rb = a.detect(im), b.detect(im)
        assert len(ra) == len(rb)
        for (ba, ca, ka), (bb, cb, kb) in zip(ra, rb, strict=True):
            assert np.allclose(ba, bb, atol=1e-2) and abs(ca - cb) < 1e-4
            assert all(np.allclose(ka[k], kb[k], atol=1e-2) for k in ka)


def test_pipeline_defaults_to_onnxruntime():
    need("yunet")
    need("sface")
    p = vision.VisionPipeline(allow_download=False)
    assert isinstance(p.detector, vision.OrtYuNetDetector) and isinstance(p.embedder, vision.OrtSFace)
    info = p.info()
    assert info["dnn_backend"] == "cpu (onnxruntime)" and info["dnn_target"] == "CPU" and info["onnxruntime"]
    dets = p.process(cv2.imread(str(FACE)))
    assert len(dets) == 1 and dets[0].embedding.shape == (128,)


def test_pose_loader_prefers_onnx():
    need("movenet")
    det = vision.load_pose_detector(allow_download=False)
    assert isinstance(det, vision.OrtMoveNetDetector)

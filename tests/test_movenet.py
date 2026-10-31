"""MoveNet MultiPose decoding (fake interpreter) + real-model smoke test when the model is present."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from presencekit import vision
from presencekit.vision import MODEL_DIR, MOVENET, MoveNetMultiPoseDetector, VisionPipeline

FACE = Path(__file__).parent / "data" / "face.jpg"


class FakeIt:
    def __init__(self, out):
        self.out, self.shape = out, None

    def resize_tensor_input(self, i, shape):
        self.shape = shape

    def allocate_tensors(self):
        pass

    def set_tensor(self, i, x):
        assert x.dtype == np.uint8 and list(x.shape) == self.shape

    def invoke(self):
        pass

    def get_tensor(self, i):
        return self.out


def make(out):
    d = MoveNetMultiPoseDetector.__new__(MoveNetMultiPoseDetector)
    d.it, d.inp, d.out, d.conf, d.size, d._shape = FakeIt(out), 0, 1, 0.3, 256, None
    return d


def test_decode_maps_padded_coords_to_pixels():
    out = np.zeros((1, 6, 56), np.float32)
    # 640x480 -> 256x192 (already multiple of 32): normalized coords map straight to the frame
    out[0, 0, 51:56] = (0.25, 0.5, 0.75, 0.75, 0.9)  # ymin, xmin, ymax, xmax, score
    out[0, 0, 0:3] = (0.3, 0.6, 0.8)  # nose y, x, score
    out[0, 1, 55] = 0.1  # below threshold
    d = make(out)
    res = d.detect(np.zeros((480, 640, 3), np.uint8))
    assert d.it.shape == [1, 192, 256, 3]
    assert len(res) == 1
    bbox, conf, kp = res[0]
    assert np.allclose(bbox, (320, 120, 480, 360)) and conf == pytest.approx(0.9)
    assert np.allclose(kp["nose"], (384, 144, 0.8), atol=1e-4)


def test_decode_with_padding():
    out = np.zeros((1, 6, 56), np.float32)
    out[0, 0, 51:56] = (0.0, 0.0, 1.0, 1.0, 0.9)
    d = make(out)
    res = d.detect(np.zeros((300, 500, 3), np.uint8))  # 256x154 -> padded to 256x160
    assert d.it.shape == [1, 160, 256, 3]
    x0, y0, x1, y1 = res[0][0]
    assert (x0, y0) == (0, 0) and x1 == pytest.approx(500) and y1 == pytest.approx(300)  # clipped


def test_face_embedder_env_switch(monkeypatch):
    monkeypatch.setenv("PRESENCE_FACE_EMBEDDER", "hist")
    called = []
    monkeypatch.setattr(vision, "ensure_model", lambda *a: called.append(a[0]) or None)
    p = VisionPipeline(detector=vision.HaarDetector())
    assert isinstance(p.embedder, vision.HistEmbedder)
    assert vision.SFACE[0] not in called


@pytest.mark.skipif(not (MODEL_DIR / MOVENET[0]).exists(), reason="MoveNet model not downloaded")
def test_real_movenet_finds_person():
    det = vision.load_pose_detector(allow_download=False)
    img = cv2.imread(str(FACE))
    res = det.detect(img)
    assert res, "expected at least one person"
    bbox, conf, kp = max(res, key=lambda r: r[1])
    x, y, c = kp["nose"]
    assert c > 0.3 and bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

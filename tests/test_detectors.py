"""Detectors layer: the protocols / implementations are the same objects as the v0.12 names; ORT is the default."""

from __future__ import annotations

import pytest

from presence_demo import detectors, embed_backends, vision


def test_vision_reexports_the_detector_layer_objects():
    assert vision.Detector is detectors.FaceDetector
    assert vision.Embedder is detectors.Embedder and vision.PoseDetector is detectors.PoseDetector
    assert vision.YuNetDetector is detectors.YuNetDetector and vision.HaarDetector is detectors.HaarDetector
    assert vision.SFaceEmbedder is detectors.SFaceEmbedder and vision.HistEmbedder is detectors.HistEmbedder
    assert vision.MoveNetMultiPoseDetector is detectors.MoveNetMultiPoseDetector
    assert detectors.OpenCVSFace is embed_backends.OpenCVSFace and detectors.OrtSFace is embed_backends.OrtSFace


def test_registry_lists_the_builtin_implementations():
    assert detectors.available("face") == ["haar", "yunet", "yunet-opencv"]
    assert detectors.available("embedder") == ["hist", "sface", "sface-opencv", "sface-tflite"]
    assert detectors.available("pose") == ["movenet", "movenet-tflite"]
    # the plain names are the ONNX Runtime implementations
    assert detectors.get("face", "yunet") is detectors.OrtYuNetDetector
    assert detectors.get("embedder", "sface") is detectors.OrtSFace
    assert detectors.get("pose", "movenet") is detectors.OrtMoveNetDetector
    assert isinstance(detectors.create("face", "haar"), vision.HaarDetector)
    assert isinstance(detectors.create("embedder", "hist"), vision.HistEmbedder)
    with pytest.raises(KeyError):
        detectors.get("face", "nope")
    with pytest.raises(ValueError):
        detectors.register("face", "haar", object)
    with pytest.raises(ValueError):
        detectors.register("voice", "x", object)


def test_haar_and_hist_satisfy_the_pipeline_contract():
    p = vision.VisionPipeline(detectors.create("face", "haar"), detectors.create("embedder", "hist"))
    assert p.name == "Haar(fallback) + HSV-hist(fallback)"

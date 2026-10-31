"""Detectors layer: face detector / embedder / pose protocols, built-in implementations and a name registry.

Since v0.14 the defaults run on ONNX Runtime through :mod:`presence_demo.inference` (``OrtYuNetDetector``,
``OrtSFace``, ``OrtMoveNetDetector``); the OpenCV DNN and LiteRT implementations remain as fallbacks. The default
pipeline with its fallbacks stays in ``presence_demo.vision``.

Depends on OpenCV (image ops), onnxruntime and optionally LiteRT, never on the web or runtime layers.
"""

from .base import Embedder, FaceDetector, PoseDetector
from .embed import HistEmbedder, OpenCVSFace, OrtSFace, SFaceEmbedder, TFLiteSFace
from .face import HaarDetector, OrtYuNetDetector, YuNetDetector
from .pose import COCO_KP, COCO_KP_IDX, MoveNetMultiPoseDetector, OrtMoveNetDetector
from .registry import available, create, get, register

__all__ = [
    "COCO_KP",
    "COCO_KP_IDX",
    "Embedder",
    "FaceDetector",
    "HaarDetector",
    "HistEmbedder",
    "MoveNetMultiPoseDetector",
    "OpenCVSFace",
    "OrtMoveNetDetector",
    "OrtSFace",
    "OrtYuNetDetector",
    "PoseDetector",
    "SFaceEmbedder",
    "TFLiteSFace",
    "YuNetDetector",
    "available",
    "create",
    "get",
    "register",
]

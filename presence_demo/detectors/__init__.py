"""Detectors layer (v0.13): face detector / embedder / pose protocols, built-in implementations
and a name registry. Model download and the default pipeline stay in ``presence_demo.vision``.

Depends on OpenCV / LiteRT but never on the web or runtime layers.
"""

from .base import Embedder, FaceDetector, PoseDetector
from .embed import HistEmbedder, OpenCVSFace, SFaceEmbedder, TFLiteSFace
from .face import HaarDetector, YuNetDetector
from .pose import COCO_KP, COCO_KP_IDX, MoveNetMultiPoseDetector
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
    "PoseDetector",
    "SFaceEmbedder",
    "TFLiteSFace",
    "YuNetDetector",
    "available",
    "create",
    "get",
    "register",
]

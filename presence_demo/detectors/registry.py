"""Name -> factory registry for detector implementations.

The plain names (``yunet``, ``sface``, ``movenet``) are the ONNX Runtime implementations (v0.14 defaults); the
suffixed names are the OpenCV DNN / LiteRT variants kept as fallbacks. The default selection with its fallbacks
lives in :class:`presence_demo.vision.VisionPipeline` and :func:`presence_demo.vision.load_pose_detector`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

KINDS = ("face", "embedder", "pose")
_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {k: {} for k in KINDS}


def register(kind: str, name: str, factory: Callable[..., Any], *, replace: bool = False) -> None:
    """Register ``factory`` (usually the class) under ``kind``/``name``."""
    if kind not in _REGISTRY:
        raise ValueError(f"unknown detector kind {kind!r} (expected one of {KINDS})")
    if name in _REGISTRY[kind] and not replace:
        raise ValueError(f"{kind} detector {name!r} is already registered")
    _REGISTRY[kind][name] = factory


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY[kind])


def get(kind: str, name: str) -> Callable[..., Any]:
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"no {kind} detector named {name!r}; available: {available(kind)}") from None


def create(kind: str, name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate a registered implementation, e.g. ``create("face", "yunet", model_path)``."""
    return get(kind, name)(*args, **kwargs)


def _register_builtin() -> None:
    from ..embed_backends import OpenCVSFace, OrtSFace, TFLiteSFace
    from .embed import HistEmbedder
    from .face import HaarDetector, OrtYuNetDetector, YuNetDetector
    from .pose import MoveNetMultiPoseDetector, OrtMoveNetDetector

    register("face", "yunet", OrtYuNetDetector)
    register("face", "yunet-opencv", YuNetDetector)
    register("face", "haar", HaarDetector)
    register("embedder", "sface", OrtSFace)
    register("embedder", "sface-opencv", OpenCVSFace)
    register("embedder", "sface-tflite", TFLiteSFace)
    register("embedder", "hist", HistEmbedder)
    register("pose", "movenet", OrtMoveNetDetector)
    register("pose", "movenet-tflite", MoveNetMultiPoseDetector)


_register_builtin()

"""Name -> factory registry for detector implementations.

v0.13 only registers the implementations that already ship; the default selection logic
(``presence_demo.vision.VisionPipeline``: YuNet+SFace with Haar/HSV fallbacks, and
``load_pose_detector``) is unchanged. Swapping or adding models is out of scope for v0.13.
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
    from ..embed_backends import OpenCVSFace, TFLiteSFace
    from .embed import HistEmbedder, SFaceEmbedder
    from .face import HaarDetector, YuNetDetector
    from .pose import MoveNetMultiPoseDetector

    register("face", "yunet", YuNetDetector)
    register("face", "haar", HaarDetector)
    register("embedder", "sface", SFaceEmbedder)
    register("embedder", "sface-opencv", OpenCVSFace)
    register("embedder", "sface-tflite", TFLiteSFace)
    register("embedder", "hist", HistEmbedder)
    register("pose", "movenet", MoveNetMultiPoseDetector)


_register_builtin()

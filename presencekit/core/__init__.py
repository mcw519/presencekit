"""Core layer: the presence state machine and multi-person tracking.

Depends only on numpy and the pure-python helper modules (``config``, ``body``, ``kalman``, ``intent``,
``presence_model``). It must never import OpenCV, FastAPI or any web/runtime module; ``tests/test_layering.py``
enforces the dependency direction.

``PresenceCore.update(Observation, now) -> Step`` is the typed entry point; :class:`PresenceTracker` is the
legacy (dict-returning) facade with unchanged behavior, also importable from ``presencekit.tracker``.
"""

from __future__ import annotations

from .engine import PresenceCore
from .geometry import cosine, iou
from .tracker import PresenceTracker
from .tracks import Detection, Track
from .types import EVENT_KINDS, Observation, PresenceEvent, Step, VideoFrame

__all__ = [
    "EVENT_KINDS",
    "Detection",
    "Observation",
    "PresenceCore",
    "PresenceEvent",
    "PresenceTracker",
    "Step",
    "Track",
    "VideoFrame",
    "cosine",
    "iou",
]

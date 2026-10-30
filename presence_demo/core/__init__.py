"""Core layer: the presence state machine and multi-person tracking.

Depends only on numpy and the pure-python helper modules (``config``, ``body``, ``kalman``, ``intent``,
``presence_model``). It must never import OpenCV, FastAPI or any web/runtime module; ``tests/test_layering.py``
enforces the dependency direction.
"""

from __future__ import annotations

from .types import EVENT_KINDS, Observation, PresenceEvent, Step, VideoFrame

__all__ = ["EVENT_KINDS", "Observation", "PresenceEvent", "Step", "VideoFrame"]

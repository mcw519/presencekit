"""Compatibility module: the tracker now lives in :mod:`presence_demo.core` (v0.13).

Every name that was importable from here in v0.12 still is, with the same objects and behavior.
"""

from __future__ import annotations

from .config import RANGES, Config, State  # noqa: F401  (re-exported: public API)
from .core.geometry import _norm, cosine, iou  # noqa: F401
from .core.tracker import PresenceTracker  # noqa: F401
from .core.tracks import Detection, Track, asdict_shallow  # noqa: F401

__all__ = ["RANGES", "Config", "Detection", "PresenceTracker", "State", "Track", "asdict_shallow", "cosine", "iou"]

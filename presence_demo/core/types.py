"""Typed values exchanged between the layers.

These types describe the data that already flows through the system (v0.12.x event dicts, per-frame detections,
binary video frames); they add no new behavior. Every type converts losslessly to the legacy dict format, so the
WebSocket protocol, logs and ``/debug/events`` output stay byte-for-byte compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Event kinds emitted by the tracker log (``PresenceTracker.log``). Every event carries a ``reason`` string.
EVENT_KINDS = (
    "transition",  # state change, extra keys ``frm`` / ``to``
    "track_new",
    "track_drop",
    "join",
    "drop",
    "far",
    "turned_away",
    "presence",
    "empty",
    "uncertain",
    "NEW_GROUP",
)


@dataclass(frozen=True)
class PresenceEvent:
    """One tracker log entry: ``{"t", "event", "reason", **extra}``."""

    t: float
    kind: str
    reason: str
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> PresenceEvent:
        extra = {k: v for k, v in d.items() if k not in ("t", "event", "reason")}
        return cls(t=d["t"], kind=d["event"], reason=d["reason"], extra=extra)

    def to_dict(self) -> dict:
        """The exact legacy dict (key order included)."""
        return {"t": self.t, "event": self.kind, "reason": self.reason, **self.extra}


@dataclass
class Observation:
    """Input to :meth:`presence_demo.core.PresenceCore.update`.

    ``detections`` is ``None`` for a speech-only (VAD change) or tick observation; ``speaking`` is ``None`` when the
    observation carries no speech information.
    """

    detections: list | None = None  # list[presence_demo.core.Detection]
    speaking: bool | None = None

    @property
    def kind(self) -> str:
        if self.detections is not None:
            return "frame"
        return "speech" if self.speaking is not None else "tick"


@dataclass(frozen=True)
class Step:
    """Output of one core update: events emitted during the update plus the snapshot afterwards."""

    events: tuple[PresenceEvent, ...]
    snapshot: dict


@dataclass(frozen=True)
class VideoFrame:
    """A validated in-memory video frame (JPEG bytes are never written to disk)."""

    data: bytes
    t: float | None  # client performance.now() milliseconds, None for legacy JSON frames without a timestamp
    speech: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> VideoFrame:
        return cls(data=d["data"], t=d["t"], speech=bool(d["speech"]))

    def to_dict(self) -> dict:
        return {"data": self.data, "t": self.t, "speech": self.speech}

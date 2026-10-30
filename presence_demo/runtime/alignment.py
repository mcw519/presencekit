"""Vision decision provenance. Never subtract the browser clock from the server clock."""

from __future__ import annotations

import math
from dataclasses import dataclass


def client_timestamp(value) -> float | None:
    """Browser performance.now() milliseconds, excluding bool/NaN/negative values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


@dataclass(frozen=True)
class VisionTiming:
    """Immutable provenance copied together with the tracker snapshot for an audio batch."""

    client_ms: float | None
    received: float
    submitted: float
    completed: float

    def measure(self, audio_client_ms, now: float, stale_after: float) -> tuple[float, dict]:
        audio_ms = client_timestamp(audio_client_ms)
        gap = audio_ms - self.client_ms if audio_ms is not None and self.client_ms is not None else None
        # A future video frame can accompany backlogged audio. Preserve the signed gap for
        # diagnosis; it cannot cancel server residence time or create negative freshness.
        age = max(0.0, now - self.received, (gap or 0.0) / 1000)
        return age, {
            "capture_gap_ms": gap,
            "processed_age_ms": max(0.0, now - self.completed) * 1000,
            "receive_to_processed_ms": max(0.0, self.completed - self.received) * 1000,
            "queue_wait_ms": max(0.0, self.submitted - self.received) * 1000,
            "processing_ms": max(0.0, self.completed - self.submitted) * 1000,
            "effective_age_ms": age * 1000,
            "stale": age > stale_after,
            "timestamp_available": gap is not None,
        }


def decision_timing(vision: VisionTiming | None, audio_client_ms, now: float, stale_after: float) -> tuple[float, dict]:
    if vision is not None:
        return vision.measure(audio_client_ms, now, stale_after)
    return math.inf, {
        "capture_gap_ms": None,
        "processed_age_ms": None,
        "receive_to_processed_ms": None,
        "queue_wait_ms": None,
        "processing_ms": None,
        "effective_age_ms": None,
        "stale": True,
        "timestamp_available": False,
    }

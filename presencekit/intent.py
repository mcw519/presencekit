"""Layer 3: per-track interaction-intent score in [0, 1] (pure python, no models, no images).

Each available cue is a number in [0, 1]; the raw score is the weighted mean over the cues that are
*available* (a missing cue, e.g. distance when distance estimation is off, is simply left out and the
remaining weights are renormalised). The track's intent is an exponential moving average of the raw
score over real time: ``intent += (raw - intent) * (1 - exp(-dt / tau))``.
"""

from __future__ import annotations

import math

# speech weight per priority_mode (only applied while speaking and attribution is plausible)
SPEECH_WEIGHT = {"vision": 0.05, "balanced": 0.15, "audio": 0.25}
FRONTAL_DEFAULT = 0.85  # face detected but no landmarks
DWELL_FULL = 4.0  # seconds in ROI for the full dwell cue
STILL_SPEED = 0.12  # lateral speed (frame widths / s) at which the "not passing by" cue reaches 0
APPROACH_GAIN = 4.0  # relative size change per second -> approach cue


def frontal_from_landmarks(row) -> float | None:
    """Frontal score from a YuNet row (x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y, ...).

    Uses the horizontal offset of the nose from the eye midpoint relative to the eye distance:
    0 offset -> 1.0 (frontal), offset >= 0.5 eye distance -> 0 (profile). None when the row has no landmarks
    (e.g. the Haar fallback, whose landmarks are all zero), so the caller uses its default.
    """
    try:
        rx, lx, nx = float(row[4]), float(row[6]), float(row[8])
    except (IndexError, TypeError, ValueError):
        return None
    eye = abs(lx - rx)
    if eye < 1e-6:
        return None
    off = abs(nx - (lx + rx) / 2) / eye
    return max(0.0, min(1.0, 1.0 - off / 0.5))


def facing_cue(has_face: bool, frontal: float | None, pose_score: float | None) -> float:
    f = (frontal if frontal is not None else FRONTAL_DEFAULT) if has_face else 0.0
    if pose_score is None:
        return f
    return 0.5 * f + 0.5 * pose_score if has_face else pose_score


def distance_cue(dist: float | None, zone_max: float, leave: float) -> float | None:
    if dist is None:
        return None
    if dist <= zone_max:
        return 1.0
    return max(0.0, 1.0 - (dist - zone_max) / max(leave - zone_max, 1e-3))


def center_cue(cx: float, cy: float, roi) -> float:
    x0, y0, x1, y1 = roi
    hx, hy = max((x1 - x0) / 2, 1e-3), max((y1 - y0) / 2, 1e-3)
    dx, dy = (cx - (x0 + x1) / 2) / hx, (cy - (y0 + y1) / 2) / hy
    return max(0.0, 1.0 - math.hypot(dx, 0.5 * dy))


def still_cue(vx: float) -> float:
    return max(0.0, 1.0 - abs(vx) / STILL_SPEED)


def approach_cue(rel_size_rate: float) -> float:
    return max(0.0, min(1.0, 0.5 + APPROACH_GAIN * rel_size_rate))


def raw_score(cues: dict, weights: dict) -> float:
    num = den = 0.0
    for k, v in cues.items():
        w = weights.get(k, 0.0)
        if v is None or w <= 0:
            continue
        num += w * v
        den += w
    return num / den if den > 0 else 0.0


FACING_GATE = 0.5  # how much looking away scales the whole score down (attention gate)


def gate(raw: float, facing: float) -> float:
    """Attention gate: someone not facing the robot cannot reach a high intent from dwell/position alone."""
    return raw * (1.0 - FACING_GATE * (1.0 - facing))


def smooth(prev: float, raw: float, dt: float, tau: float) -> float:
    a = 1.0 - math.exp(-max(dt, 0.0) / max(tau, 1e-3))
    return prev + (raw - prev) * a

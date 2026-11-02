"""Layer 2: per-track probabilistic presence (2-state present/left Bayes filter in log-odds).

Each track keeps L = log(P(present) / P(left)). Between two processed frames (real ``dt``) the
evidence observed in the new frame is integrated as a *rate* (log-likelihood ratio per second), so the
result does not depend on the frame rate:

    L <- clip(L + rate * dt, L_MIN, L_MAX)

* nobody seen:      rate = -ln2 / leave_halflife   (the odds of "present" halve every half-life)
* detected:         rate = +detect_gain * conf     (face or body confidence)
* predicted/coast:  rate = -leave_rate * (1 - vis_conf)  (weaker, fades toward "nobody seen")
* low-confidence:   rate = -leave_rate * lowconf_factor  (near neutral: graceful degrade)
* turned away + receding: rate = -leave_rate * TURN_FACTOR (even while detected)
* far (> leave_distance): rate = -leave_rate * FAR_FACTOR

Audio is *global* weak evidence added to the group log-odds (never assigned to a person), weighted by
``priority_mode``. Pure python + math; unit-testable.
"""

from __future__ import annotations

import math

P_MAX, P_MIN = 0.98, 0.02
L_MAX, L_MIN = math.log(P_MAX / (1 - P_MAX)), math.log(P_MIN / (1 - P_MIN))
TURN_FACTOR = 1.0
FAR_FACTOR = 3.0
MAX_DT = 3.0  # cap one integration step (e.g. after a stall)
# audio: (weight per priority_mode) x (speech bonus | silence penalty ramp). In vision mode silence is
# ignored (a quiet person in view is still present) and speech is only a tiny nudge (TV noise).
AUDIO_WEIGHT = {"vision": 0.05, "balanced": 0.8, "audio": 1.0}
SILENCE_WEIGHT = {"vision": 0.0, "balanced": 0.8, "audio": 1.0}
SPEECH_BONUS = 5.0
SILENCE_PENALTY = 6.0


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def leave_rate(halflife: float) -> float:
    return math.log(2) / max(halflife, 1e-3)


def track_rate(
    kind: str,
    conf: float,
    vis_conf: float,
    *,
    halflife: float,
    detect_gain: float,
    lowconf_factor: float,
    low_frame: bool = False,
    turned_away: bool = False,
    far: bool = False,
) -> float:
    """Evidence rate (log-odds per second) for one track in one frame."""
    lr = leave_rate(halflife)
    if far:
        return -lr * FAR_FACTOR
    if kind == "detected":
        return -lr * TURN_FACTOR if turned_away else detect_gain * conf
    if kind == "predicted":
        return -lr * (1.0 - vis_conf)
    if low_frame:
        return -lr * lowconf_factor
    return -lr


def step(L: float, rate: float, dt: float) -> float:
    return min(L_MAX, max(L_MIN, L + rate * min(max(dt, 0.0), MAX_DT)))


def audio_offset(mode: str, speaking: bool, silence: float, base_timeout: float) -> float:
    """Global audio evidence added to the group log-odds (never to a single person)."""
    if speaking:
        return AUDIO_WEIGHT.get(mode, 0.05) * SPEECH_BONUS
    return -SILENCE_WEIGHT.get(mode, 0.0) * SILENCE_PENALTY * min(1.0, silence / max(base_timeout, 1e-3))

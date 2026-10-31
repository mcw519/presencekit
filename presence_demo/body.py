"""Pure helpers for pose/body fusion and distance estimation (numpy only, unit-testable).

Coordinates are normalized [0,1] relative to the full frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

KP_CONF = 0.5


@dataclass
class Body:
    """A person found by the pose detector."""

    bbox: tuple  # normalized x0,y0,x1,y1
    conf: float
    keypoints: dict = field(default_factory=dict)  # name -> (x, y, conf), normalized


def _kp(b: Body, name: str, min_conf: float = KP_CONF):
    p = b.keypoints.get(name)
    return p if p is not None and p[2] >= min_conf else None


def shoulder_width(b: Body) -> float | None:
    """Horizontal shoulder distance (normalized frame width) or None if a shoulder is not visible."""
    ls, rs = _kp(b, "l_shoulder"), _kp(b, "r_shoulder")
    return abs(ls[0] - rs[0]) if ls and rs else None


def orientation(b: Body) -> tuple[float, str]:
    """Return (facing score 0..1, label) where label is 'facing', 'side' or 'away'.

    Cues: nose/eyes visible; left shoulder appears on image-right when facing the camera
    (mirror relation); shoulder width relative to torso height (small => side view).
    """
    score = 0.0
    nose, le, re = _kp(b, "nose"), _kp(b, "l_eye"), _kp(b, "r_eye")
    ls, rs = _kp(b, "l_shoulder"), _kp(b, "r_shoulder")
    lh, rh = _kp(b, "l_hip", 0.3), _kp(b, "r_hip", 0.3)
    if nose:
        score += 0.35
    if le and re:
        score += 0.25
    if ls and rs:
        if ls[0] > rs[0]:  # person's left shoulder on image right => facing camera
            score += 0.2
        torso_h = None
        if lh or rh:
            hy = (lh or rh)[1] if not (lh and rh) else (lh[1] + rh[1]) / 2
            torso_h = abs(hy - (ls[1] + rs[1]) / 2)
        else:
            torso_h = (b.bbox[3] - b.bbox[1]) * 0.35
        ratio = abs(ls[0] - rs[0]) / torso_h if torso_h and torso_h > 1e-6 else 0.0
        score += 0.2 * min(1.0, ratio / 0.6)
    label = "facing" if score >= 0.6 else ("side" if score >= 0.3 else "away")
    return round(score, 3), label


def head_box(b: Body) -> tuple:
    """Approximate head box from keypoints (or top of body bbox) for IoU continuity without a face."""
    pts = [p for n in ("nose", "l_eye", "r_eye", "l_ear", "r_ear") if (p := _kp(b, n, 0.3))]
    sw = shoulder_width(b)
    bw = b.bbox[2] - b.bbox[0]
    size = max(0.02, (sw if sw else bw) * 0.55)
    if pts:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
    else:
        cx = (b.bbox[0] + b.bbox[2]) / 2
        cy = b.bbox[1] + size * 0.6
    return (cx - size / 2, cy - size * 0.6, cx + size / 2, cy + size * 0.6)


def face_in_head_region(face_bbox: tuple, b: Body) -> bool:
    """Face center lies in the body's head region.

    True if the face center is near the keypoint head center (within one face width), or, when no head
    keypoints are visible, inside the upper 40% of the body box.
    """
    cx, cy = (face_bbox[0] + face_bbox[2]) / 2, (face_bbox[1] + face_bbox[3]) / 2
    x0, y0, x1, y1 = b.bbox
    if not (x0 <= cx <= x1 and y0 - 0.05 * (y1 - y0) <= cy <= y1):
        return False
    pts = [p for n in ("nose", "l_eye", "r_eye") if (p := _kp(b, n, 0.3))]
    if pts:
        hx = sum(p[0] for p in pts) / len(pts)
        hy = sum(p[1] for p in pts) / len(pts)
        fw = max(face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1])
        return math.dist((cx, cy), (hx, hy)) <= fw
    return cy <= y0 + 0.4 * (y1 - y0)


def associate(faces: list[tuple], bodies: list[Body]) -> list[tuple[int | None, int | None]]:
    """Greedy face<->body association. Returns (face_idx|None, body_idx|None) pairs covering all inputs."""
    pairs = []
    for fi, f in enumerate(faces):
        fc = ((f[0] + f[2]) / 2, (f[1] + f[3]) / 2)
        for bi, b in enumerate(bodies):
            if face_in_head_region(f, b):
                hb = head_box(b)
                hc = ((hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2)
                pairs.append((math.dist(fc, hc), fi, bi))
    pairs.sort()
    used_f, used_b, out = set(), set(), []
    for _, fi, bi in pairs:
        if fi in used_f or bi in used_b:
            continue
        used_f.add(fi)
        used_b.add(bi)
        out.append((fi, bi))
    out += [(fi, None) for fi in range(len(faces)) if fi not in used_f]
    out += [(None, bi) for bi in range(len(bodies)) if bi not in used_b]
    return out


def width_to_distance(width_norm: float, real_width_m: float, hfov_deg: float, calib: float = 1.0) -> float | None:
    """Pinhole model: distance = real_width / (width_norm * 2 * tan(hfov/2)) * calib."""
    if not width_norm or width_norm <= 0:
        return None
    return real_width_m / (width_norm * 2.0 * math.tan(math.radians(hfov_deg) / 2.0)) * calib


def estimate_distance(
    face_w: float | None,
    shoulder_w: float | None,
    facing: str | None,
    hfov_deg: float,
    face_width_m: float = 0.15,
    shoulder_width_m: float = 0.38,
    calib: float = 1.0,
) -> float | None:
    """Combine face-width and shoulder-width estimates (shoulders only when facing the camera)."""
    ests = []
    if face_w:
        ests.append(width_to_distance(face_w, face_width_m, hfov_deg, calib))
    if shoulder_w and facing == "facing":
        ests.append(width_to_distance(shoulder_w, shoulder_width_m, hfov_deg, calib))
    ests = [e for e in ests if e]
    return sum(ests) / len(ests) if ests else None


def calibration_factor(raw_distance: float, known_m: float) -> float:
    """Factor so that raw_distance (computed with calib=1) maps to the known distance."""
    if raw_distance <= 0 or known_m <= 0:
        raise ValueError("distances must be positive")
    return known_m / raw_distance


def calibrate(tracker, meters) -> list[str]:
    """Set ``distance_calib`` so the closest visible person's raw distance maps to ``meters``. Returns warnings.

    ``tracker`` is a :class:`presence_demo.core.PresenceTracker` (duck-typed: ``tracks``, ``cfg``)."""
    try:
        known = float(meters)
    except (TypeError, ValueError):
        return ["calibration: invalid distance"]
    cands = [t for t in tracker.tracks.values() if t.visible and t.dist_raw]
    if not cands or not known > 0:
        return ["calibration: no visible person with a distance estimate"]
    t = min(cands, key=lambda x: x.dist_raw)
    warns = tracker.cfg.update({"distance_calib": calibration_factor(t.dist_raw, known)})
    for x in tracker.tracks.values():
        x.distance = None  # restart EMA with the new factor
    return warns + [f"calibrated: factor {tracker.cfg.distance_calib:.3f} (track #{t.id})"]

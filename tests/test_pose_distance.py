import math

import numpy as np
import pytest

from presencekit.body import (
    Body,
    associate,
    calibration_factor,
    estimate_distance,
    face_in_head_region,
    orientation,
    width_to_distance,
)
from presencekit.optimizer import fuse
from presencekit.server import calibrate
from presencekit.tracker import Config, Detection, PresenceTracker, State


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def emb(seed=1):
    v = np.random.default_rng(seed).normal(size=64)
    return v / np.linalg.norm(v)


def body(cx=0.5, top=0.2, h=0.7, w=0.3, facing=True, conf=0.9):
    """Synthetic body: shoulders ~ w*0.8 apart; facing => nose/eyes visible and mirrored shoulders."""
    x0, x1 = cx - w / 2, cx + w / 2
    sw = w * 0.8
    lx, rx = (cx + sw / 2, cx - sw / 2) if facing else (cx - sw / 2, cx + sw / 2)
    sy = top + h * 0.25
    kp = {"l_shoulder": (lx, sy, 0.9), "r_shoulder": (rx, sy, 0.9), "l_hip": (cx + 0.05, top + h * 0.6, 0.8)}
    kp["r_hip"] = (cx - 0.05, top + h * 0.6, 0.8)
    if facing:
        kp.update(nose=(cx, top + h * 0.1, 0.95), l_eye=(cx + 0.02, top + h * 0.08, 0.9))
        kp["r_eye"] = (cx - 0.02, top + h * 0.08, 0.9)
    return Body((x0, top, x1, top + h), conf, kp)


# ---------------------------------------------------------------- distance
def test_width_to_distance_pinhole():
    w = 0.15 / (1.0 * 2 * math.tan(math.radians(30)))  # face width giving exactly 1 m at 60° HFOV
    assert width_to_distance(w, 0.15, 60) == pytest.approx(1.0)
    assert width_to_distance(w / 2, 0.15, 60) == pytest.approx(2.0)
    assert width_to_distance(w, 0.15, 60, calib=1.2) == pytest.approx(1.2)
    assert width_to_distance(0, 0.15, 60) is None


def test_estimate_distance_combines_and_ignores_turned_shoulders():
    k = 2 * math.tan(math.radians(30))
    fw, sw = 0.15 / (1.0 * k), 0.38 / (3.0 * k)  # face says 1 m, shoulders say 3 m
    assert estimate_distance(fw, sw, "facing", 60) == pytest.approx(2.0)
    assert estimate_distance(fw, sw, "side", 60) == pytest.approx(1.0)
    assert estimate_distance(None, sw, "facing", 60) == pytest.approx(3.0)
    assert estimate_distance(None, sw, "away", 60) is None


def test_calibration_factor_and_server_calibrate():
    assert calibration_factor(2.0, 1.5) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        calibration_factor(0, 1)
    clk = Clock()
    tr = PresenceTracker(Config(distance_enabled=True), clock=clk)
    k = 2 * math.tan(math.radians(30))
    fw = 0.15 / (2.0 * k)  # raw 2 m
    clk.t = 1
    tr.update([Detection((0.45, 0.3, 0.45 + fw, 0.3 + fw), 0.9, emb())], speaking=True)
    w = calibrate(tr, 1.5)
    assert tr.cfg.distance_calib == pytest.approx(0.75) and any("calibrated" in x for x in w)
    clk.t = 2
    s = tr.update([Detection((0.45, 0.3, 0.45 + fw, 0.3 + fw), 0.9, emb())])
    assert s["tracks"][0]["distance"] == pytest.approx(1.5, abs=0.01)
    assert "no visible person" in calibrate(PresenceTracker(Config()), 1.0)[0]


def test_distance_ema_and_zone():
    clk = Clock()
    tr = PresenceTracker(Config(distance_enabled=True, distance_ema=0.5), clock=clk)
    k = 2 * math.tan(math.radians(30))
    for i, d in enumerate([1.0, 2.0]):
        clk.t = i + 1
        fw = 0.15 / (d * k)
        s = tr.update([Detection((0.45, 0.3, 0.45 + fw, 0.3 + fw), 0.9, emb())], speaking=True)
    assert s["tracks"][0]["distance"] == pytest.approx(1.5, abs=0.01)
    assert s["tracks"][0]["zone"] == "in"


def test_leave_distance_counts_as_left_even_if_visible():
    k = 2 * math.tan(math.radians(30))
    near_fw, far_fw = 0.15 / (1.0 * k), 0.15 / (3.5 * k)

    def run(enabled):
        clk = Clock()
        tr = PresenceTracker(Config(distance_enabled=enabled, distance_ema=1.0), clock=clk)
        for _ in range(4):
            clk.t += 1
            tr.update([Detection((0.4, 0.3, 0.4 + near_fw, 0.3 + near_fw), 0.9, emb())], speaking=True)
        for _ in range(12):  # walks back to 3.5 m, still visible in ROI
            clk.t += 1
            tr.update([Detection((0.45, 0.3, 0.45 + far_fw, 0.3 + far_fw), 0.9, emb())])
        return tr

    tr = run(True)
    assert State.AWAY.value in [e.get("to") for e in tr.log]
    assert any(e["event"] == "far" for e in tr.log)
    tr = run(False)
    assert tr.state in (State.ACTIVE, State.IDLE) and tr.present


# ---------------------------------------------------------------- pose / association
def test_orientation_labels():
    assert orientation(body(facing=True))[1] == "facing"
    assert orientation(body(facing=False))[1] == "away"


def test_face_body_association_and_fuse():
    b1, b2 = body(cx=0.3), body(cx=0.7)
    f_left = (0.28, 0.20, 0.32, 0.26)  # near b1 nose (0.3, 0.27)
    f_right = (0.68, 0.20, 0.72, 0.26)
    assert face_in_head_region(f_left, b1) and not face_in_head_region(f_left, b2)
    pairs = associate([f_right, f_left], [b1, b2])
    assert (0, 1) in pairs and (1, 0) in pairs
    stray = (0.05, 0.9, 0.08, 0.95)
    pairs = associate([stray], [b1])
    assert (0, None) in pairs and (None, 0) in pairs
    dets = fuse([Detection(f_left, 0.9, emb())], [b1, b2])
    assert len(dets) == 2
    with_face = [d for d in dets if d.face_bbox is not None][0]
    body_only = [d for d in dets if d.face_bbox is None][0]
    assert with_face.body is b1 and body_only.body is b2 and body_only.embedding is None


def test_roi_separation_rules():
    edge_body = body(cx=0.1)  # body center at x=0.1: outside face ROI, inside full pose ROI
    det_body = Detection((0.08, 0.2, 0.12, 0.26), 0.9, None, face_bbox=None, body=edge_body)
    det_face = Detection((0.48, 0.3, 0.52, 0.36), 0.9, emb())
    for rule, pose, expect_body, expect_face in [
        ("either", True, True, True),
        ("face", True, False, True),
        ("body", True, True, False),  # face-only detection has no body
        ("either", False, False, True),  # pose disabled -> face ROI only
    ]:
        tr = PresenceTracker(Config(pose_enabled=pose, roi_rule=rule))
        assert tr.det_in_roi(det_body) is expect_body, rule
        assert tr.det_in_roi(det_face) is expect_face, rule
    tr = PresenceTracker(Config(pose_enabled=True, roi_rule="body", pose_roi=(0.0, 0.0, 0.2, 1.0)))
    assert tr.det_in_roi(det_body)
    assert not tr.det_in_roi(Detection((0.48, 0.3, 0.52, 0.36), 0.9, None, body=body(cx=0.5)))


def test_body_keeps_presence_and_identity_without_face():
    clk = Clock()
    tr = PresenceTracker(Config(pose_enabled=True), clock=clk)
    b = body(cx=0.5)
    for _ in range(3):
        clk.t += 1
        tr.update(fuse([Detection((0.48, 0.2, 0.52, 0.26), 0.9, emb())], [b]), speaking=True)
    tid = next(iter(tr.tracks))
    for _ in range(30):  # face lost (turned to the side), body still there
        clk.t += 1
        tr.update(fuse([], [body(cx=0.5, facing=False)]))
    assert list(tr.tracks) == [tid] and tr.present and tr.state in (State.ACTIVE, State.IDLE)


def _away_run(early: bool):
    clk = Clock()
    tr = PresenceTracker(Config(pose_enabled=True, turned_away_early=early, turned_away_seconds=2.0), clock=clk)
    for _ in range(3):
        clk.t += 1
        tr.update(fuse([Detection((0.48, 0.2, 0.52, 0.26), 0.9, emb())], [body()]), speaking=True)
    clk.t += 5
    tr.update(fuse([Detection((0.48, 0.2, 0.52, 0.26), 0.9, emb())], [body()]), speaking=False)
    h = 0.7
    for _ in range(6):  # turned away and getting smaller (walking away), still in ROI
        clk.t += 1
        h *= 0.85
        tr.update(fuse([], [body(top=0.2, h=h, w=0.3 * h / 0.7, facing=False)]), speaking=False)
    return tr


def test_turned_away_and_receding_triggers_early_maybe_away():
    tr = _away_run(True)
    assert tr.state == State.MAYBE_AWAY
    assert any(e["event"] == "turned_away" for e in tr.log)
    assert any("turned away" in e.get("reason", "") and e.get("to") == "MAYBE_AWAY" for e in tr.log)
    tr = _away_run(False)
    assert tr.state in (State.ACTIVE, State.IDLE)


# ---------------------------------------------------------------- config migration
def test_legacy_roi_migrates_to_face_roi():
    c = Config()
    c.update({"roi": [0.1, 0.1, 0.6, 0.7]})
    assert c.face_roi == (0.1, 0.1, 0.6, 0.7) and c.roi == c.face_roi
    c.update({"roi": [0.0, 0.0, 0.5, 0.5], "face_roi": [0.2, 0.2, 0.8, 0.8]})  # explicit wins
    assert c.face_roi == (0.2, 0.2, 0.8, 0.8)
    c.update({"pose_roi": [1, 1, 0, 0], "roi_rule": "body", "zone_min": 5, "zone_max": 3})
    assert c.pose_roi == (0.0, 0.0, 1.0, 1.0) and c.roi_rule == "body" and c.zone_min == 3
    assert c.update({"roi_rule": "nope"})

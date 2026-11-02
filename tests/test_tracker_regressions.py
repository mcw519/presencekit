"""Tracker regression tests (behaviours fixed after an algorithm audit)."""

import math

import numpy as np

from presencekit.body import Body
from presencekit.tracker import Config, Detection, PresenceTracker, State


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def emb(seed):
    v = np.random.default_rng(seed).normal(size=64)
    return v / np.linalg.norm(v)


FACE = {1: (0.40, 0.30, 0.50, 0.45), 2: (0.60, 0.30, 0.70, 0.45)}


def face(i, conf=0.9):
    return Detection(FACE[i], conf, emb(i))


def mk(mode="vision", **kw):
    clk = Clock()
    return PresenceTracker(Config(priority_mode=mode, **kw), clock=clk), clk


def step(tr, clk, dt, dets, speaking=None):
    clk.t += dt
    return tr.update(dets, speaking=speaking)


# 1. empty_since uses the first empty frame's timestamp (independent of fps)
def test_empty_since_is_first_empty_frame_at_2fps():
    tr, clk = mk()
    for _ in range(4):
        step(tr, clk, 0.5, [face(1)])
    first_empty = clk.t + 0.5
    for _ in range(3):  # absent_frames=3 at 2 fps
        step(tr, clk, 0.5, [])
    assert tr.empty and tr.empty_since == first_empty


def test_empty_since_at_slow_fps():
    tr, clk = mk()
    for _ in range(3):
        step(tr, clk, 2.0, [face(1)])
    first_empty = clk.t + 2.0
    for _ in range(3):
        step(tr, clk, 2.0, [])
    assert tr.empty_since == first_empty  # old code: now - 2 s (too late by 2 s)
    assert tr.state == State.MAYBE_AWAY  # empty for 4 s (< empty_timeout 5 s)
    clk.t += 1.0
    tr.tick()
    assert tr.state == State.AWAY


# 2. body-only detections use body_min_conf
def body(conf=0.5, cx=0.5):
    kp = {
        "nose": (cx, 0.3, 0.9),
        "l_eye": (cx + 0.02, 0.28, 0.9),
        "r_eye": (cx - 0.02, 0.28, 0.9),
        "l_shoulder": (cx + 0.1, 0.45, 0.9),
        "r_shoulder": (cx - 0.1, 0.45, 0.9),
        "l_hip": (cx + 0.05, 0.75, 0.8),
        "r_hip": (cx - 0.05, 0.75, 0.8),
    }
    return Body((cx - 0.15, 0.2, cx + 0.15, 0.95), conf, kp)


def body_only(conf=0.5):
    b = body(conf)
    return Detection((0.45, 0.25, 0.55, 0.35), conf, None, face_bbox=None, body=b)


def test_body_only_detection_counts_as_present():
    tr, clk = mk(pose_enabled=True)
    for _ in range(3):
        s = step(tr, clk, 1, [body_only(0.5)])
    assert s["path"] == "present" and tr.present
    tr2, clk2 = mk(pose_enabled=True, body_min_conf=0.6)
    for _ in range(3):
        s = step(tr2, clk2, 1, [body_only(0.5)])
    assert tr2.state == State.ENDED  # below body_min_conf: low confidence, never starts


def test_face_detection_still_uses_min_conf():
    tr, clk = mk(min_conf=0.6)
    for _ in range(3):
        step(tr, clk, 1, [face(1, conf=0.5)])
    assert tr.state == State.ENDED


# 3. far people neither start a session nor become NEW_GROUP newcomers
K = 2 * math.tan(math.radians(30))


def face_at(meters, x=0.45, seed=1):
    w = 0.15 / (meters * K)
    return Detection((x, 0.3, x + w, 0.3 + w), 0.9, emb(seed))


def test_far_person_does_not_start_session():
    tr, clk = mk(distance_enabled=True, distance_ema=1.0)
    for _ in range(25):
        step(tr, clk, 1, [face_at(3.5)])
    assert tr.state == State.ENDED and tr.session == 0
    step(tr, clk, 1, [face_at(1.0)])
    assert tr.state == State.ACTIVE


def test_far_newcomer_does_not_trigger_new_group():
    tr, clk = mk(distance_enabled=True, distance_ema=1.0, grace_seconds=60)
    for k in range(4):
        step(tr, clk, 1, [face_at(1.0, 0.3, seed=1)])
    assert tr._participants()
    for k in range(10):
        step(tr, clk, 1, [face_at(3.5, 0.55, seed=2)])
    assert not any(e["event"] == "NEW_GROUP" for e in tr.log)


# 4. speech_join
def test_speech_join_modes():
    def run(**kw):
        tr, clk = mk(**kw)
        step(tr, clk, 1, [face(1), face(2)], speaking=True)
        return {t.id: t.participant for t in tr.tracks.values()}

    assert run() == {1: False, 2: False}  # vision + auto -> off
    assert run(speech_join="all") == {1: True, 2: True}
    assert run(speech_join="single_candidate") == {1: False, 2: False}  # two candidates
    tr, clk = mk(speech_join="single_candidate")
    step(tr, clk, 1, [face(1)], speaking=True)
    assert tr.tracks[1].participant
    tr, clk = mk("balanced")  # auto -> single_candidate
    step(tr, clk, 1, [face(1)], speaking=True)
    assert tr.tracks[1].participant


def test_speech_join_single_candidate_requires_facing():
    tr, clk = mk("balanced", pose_enabled=True)
    b = body(0.9)
    b.keypoints = {k: v for k, v in b.keypoints.items() if k not in ("nose", "l_eye", "r_eye")}
    b.keypoints["l_shoulder"], b.keypoints["r_shoulder"] = (0.48, 0.45, 0.9), (0.52, 0.45, 0.9)  # back view
    d = Detection(FACE[1], 0.9, emb(1), face_bbox=FACE[1], body=b)
    step(tr, clk, 1, [d], speaking=True)
    assert tr.tracks[1].facing != "facing" and not tr.tracks[1].participant


# 5. vision mode: speech never recovers from AWAY
def test_vision_away_not_recovered_by_speech_when_uncertain():
    tr, clk = mk()
    for _ in range(3):
        step(tr, clk, 1, [face(1)])
    for _ in range(8):
        step(tr, clk, 1, [])
    assert tr.state == State.AWAY
    step(tr, clk, 0.5, [face(1, conf=0.3)], speaking=True)  # low confidence -> uncertain path
    assert tr.path == "uncertain" and tr.state == State.AWAY
    for _ in range(2):
        step(tr, clk, 0.5, [face(1)])
    assert tr.state in (State.ACTIVE, State.IDLE)


# small fixes
def test_low_streak_reset_on_new_session():
    tr, clk = mk()
    tr._low_streak = 7
    tr._reset_session(clk.t)
    assert tr._low_streak == 0


def test_low_confidence_frames_do_not_make_group_missing():
    tr, clk = mk(grace_seconds=60)
    for _ in range(4):
        step(tr, clk, 1, [face(1)])
    assert tr._participants()
    step(tr, clk, 1, [face(1, conf=0.3)])
    assert not tr._old_group_missing()
    for _ in range(3):
        step(tr, clk, 1, [])
    assert tr._old_group_missing()  # confirmed empty ROI is evidence


def test_snapshot_has_face_and_pose_roi():
    tr, _ = mk(pose_roi=(0.1, 0.1, 0.9, 0.9))
    s = tr.snapshot()
    assert s["roi"] == s["face_roi"] == list(tr.cfg.face_roi) and s["pose_roi"] == [0.1, 0.1, 0.9, 0.9]

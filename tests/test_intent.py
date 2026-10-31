"""Layer 3: per-track interaction-intent score (intent_model='score')."""

import math

import numpy as np
import pytest

from presence_demo import intent as im
from presence_demo.body import Body
from presence_demo.tracker import Config, Detection, PresenceTracker, State


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def emb(seed):
    v = np.random.default_rng(seed).normal(size=64)
    return v / np.linalg.norm(v)


def body(cx, cy, view="facing", face=True):
    w = 0.25
    if view == "facing":
        ls, rs = (cx + w * 0.35, cy + 0.12), (cx - w * 0.35, cy + 0.12)
    elif view == "away":
        ls, rs = (cx - w * 0.35, cy + 0.12), (cx + w * 0.35, cy + 0.12)
    else:
        ls, rs = (cx - 0.01, cy + 0.12), (cx + 0.01, cy + 0.12)
    kp = {
        "l_shoulder": (*ls, 0.9),
        "r_shoulder": (*rs, 0.9),
        "l_hip": (cx, cy + 0.4, 0.8),
        "r_hip": (cx, cy + 0.4, 0.8),
    }
    if face:
        kp.update(nose=(cx, cy, 0.9), l_eye=(cx + 0.015, cy - 0.01, 0.9), r_eye=(cx - 0.015, cy - 0.01, 0.9))
    return Body((cx - w / 2, cy - 0.08, cx + w / 2, cy + 0.6), 0.8, kp)


def person(i, cx, cy=0.4, frontal=0.95, view="facing", face=True):
    fb = (cx - 0.04, cy - 0.055, cx + 0.04, cy + 0.055)
    if not face:
        return Detection(fb, 0.75, None, face_bbox=None, body=body(cx, cy, view, False))
    return Detection(fb, 0.9, emb(i), face_bbox=fb, body=body(cx, cy, view), frontal=frontal)


def mk(**kw):
    clk = Clock()
    base = dict(intent_model="score", pose_enabled=True, leave_halflife=0.6, min_conf=0.6)  # pre-v0.11.1 tuning
    base.update(kw)
    return PresenceTracker(Config(**base), clock=clk), clk


def feed(tr, clk, seconds, dets_fn, dt=1.0, speaking=None):
    s = None
    for _ in range(int(round(seconds / dt))):
        clk.t += dt
        dets = dets_fn(clk.t) if callable(dets_fn) else list(dets_fn)
        s = tr.update(dets, speaking=speaking)
    return s


# ---------------------------------------------------------------- pure cues
def test_frontal_from_landmarks():
    frontal = [0, 0, 10, 10, 30, 40, 50, 40, 40, 50]  # nose midway between eyes
    profile = [0, 0, 10, 10, 30, 40, 50, 40, 52, 50]
    assert im.frontal_from_landmarks(frontal) == pytest.approx(1.0)
    assert im.frontal_from_landmarks(profile) == 0.0
    assert im.frontal_from_landmarks([1, 2, 3]) is None
    assert im.frontal_from_landmarks([0.0] * 15) is None  # Haar fallback rows carry no landmarks


def test_cues():
    assert im.facing_cue(True, None, None) == im.FRONTAL_DEFAULT
    assert im.facing_cue(False, None, 0.1) == 0.1
    assert im.distance_cue(1.5, 2.0, 2.5) == 1.0 and im.distance_cue(2.25, 2.0, 2.5) == pytest.approx(0.5)
    assert im.distance_cue(None, 2, 2.5) is None
    assert im.center_cue(0.5, 0.5, (0.2, 0.05, 0.8, 0.95)) == 1.0
    assert im.center_cue(0.8, 0.5, (0.2, 0.05, 0.8, 0.95)) == pytest.approx(0.0)
    assert im.still_cue(0.0) == 1.0 and im.still_cue(0.2) == 0.0
    assert im.approach_cue(0.2) > 0.5 > im.approach_cue(-0.2)


def test_missing_cue_renormalises_weights():
    w = {"facing": 0.5, "distance": 0.5}
    assert im.raw_score({"facing": 0.8, "distance": None}, w) == pytest.approx(0.8)
    assert im.raw_score({"facing": 0.8, "distance": 0.2}, w) == pytest.approx(0.5)


def test_smoothing_is_dt_invariant():
    a = b = 0.0
    for _ in range(8):
        a = im.smooth(a, 1.0, 0.5, 1.5)
    for _ in range(2):
        b = im.smooth(b, 1.0, 2.0, 1.5)
    assert a == pytest.approx(b) == pytest.approx(1 - math.exp(-4 / 1.5))


# ---------------------------------------------------------------- tracker behaviour
def test_engaged_person_starts_session_and_joins():
    tr, clk = mk()
    s = feed(tr, clk, 1, [person(1, 0.5)])
    assert s["state"] == "ENDED"  # score mode: a session starts on intent, not on any detection
    s = feed(tr, clk, 2, [person(1, 0.5)])
    assert s["state"] == "ACTIVE" and s["tracks"][0]["role"] == "participant" and s["tracks"][0]["intent"] > 0.6


def test_rules_default_unchanged():
    assert Config().intent_model == "rules"
    clk = Clock()
    tr = PresenceTracker(Config(pose_enabled=True), clock=clk)
    s = feed(tr, clk, 1, [person(1, 0.5)])
    assert s["state"] == "ACTIVE" and s["tracks"][0]["intent"] is None


def test_passerby_and_phone_user_do_not_join():
    tr, clk = mk()
    feed(tr, clk, 4, [person(1, 0.45)])

    def frame(t):
        return [person(1, 0.45), person(2, 0.1 + 0.12 * (t - 4), 0.42, face=False, view="side")]

    feed(tr, clk, 6, frame)
    feed(tr, clk, 10, [person(1, 0.45), person(3, 0.68, 0.42, face=False, view="away")])
    assert [t.id for t in tr.tracks.values() if t.participant] == [1]
    assert not any(e["event"] == "join" and e["track"] != 1 for e in tr.log)


def test_newcomer_facing_joins_and_speech_is_weak():
    tr, clk = mk()
    feed(tr, clk, 4, [person(1, 0.4)])
    feed(tr, clk, 3, [person(1, 0.4), person(2, 0.62)], speaking=True)
    assert tr.tracks[2].participant
    assert tr.tracks[2].intent_cues.get("speech") is None  # two people facing us: speech not attributable


def test_drop_to_bystander_and_bystander_does_not_keep_session():
    tr, clk = mk(presence_model="hmm", drop_seconds=4)
    feed(tr, clk, 4, [person(1, 0.4), person(2, 0.6)])
    assert all(t.participant for t in tr.tracks.values())
    feed(tr, clk, 8, [person(1, 0.4), person(2, 0.6, face=False, view="away")])
    assert tr.tracks[1].participant and not tr.tracks[2].participant
    assert any(e["event"] == "drop" and e["track"] == 2 for e in tr.log)
    feed(tr, clk, 7, [person(2, 0.6, face=False, view="away")])  # #1 leaves, only the bystander remains
    assert tr.state in (State.AWAY, State.ENDED)


def test_last_participant_is_never_dropped():
    tr, clk = mk(presence_model="hmm", drop_seconds=2)
    feed(tr, clk, 4, [person(1, 0.5)])
    s = feed(tr, clk, 15, [person(1, 0.5, face=False, view="away")])
    assert tr.tracks[1].participant and s["state"] in ("ACTIVE", "IDLE")


def test_new_group_requires_intent():
    tr, clk = mk(presence_model="hmm")
    feed(tr, clk, 4, [person(1, 0.5)])
    feed(tr, clk, 12, [person(2, 0.62, face=False, view="away")])
    assert not any(e["event"] == "NEW_GROUP" for e in tr.log)
    tr, clk = mk(presence_model="hmm")
    feed(tr, clk, 4, [person(1, 0.5)])
    feed(tr, clk, 12, [person(2, 0.62)])
    assert any(e["event"] == "NEW_GROUP" for e in tr.log)


def test_pending_new_group_timer_does_not_block_joins_after_end():
    # audio priority: the old participant left; a newcomer qualified for NEW_GROUP just before the AWAY timeout
    # ended the session. The pending timer used to survive ENDED and block every later join (no session ever again).
    tr, clk = mk(priority_mode="audio", presence_model="hmm")
    feed(tr, clk, 4, [person(1, 0.5)], speaking=True)
    feed(tr, clk, 6, [], speaking=False)
    feed(tr, clk, 4, [person(2, 0.62)], speaking=False)
    assert tr.state == State.ENDED and not any(e["event"] == "NEW_GROUP" for e in tr.log)
    feed(tr, clk, 3, [], speaking=False)  # the newcomer leaves the ROI ...
    s = feed(tr, clk, 3, [person(3, 0.5)], speaking=False)  # ... and an engaged person steps up
    assert s["state"] == "ACTIVE" and s["tracks"][0]["participant"]


def test_intent_weighted_presence():
    tr, clk = mk(presence_model="hmm", intent_weighted_presence=True)
    feed(tr, clk, 4, [person(1, 0.5)])
    t = tr.tracks[1]
    t.intent = tr.cfg.join_threshold / 2
    assert tr._track_group_p() == pytest.approx(t.p / 2)


def test_threshold_order():
    c = Config()
    c.update({"drop_threshold": 0.9})
    assert c.drop_threshold <= c.join_threshold

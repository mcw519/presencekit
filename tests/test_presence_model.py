"""Layer 2: per-track Bayes presence filter (presence_model='hmm')."""

import math

import numpy as np
import pytest

from presencekit import presence_model as pm
from presencekit.tracker import Config, Detection, PresenceTracker, State


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def emb(seed):
    v = np.random.default_rng(seed).normal(size=64)
    return v / np.linalg.norm(v)


BOX = {1: (0.40, 0.30, 0.48, 0.41), 2: (0.60, 0.30, 0.68, 0.41)}


def face(i, conf=0.9):
    return Detection(BOX[i], conf, emb(i))


def mk(mode="vision", **kw):
    clk = Clock()
    kw.setdefault("leave_halflife", 0.6)  # model math below was derived for the pre-v0.11.1 default
    kw.setdefault("min_conf", 0.6)
    return PresenceTracker(Config(presence_model="hmm", priority_mode=mode, **kw), clock=clk), clk


def feed(tr, clk, seconds, dets, dt=1.0, speaking=None):
    s = None
    for _ in range(int(round(seconds / dt))):
        clk.t += dt
        s = tr.update(list(dets), speaking=speaking)
    return s


# ---------------------------------------------------------------- math
def test_rates_and_clamp():
    lr = pm.leave_rate(0.6)
    assert lr == pytest.approx(math.log(2) / 0.6)
    kw = dict(halflife=0.6, detect_gain=4.0, lowconf_factor=0.15)
    assert pm.track_rate("detected", 0.9, 1.0, **kw) == pytest.approx(3.6)
    assert pm.track_rate("none", 0, 0, **kw) == pytest.approx(-lr)
    assert pm.track_rate("predicted", 0, 0.5, **kw) == pytest.approx(-0.5 * lr)
    assert pm.track_rate("none", 0, 0, low_frame=True, **kw) == pytest.approx(-0.15 * lr)
    assert pm.track_rate("detected", 0.9, 1, turned_away=True, **kw) < 0
    assert pm.track_rate("detected", 0.9, 1, far=True, **kw) == pytest.approx(-pm.FAR_FACTOR * lr)
    assert pm.step(pm.L_MAX, 10, 1) == pm.L_MAX and pm.step(pm.L_MIN, -10, 1) == pm.L_MIN
    assert pm.sigmoid(pm.logit(0.3)) == pytest.approx(0.3)


def test_odds_halve_every_halflife():
    L0 = pm.logit(0.8)
    L1 = pm.step(L0, pm.track_rate("none", 0, 0, halflife=2.0, detect_gain=4, lowconf_factor=0.1), 2.0)
    assert math.exp(L1) == pytest.approx(math.exp(L0) / 2)


def test_dt_invariance_over_fps():
    kw = dict(halflife=0.6, detect_gain=4.0, lowconf_factor=0.15)
    r = pm.track_rate("none", 0, 0, **kw)
    a = b = 2.0
    for _ in range(8):  # 2 fps for 4 s
        a = pm.step(a, r, 0.5)
    for _ in range(2):  # 0.5 fps for 4 s
        b = pm.step(b, r, 2.0)
    assert a == pytest.approx(b)


def test_tracker_dt_invariance_time_to_away():
    def away_time(dt):
        tr, clk = mk()
        feed(tr, clk, 6, [face(1)], dt=dt)
        t_leave = clk.t
        while tr.state != State.AWAY and clk.t < t_leave + 20:
            clk.t += dt
            tr.update([])
            for k in np.arange(0.5, dt, 0.5):
                tr.tick(clk.t + k)
        return clk.t - t_leave

    a, b = away_time(0.5), away_time(1.0)
    assert 4.0 <= a <= 6.0 and abs(a - b) <= 1.0


# ---------------------------------------------------------------- hysteresis
def test_hysteresis_thresholds_and_away_hold():
    tr, clk = mk()
    feed(tr, clk, 4, [face(1)])
    assert tr.state in (State.ACTIVE, State.IDLE)
    for p, expect in [(0.7, None), (0.55, State.MAYBE_AWAY), (0.75, State.MAYBE_AWAY), (0.85, State.IDLE)]:
        clk.t += 0.5
        tr.p_tracks = p
        tr.tick()
        if expect:
            assert tr.state == expect, p
        else:
            assert tr.state in (State.ACTIVE, State.IDLE)
    tr.p_tracks = 0.2
    clk.t += 0.5
    tr.tick()
    assert tr.state == State.MAYBE_AWAY  # below p_away but not yet for away_hold_seconds
    clk.t += 1.0
    tr.tick()
    assert tr.state == State.AWAY


def test_low_confidence_degrades_gracefully():
    def p_after(dets):
        tr, clk = mk()
        feed(tr, clk, 4, [face(1)])
        feed(tr, clk, 6, dets)
        return tr.p_tracks, tr.state

    p_low, st_low = p_after([face(1, conf=0.3)])
    p_none, st_none = p_after([])
    assert p_low > 0.9 > 0.6 > p_none and st_low in (State.ACTIVE, State.IDLE) and st_none == State.AWAY


# ---------------------------------------------------------------- audio
def test_audio_offset_weights_per_mode():
    sp = {m: pm.audio_offset(m, True, 0, 20) for m in ("vision", "balanced", "audio")}
    assert sp["vision"] < sp["balanced"] < sp["audio"]
    assert pm.audio_offset("vision", False, 60, 20) == 0.0
    assert pm.audio_offset("audio", False, 60, 20) < pm.audio_offset("balanced", False, 60, 20) < 0


def test_speech_with_empty_roi_per_mode():
    res = {}
    for mode in ("vision", "audio"):
        tr, clk = mk(mode)
        feed(tr, clk, 4, [face(1)], speaking=True)
        feed(tr, clk, 10, [], speaking=True)
        res[mode] = tr.state
    assert res["vision"] in (State.AWAY, State.ENDED) and res["audio"] in (State.ACTIVE, State.IDLE)


def test_audio_mode_long_silence_leaves_even_with_person():
    tr, clk = mk("audio")
    feed(tr, clk, 3, [face(1)], speaking=True)
    feed(tr, clk, 22, [face(1)], speaking=False)
    assert "AWAY" in [e.get("to") for e in tr.log]  # silence drives audio mode
    tr, clk = mk("vision")
    feed(tr, clk, 3, [face(1)], speaking=True)
    s = feed(tr, clk, 60, [face(1)], speaking=False)
    assert s["state"] == "IDLE" and s["p_group"] > 0.9


def test_vision_speech_never_recovers_from_away():
    tr, clk = mk()
    feed(tr, clk, 4, [face(1)])
    feed(tr, clk, 6, [])
    assert tr.state == State.AWAY
    clk.t += 0.5
    tr.update([], speaking=True)
    assert tr.state == State.AWAY
    feed(tr, clk, 2, [face(1)])
    assert tr.state in (State.ACTIVE, State.IDLE)


# ---------------------------------------------------------------- NEW_GROUP + snapshot
def test_new_group_in_hmm():
    tr, clk = mk()
    feed(tr, clk, 4, [face(1)])
    s = tr.session
    feed(tr, clk, 12, [face(2)])
    assert any(e["event"] == "NEW_GROUP" for e in tr.log) and tr.session > s
    assert [t.participant for t in tr.tracks.values()] == [True]
    # old participant still visible -> no NEW_GROUP
    tr, clk = mk()
    feed(tr, clk, 4, [face(1)])
    feed(tr, clk, 12, [face(1), face(2)])
    assert not any(e["event"] == "NEW_GROUP" for e in tr.log)


def test_snapshot_exposes_probabilities_and_rules_default():
    assert Config().presence_model == "rules"
    tr, clk = mk()
    s = feed(tr, clk, 3, [face(1)])
    assert 0.9 < s["p_group"] <= 1 and s["tracks"][0]["p"] == pytest.approx(pm.P_MAX, abs=1e-3)
    clk2 = Clock()
    rules = PresenceTracker(Config(), clock=clk2)
    clk2.t += 1
    s = rules.update([face(1)])
    assert s["p_group"] is None and s["tracks"][0]["p"] is None


def test_threshold_order_is_enforced():
    c = Config()
    w = c.update({"p_maybe": 0.9})
    assert c.p_away <= c.p_maybe <= c.p_active and w

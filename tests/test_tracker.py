import numpy as np
import pytest

from presencekit.tracker import Config, Detection, PresenceTracker, State


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def emb(seed):
    v = np.random.default_rng(seed).normal(size=128)
    return v / np.linalg.norm(v)


def noisy(seed, k):
    v = emb(seed) + 0.15 * np.random.default_rng(1000 + k).normal(size=128) / np.sqrt(128) * 4
    return v / np.linalg.norm(v)


BOX = {
    1: (0.4, 0.3, 0.55, 0.6),
    2: (0.6, 0.3, 0.75, 0.6),
    3: (0.25, 0.3, 0.4, 0.6),
    4: (0.45, 0.25, 0.6, 0.55),
    5: (0.3, 0.3, 0.45, 0.6),
}


def dets(ids, k=0, conf=0.9):
    return [Detection(BOX[i], conf, noisy(i, k)) for i in ids]


def run(tr, clk, seconds, ids, speaking=False, conf=0.9):
    snap = None
    for _ in range(int(seconds)):
        clk.t += 1.0
        snap = tr.update(dets(ids, int(clk.t), conf), speaking=speaking)
    return snap


@pytest.fixture
def setup():
    clk = Clock()
    return PresenceTracker(Config(priority_mode="balanced"), clock=clk), clk


def mk(mode, **kw):
    clk = Clock()
    return PresenceTracker(Config(priority_mode=mode, **kw), clock=clk), clk


def states(tr):
    return [e["to"] for e in tr.log if e["event"] == "transition"]


def test_single_user_leaves(setup):
    tr, clk = setup
    s = run(tr, clk, 5, [1], speaking=True)
    assert s["state"] == "ACTIVE" and s["tracks"][0]["participant"]
    run(tr, clk, 5, [1])  # silent but present
    assert tr.state in (State.IDLE, State.ACTIVE)
    run(tr, clk, 10, [])  # walks away
    assert State.AWAY.value in states(tr)
    run(tr, clk, 8, [])
    assert tr.state == State.ENDED and tr.tracks == {}  # embeddings cleared


def test_empty_roi_shortens_timeout(setup):
    tr, clk = setup
    run(tr, clk, 5, [1], speaking=True)
    t_leave = clk.t
    for _ in range(30):
        clk.t += 1
        tr.update([], speaking=False)
        if tr.state == State.AWAY:
            break
    assert clk.t - t_leave <= tr.cfg.empty_timeout + tr.cfg.absent_frames + 1
    assert clk.t - t_leave < tr.cfg.base_timeout


def test_group_partially_leaves_stays_active(setup):
    tr, clk = setup
    run(tr, clk, 5, [1, 2, 3], speaking=True)
    assert all(t["participant"] for t in tr.snapshot()["tracks"])
    run(tr, clk, 12, [3])  # two leave, one stays silent
    assert tr.state != State.AWAY and tr.state != State.ENDED
    assert tr.present and tr.timeout == tr.cfg.present_timeout
    assert not any(e["event"] == "NEW_GROUP" for e in tr.log)


def test_occlusion_flicker_does_not_trigger(setup):
    tr, clk = setup
    run(tr, clk, 5, [1], speaking=True)
    for k in range(12):  # every 3rd frame missing (< present_timeout)
        clk.t += 1
        tr.update(dets([1], k) if k % 3 else [], speaking=False)
        assert tr.state not in (State.AWAY, State.ENDED)
        assert tr.present
    assert len(tr.tracks) == 1  # same track id kept


def test_new_group_replaces_old(setup):
    tr, clk = setup
    run(tr, clk, 5, [1, 2], speaking=True)
    old_session = tr.session
    run(tr, clk, 8, [4, 5], speaking=True)  # different people step in
    ng = [e for e in tr.log if e["event"] == "NEW_GROUP"]
    assert ng, tr.log
    assert tr.session == old_session + 1
    ids = {t.id for t in tr.tracks.values()}
    assert len(ids) == 2 and all(t.participant for t in tr.tracks.values())
    assert tr.state == State.ACTIVE


def test_silence_with_presence_extends_timeout(setup):
    tr, clk = setup
    run(tr, clk, 5, [1], speaking=True)
    run(tr, clk, tr.cfg.empty_timeout + 5, [1])  # silent longer than empty-ROI timeout
    assert tr.state not in (State.AWAY, State.ENDED)
    assert tr.timeout == tr.cfg.present_timeout
    run(tr, clk, tr.cfg.present_timeout, [1])  # eventually still times out
    assert State.AWAY.value in states(tr)


def test_low_confidence_falls_back_to_base_timeout(setup):
    tr, clk = setup
    run(tr, clk, 5, [1], speaking=True)
    run(tr, clk, 6, [1], conf=0.3)  # only low-confidence detections
    assert tr.timeout == tr.cfg.base_timeout
    assert tr.state != State.AWAY


def test_returning_user_reactivates(setup):
    tr, clk = setup
    run(tr, clk, 5, [1], speaking=True)
    run(tr, clk, 9, [])
    assert tr.state == State.AWAY
    run(tr, clk, 1, [1], speaking=True)
    assert tr.state == State.ACTIVE


def test_roi_filter(setup):
    tr, clk = setup
    clk.t += 1
    tr.update([Detection((0.0, 0.3, 0.1, 0.6), 0.9, emb(9))])
    assert tr.tracks == {} and tr.state == State.ENDED


# ---------------- priority modes ----------------
def test_vision_noisy_speech_with_empty_roi_goes_away():
    tr, clk = mk("vision")
    run(tr, clk, 5, [1], speaking=True)
    assert tr.state == State.ACTIVE
    for _ in range(15):  # person leaves, TV/noise keeps "speech" on
        clk.t += 1
        tr.update([], speaking=True)
        if tr.state == State.AWAY:
            break
    assert tr.state == State.AWAY
    assert any("vision priority" in e["reason"] for e in tr.log)
    run(tr, clk, 8, [], speaking=True)  # speech alone does not revive or restart
    assert tr.state == State.ENDED and tr.tracks == {}


def test_vision_present_silent_long_stays():
    tr, clk = mk("vision")
    run(tr, clk, 5, [1], speaking=True)
    run(tr, clk, 300, [1])  # 5 minutes of silence, person visible
    assert tr.state == State.IDLE
    assert State.AWAY.value not in states(tr) and State.MAYBE_AWAY.value not in states(tr)


@pytest.mark.parametrize("model", ["rules", "hmm"])
def test_vision_present_max_silence_cap(model):
    tr, clk = mk("vision", present_max_silence=30, presence_model=model)
    run(tr, clk, 5, [1], speaking=True)
    run(tr, clk, 31, [1])
    assert tr.state == State.AWAY
    run(tr, clk, 60, [1])  # the same person stays in view, silent
    # AWAY once, then ENDED (used to flip AWAY <-> ACTIVE on every frame and never end); no silent restart
    assert states(tr).count(State.AWAY.value) == 1 and states(tr)[-1] == State.ENDED.value
    assert tr.state == State.ENDED
    session = tr.session
    run(tr, clk, 2, [1], speaking=True)  # the person in view speaks: a new session
    assert tr.state == State.ACTIVE and tr.session == session + 1


@pytest.mark.parametrize("model", ["rules", "hmm"])
def test_vision_capped_away_returns_on_new_evidence(model):
    tr, clk = mk("vision", present_max_silence=30, end_after_away=30, presence_model=model)
    run(tr, clk, 5, [1], speaking=True)
    run(tr, clk, 31, [1])
    assert tr.state == State.AWAY
    run(tr, clk, 1, [1], speaking=True)  # speech from the person in view
    assert tr.state == State.ACTIVE
    run(tr, clk, 31, [1])
    assert tr.state == State.AWAY
    run(tr, clk, 10, [])  # leaves (long enough for hmm P to fall below p_maybe) ...
    run(tr, clk, 3, [1])  # ... and comes back
    assert tr.state == State.ACTIVE and tr.log[-1]["reason"].startswith("user returned")
    run(tr, clk, 5, [1])  # the return restarted the silence clock: no immediate fall back to AWAY
    assert tr.state != State.AWAY


def test_vision_uncertain_falls_back_to_audio():
    tr, clk = mk("vision")
    run(tr, clk, 5, [1], speaking=True)
    run(tr, clk, 6, [1], conf=0.3)  # low confidence only
    assert tr.timeout == tr.cfg.base_timeout and tr.state != State.AWAY
    run(tr, clk, 20, [1], conf=0.3)
    assert State.AWAY.value in states(tr)


def test_vision_speech_does_not_join_instantly_by_default():
    tr, clk = mk("vision")
    clk.t += 1
    s = tr.update(dets([1]), speaking=True)
    assert s["state"] == "ACTIVE" and not s["tracks"][0]["participant"]  # speech_join auto -> off in vision
    clk.t += 2
    s = tr.update(dets([1], 1), speaking=True)
    assert s["tracks"][0]["participant"]  # after join_seconds


def test_balanced_speech_keeps_empty_roi_alive():
    tr, clk = mk("balanced")
    run(tr, clk, 5, [1], speaking=True)
    run(tr, clk, 15, [], speaking=True)
    assert tr.state == State.ACTIVE


def test_audio_mode_silence_drives():
    tr, clk = mk("audio")
    run(tr, clk, 5, [1], speaking=True)
    run(tr, clk, 15, [], speaking=True)  # empty ROI but speech -> still active
    assert tr.state == State.ACTIVE
    run(tr, clk, 1, [1], speaking=True)
    run(tr, clk, 25, [1])  # present: timeout = 1.5 * base = 30s
    assert tr.timeout == 30 and tr.state != State.AWAY
    run(tr, clk, 6, [1])
    assert State.AWAY.value in states(tr)


def test_priority_mode_validation():
    c = Config()
    assert c.priority_mode == "vision"
    w = c.update({"priority_mode": "foo"})
    assert c.priority_mode == "vision" and w
    c.update({"priority_mode": "audio"})
    assert c.priority_mode == "audio"


def test_stopped_counters_do_not_tick():
    tr, clk = mk("vision")
    clk.t += 100
    s = tr.tick()
    assert s["state"] == "ENDED" and s["silence"] == 0 and s["elapsed"] == 0 and s["in_state"] == 0

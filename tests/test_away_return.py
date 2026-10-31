"""Regression: in balanced/audio priority a silent person who stays in view must not flip AWAY<->ACTIVE
(v0.8.1 bug); a return from AWAY needs new evidence and AWAY -> ENDED must proceed. Vision unchanged."""

import numpy as np
import pytest

from presencekit.tracker import Config, Detection, PresenceTracker

E = {k: v / np.linalg.norm(v) for k, v in zip("AB", np.random.default_rng(0).normal(size=(2, 128)), strict=True)}
BOX = {"A": (0.4, 0.3, 0.55, 0.55), "B": (0.3, 0.3, 0.45, 0.55)}


def det(p):
    return Detection(BOX[p], 0.9, E[p].copy(), frontal=1.0)


def run(tr, t, script, t_end, speak=lambda x: False, step=0.5):
    """Frames + ticks every ``step`` s; returns list of (t, state) transitions."""
    trans = []
    x = t[0]
    while x <= t_end + 1e-9:
        t[0] = x
        sp = speak(x)
        if sp != tr.speaking:
            tr.set_speech(sp, now=x)
        prev = tr.state.value
        tr.update([det(p) for p in script(x)], speaking=sp, now=x)
        tr.tick(now=x)
        if tr.state.value != prev:
            trans.append((round(x, 2), tr.state.value))
        x = round(x + step, 6)
    return trans


def make(priority, presence, intent="rules", tracker="legacy"):
    t = [0.0]
    c = Config()
    c.update(dict(priority_mode=priority, presence_model=presence, intent_model=intent, tracker_mode=tracker))
    return PresenceTracker(c, clock=lambda: t[0]), t


def away_entries(tr):
    return sum(1 for e in tr.log if e["event"] == "transition" and e["to"] == "AWAY")


MODES = [(p, m) for p in ("vision", "balanced", "audio") for m in ("rules", "hmm")]


@pytest.mark.parametrize("priority,presence", MODES)
def test_silent_present_person_no_flip_flop(priority, presence):
    tr, t = make(priority, presence)
    run(tr, t, lambda x: ["A"], 120)
    if priority == "vision":
        assert tr.state.value == "IDLE"  # a quiet person in view never leaves in vision mode (unchanged)
        assert away_entries(tr) == 0
    else:
        assert tr.state.value == "ENDED"  # session ends ...
        assert away_entries(tr) == 1  # ... after exactly one AWAY, no AWAY<->ACTIVE flapping
        assert tr.session == 1  # continued presence does not restart a session


@pytest.mark.parametrize("priority,presence", MODES)
def test_silent_present_timings(priority, presence):
    tr, t = make(priority, presence)
    trans = run(tr, t, lambda x: ["A"], 60)
    first = {s: x for x, s in reversed(trans)}
    if priority == "vision":
        assert "AWAY" not in first
        return
    expected = {
        ("balanced", "rules"): (7.5, 15.0, 20.0),
        ("audio", "rules"): (15.0, 30.0, 35.0),
        ("balanced", "hmm"): (15.0, 21.0, 26.0),
        ("audio", "hmm"): (12.0, 17.0, 22.0),
    }[(priority, presence)]
    got = (first["MAYBE_AWAY"], first["AWAY"], first.get("ENDED"))
    assert got == pytest.approx(expected, abs=0.6)


@pytest.mark.parametrize("priority,presence", [m for m in MODES if m[0] != "vision"])
def test_speech_onset_returns_from_away(priority, presence):
    tr, t = make(priority, presence)
    run(tr, t, lambda x: ["A"], 40)  # until AWAY (before ENDED in all four combos? check below)
    tr2, t2 = make(priority, presence)
    trans = run(tr2, t2, lambda x: ["A"], 200, step=0.5)
    away_t = next(x for x, s in trans if s == "AWAY")
    tr3, t3 = make(priority, presence)
    run(tr3, t3, lambda x: ["A"], away_t + 1.0)
    assert tr3.state.value == "AWAY"
    run(tr3, t3, lambda x: ["A"], away_t + 2.0, speak=lambda x: True)
    assert tr3.state.value == "ACTIVE"
    assert tr3.session == 1


@pytest.mark.parametrize("priority,presence", [m for m in MODES if m[0] != "vision"])
def test_reentry_after_absence_returns_or_restarts(priority, presence):
    tr, t = make(priority, presence)
    trans = run(tr, t, lambda x: ["A"], 200)
    end_t = next(x for x, s in trans if s == "ENDED")
    tr2, t2 = make(priority, presence)
    run(tr2, t2, lambda x: ["A"], end_t + 2)
    assert tr2.state.value == "ENDED"
    # person leaves the ROI, then comes back: that is new evidence -> new session
    run(tr2, t2, lambda x: [] if x < end_t + 6 else ["A"], end_t + 10)
    assert tr2.state.value in ("ACTIVE", "IDLE")
    assert tr2.session == 2


@pytest.mark.parametrize("priority,presence", [m for m in MODES if m[0] != "vision"])
def test_speech_restarts_after_away_timeout(priority, presence):
    tr, t = make(priority, presence)
    trans = run(tr, t, lambda x: ["A"], 200)
    end_t = next(x for x, s in trans if s == "ENDED")
    tr2, t2 = make(priority, presence)
    run(tr2, t2, lambda x: ["A"], end_t + 1)
    run(tr2, t2, lambda x: ["A"], end_t + 2, speak=lambda x: True)
    assert tr2.state.value == "ACTIVE" and tr2.session == 2


@pytest.mark.parametrize("priority", ["balanced", "audio"])
def test_new_person_during_away_returns_rules(priority):
    tr, t = make(priority, "rules")
    trans = run(tr, t, lambda x: ["A"], 200)
    away_t = next(x for x, s in trans if s == "AWAY")
    tr2, t2 = make(priority, "rules")
    run(tr2, t2, lambda x: ["A"], away_t + 0.5)
    assert tr2.state.value == "AWAY"
    run(tr2, t2, lambda x: ["A", "B"], away_t + 2)
    assert tr2.state.value in ("ACTIVE", "IDLE")


@pytest.mark.parametrize("presence", ["rules", "hmm"])
@pytest.mark.parametrize("priority", ["vision", "balanced", "audio"])
def test_score_mode_no_restart_churn(priority, presence):
    tr, t = make(priority, presence, intent="score", tracker="kalman")
    run(tr, t, lambda x: ["A"], 150)
    assert tr.session == 1
    if priority != "vision":
        assert tr.state.value == "ENDED" and away_entries(tr) == 1

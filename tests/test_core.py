"""PresenceCore (v0.13 typed core API) is a behavior-preserving wrapper around PresenceTracker."""

from __future__ import annotations

import numpy as np

from presencekit import tracker as legacy
from presencekit.core import Detection, Observation, PresenceCore, PresenceEvent, PresenceTracker, Step
from presencekit.tracker import Config


def _face(x, seed=0):
    emb = np.random.default_rng(seed).normal(size=16)
    return Detection((x, 0.3, x + 0.15, 0.5), 0.9, emb)


def _script():
    yield 0.0, Observation(detections=[_face(0.4)])
    yield 0.5, Observation(detections=[_face(0.4)], speaking=True)
    for i in range(1, 8):
        yield 0.5 + i * 0.5, Observation(detections=[_face(0.4)])
    yield 5.0, Observation(speaking=False)
    for i in range(30):
        yield 6.0 + i, Observation(detections=[])
    yield 40.0, Observation()


def test_legacy_module_exports_the_core_objects():
    assert legacy.PresenceTracker is PresenceTracker and legacy.Detection is Detection


def test_core_steps_match_the_tracker_and_carry_the_new_log_entries():
    for kw in ({}, {"presence_model": "hmm", "tracker_mode": "kalman"}, {"priority_mode": "balanced"}):
        ref = PresenceTracker(Config(**kw), clock=lambda: 0.0)
        core = PresenceCore(Config(**kw), clock=lambda: 0.0)
        seen = []
        for now, obs in _script():
            if obs.detections is not None:
                want = ref.update([_face(0.4)] if obs.detections else [], obs.speaking, now)
            elif obs.speaking is not None:
                want = ref.set_speech(obs.speaking, now)
            else:
                want = ref.tick(now)
            step = core.update(obs, now)
            assert isinstance(step, Step) and step.snapshot == want
            assert all(isinstance(e, PresenceEvent) for e in step.events)
            seen.extend(e.to_dict() for e in step.events)
        assert seen == core.tracker.log == ref.log
        kinds = [e["event"] for e in seen]
        assert "join" in kinds and any(e.get("to") == "ENDED" for e in seen), kinds

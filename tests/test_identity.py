"""v0.13 identity layer: pure decision, call guard, and v0.12 compatibility of presencekit.reid."""

from __future__ import annotations

from pathlib import Path

import pytest

from presencekit import identity, reid
from presencekit.identity import CallGuard, Decision, MatchTally, Ref, decide


def _ref(cid, participant="p"):
    return Ref(1, cid, b"jpg", 0.0, participant)


def _tally(*scores):
    t = MatchTally(threshold=0.8, uncertain_low=0.5)
    for track, ref, score in scores:
        t.add(track, ref, score)
    return t


def test_single_confident_match_resumes_with_identities():
    a = _ref("A", "1:3")
    d = decide(_tally(("7", a, 0.9), ("7", _ref("B"), 0.1)), retried=False)
    assert d == Decision("resume", "match", a, 0.9, {"7": "1:3"})


def test_best_score_per_conversation_is_kept():
    a1, a2 = _ref("A", "x"), _ref("A", "y")
    d = decide(_tally(("1", a1, 0.85), ("1", a2, 0.95)), retried=False)
    assert d.ref is a2 and d.score == 0.95 and d.identities == {"1": "y"}


@pytest.mark.parametrize(
    "scores",
    [
        [("1", _ref("A"), 0.9), ("1", _ref("B"), 0.9)],  # two conversations match
        [("1", _ref("A"), 0.9), ("1", _ref("B"), 0.6)],  # one match, another uncertain
    ],
)
def test_ambiguous_starts_a_new_conversation(scores):
    assert decide(_tally(*scores), retried=False) == Decision("new", "ambiguous")


def test_uncertain_retries_once_then_new():
    t = _tally(("1", _ref("A"), 0.6), ("1", _ref("B"), 0.7))
    assert decide(t, retried=False) == Decision("retry", "retry", score=0.7)
    assert decide(t, retried=True) == Decision("new", "uncertain", score=0.7)


def test_below_threshold_and_empty():
    assert decide(_tally(("1", _ref("A"), 0.2)), retried=False) == Decision("new", "below_threshold")
    assert decide(_tally(), retried=True) == Decision("new", "below_threshold")


def test_call_guard_caps_breaker_and_cooldown():
    g = CallGuard(breaker_errors=2, breaker_pause=60)
    assert g.block_reason(0, cap_per_hour=2, cap_total=10, key_missing=False) is None
    g.reserve(0)
    g.reserve(1)
    assert g.block_reason(2, 2, 10, False) == "cap"
    assert g.block_reason(3602, 2, 10, False) is None  # hour window slides
    assert g.block_reason(3602, 5, 10, True) == "no_key"
    assert not g.failed(10) and g.failed(11) and g.paused_until == 71
    assert g.block_reason(70, 5, 10, False) == "breaker"
    g.succeeded(12)
    assert g.errors == 0 and g.tokens == 12
    g.decided(100)
    assert g.cooling_down(105, 10) and not g.cooling_down(111, 10)


def test_reid_module_reexports_the_identity_layer():
    assert reid.CrossSessionReid is identity.ContinuationDriver
    assert reid.JevClient is identity.JevMatcher and reid.ReferenceStore is identity.ReferenceStore
    assert reid.set_key is identity.set_key and reid.wipe_all is identity.wipe_all
    assert isinstance(reid.JevClient(), identity.IdentityMatcher)


def test_driver_keeps_v012_counter_attributes():
    from presencekit.tracker import Config

    d = identity.ContinuationDriver(Config(), client=reid.CloudReid())
    d.calls, d.tokens = 3, 9
    assert d.guard.calls == 3 and d.guard.tokens == 9 and d.errors == 0 and d.last_decision_t == -1e9


def test_identity_layer_never_writes_files():
    for path in (Path(identity.__file__).parent).glob("*.py"):
        src = path.read_text()
        for bad in (" open(", "imwrite", "tofile", "NamedTemporaryFile", "write_bytes"):
            assert bad not in src, (path.name, bad)

"""Pure re-ID decision: fold the comparison scores of one session into resume / new / retry.

No I/O, no clock, no asyncio: :class:`ContinuationDriver` feeds every successful comparison into a
:class:`MatchTally` and calls :func:`decide` once all (query, reference) pairs have been compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MatchTally:
    threshold: float
    uncertain_low: float
    matched: dict = field(default_factory=dict)  # conversation_id -> (best score, ref)
    identities: dict = field(default_factory=dict)  # conversation_id -> {query track: participant}
    uncertain_conversations: set = field(default_factory=set)
    best_uncertain: float | None = None

    def add(self, track: str, ref: Any, score: float) -> None:
        if score >= self.threshold:
            old = self.matched.get(ref.conversation_id)
            if old is None or score > old[0]:
                self.matched[ref.conversation_id] = (score, ref)
            self.identities.setdefault(ref.conversation_id, {})[track] = ref.participant
        elif score >= self.uncertain_low:
            self.uncertain_conversations.add(ref.conversation_id)
            if self.best_uncertain is None or score > self.best_uncertain:
                self.best_uncertain = score


@dataclass(frozen=True)
class Decision:
    decision: str  # "resume" | "new" | "retry"
    reason: str  # match | ambiguous | uncertain | below_threshold | retry
    ref: Any = None
    score: float | None = None
    identities: dict | None = None  # resumed participant identities (resume only)


def decide(tally: MatchTally, retried: bool) -> Decision:
    """Same person in exactly one previous conversation -> resume; several / mixed -> new ("ambiguous");
    only uncertain scores -> one retry with fresh crops, then new; nothing close -> new."""
    if len(tally.matched) > 1:
        return Decision("new", "ambiguous")
    if len(tally.matched) == 1:
        cid, (score, ref) = next(iter(tally.matched.items()))
        if tally.uncertain_conversations - {cid}:
            return Decision("new", "ambiguous")
        return Decision("resume", "match", ref, score, tally.identities[cid])
    best = tally.best_uncertain
    if best is not None and not retried:
        return Decision("retry", "retry", score=best)
    return Decision("new", "uncertain" if best else "below_threshold", score=best)

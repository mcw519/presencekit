"""Memory-only reference store of previous conversations' participant face crops (see docs/reid.md)."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field


@dataclass
class Ref:
    session: int
    conversation_id: str
    jpeg: bytes = field(repr=False)
    t: float = 0.0
    participant: str = ""


class ReferenceStore:
    """Memory-only references; the history cap counts conversations, not faces.

    ``active`` survives a same-page transport interruption separately from history.
    Its timestamps are not refreshed by reconnecting or unsuccessful comparisons.
    """

    def __init__(self):
        self.refs: list[Ref] = []
        self.active: list[Ref] = []
        self.generation = 0
        self.active_connected = False
        self.active_until = 0.0
        self.active_cid = None
        self.active_expiries = {}
        self.ended_at = {}  # Updated only by a normal session ENDED commit.
        self.drivers = weakref.WeakSet()

    @staticmethod
    def _clear(refs):
        for ref in refs:
            ref.jpeg = b""

    def add(self, ref: Ref, max_refs: int) -> None:
        members = [r for r in self.refs if r.conversation_id == ref.conversation_id]
        members = [r for r in members if r.participant != ref.participant] + [ref]
        self.commit(members, max_refs, ref.t)

    def commit(self, members: list[Ref], max_refs: int, now: float) -> None:
        if not members:
            return
        cid = members[0].conversation_id
        old = [r for r in self.refs + self.active if r.conversation_id == cid]
        merged = {r.participant: r for r in old}
        for ref in members:
            previous = merged.get(ref.participant)
            if previous is not None and previous is not ref:
                previous.jpeg = b""
            merged[ref.participant] = ref
        selected = list(merged.values())[-3:]
        self._clear([r for r in old if r not in selected])
        for ref in selected:
            ref.t = now
        self.refs = [r for r in self.refs if r.conversation_id != cid] + selected
        self.active = [r for r in self.active if r.conversation_id != cid]
        if self.active_cid == cid:
            self.active_cid = None
            self.active_connected = False
        self.active_expiries.pop(cid, None)
        self.ended_at[cid] = now
        self.trim(max_refs)

    def trim(self, max_refs):
        inactive = [
            r for r in self.refs + self.active if not (self.active_connected and r.conversation_id == self.active_cid)
        ]
        times = {}
        for ref in inactive:
            times[ref.conversation_id] = max(times.get(ref.conversation_id, -1e30), ref.t)
        oldest = sorted(times, key=times.get)
        evict = set(oldest[: max(0, len(oldest) - max_refs)])
        self._clear([r for r in self.refs + self.active if r.conversation_id in evict])
        self.refs = [r for r in self.refs if r.conversation_id not in evict]
        self.active = [r for r in self.active if r.conversation_id not in evict]
        for cid in evict:
            self.active_expiries.pop(cid, None)
            self.ended_at.pop(cid, None)

    def interrupt(self, conversation_id, now, ttl):
        # A reconnect must never renew an ended conversation's retention lease.
        # A never-ended conversation gets exactly one lease at first interruption.
        if conversation_id in self.ended_at:
            deadline = self.ended_at[conversation_id] + ttl
            self.active_expiries[conversation_id] = deadline
        else:
            deadline = self.active_expiries.setdefault(conversation_id, now + ttl)
        self.active_connected = False
        self.active_until = deadline

    def activate(self, conversation_id):
        self.active.extend(r for r in self.refs if r.conversation_id == conversation_id)
        self.refs = [r for r in self.refs if r.conversation_id != conversation_id]
        self.active_connected = True
        self.active_cid = conversation_id

    def expire(self, now: float, ttl: float) -> int:
        expired = [r for r in self.refs if now - r.t >= ttl]
        self.refs = [r for r in self.refs if now - r.t < ttl]
        expired_active = [
            r
            for r in self.active
            if not (self.active_connected and r.conversation_id == self.active_cid)
            and now >= self.active_expiries.get(r.conversation_id, r.t + ttl)
        ]
        expired.extend(expired_active)
        self.active = [r for r in self.active if r not in expired_active]
        self._clear(expired)
        retained = {r.conversation_id for r in self.refs + self.active}
        for cid in (set(self.ended_at) | set(self.active_expiries)) - retained:
            self.ended_at.pop(cid, None)
            self.active_expiries.pop(cid, None)
        return len(expired)

    def valid(self, now: float, ttl: float) -> list[Ref]:
        history = [r for r in self.refs if now - r.t < ttl]
        active = [
            r
            for r in self.active
            if (self.active_connected and r.conversation_id == self.active_cid)
            or now < self.active_expiries.get(r.conversation_id, r.t + ttl)
        ]
        return [r for r in reversed(history + active) if r.jpeg]

    def wipe(self) -> int:
        n = len(self.refs) + len(self.active)
        self._clear(self.refs + self.active)
        self.refs.clear()
        self.active.clear()
        self.active_connected = False
        self.active_until = 0.0
        self.active_cid = None
        self.active_expiries.clear()
        self.ended_at.clear()
        self.generation += 1
        for driver in list(self.drivers):
            driver._store_wiped()
        return n


_STORES: weakref.WeakSet[ReferenceStore] = weakref.WeakSet()


def wipe_all() -> int:
    """Wipe every connection's references (``POST /api/reid_reset``, shutdown)."""
    return sum(s.wipe() for s in list(_STORES))

"""Per-page continuation driver: resume the previous conversation or start a new one (see docs/reid.md).

PRIVACY: face crops stay in memory (:mod:`.store`); only crops, never whole frames, go to the matcher.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from .crops import face_quality, thumb
from .decision import MatchTally, decide
from .guard import CallGuard
from .keys import key_status
from .matcher import JevClient
from .store import _STORES, Ref, ReferenceStore


class ContinuationDriver:
    """Nonblocking, per-page re-ID driver with generation-safe crop and task delivery.

    Decides whether a new presence session continues a previous conversation (``resume``) or starts a
    new one. Remote calls go through an :class:`IdentityMatcher`; budget / cooldown / breaker live in a
    :class:`CallGuard`; the final verdict comes from the pure :func:`decide`. v0.12 name: ``CrossSessionReid``."""

    def __init__(self, cfg, client=None, emit=None, event=None, clock=time.monotonic, store=None):
        self.cfg = cfg
        self.client = client or JevClient()
        self.emit = emit or (lambda d: None)
        self.event = event or (lambda *args, **kv: None)
        self.clock = clock
        self.store = store if store is not None else ReferenceStore()
        self.store.drivers.add(self)
        _STORES.add(self.store)
        self.session = 0
        self.conversation_id = None
        self.members = {}  # track ID -> (quality, JPEG, last improvement time)
        self.member_ids = {}  # query track ID -> historical participant identity on resume
        self.phase = "idle"
        self.started = 0.0
        self.first_part_t = None
        self.wait_why = {}
        self.retried = False
        self.task = None
        self.deadline_task = None
        self.epoch = 0
        self.ended = True
        self.queries = {}
        self.expected = set()
        self.guard = CallGuard()
        self.last = None
        self.committed = True
        self.was_on = False
        self.view_probe = self.view_ref = None

    # v0.12 attribute names, backed by the CallGuard
    calls = property(lambda self: self.guard.calls, lambda self, v: setattr(self.guard, "calls", v))
    tokens = property(lambda self: self.guard.tokens, lambda self, v: setattr(self.guard, "tokens", v))
    errors = property(lambda self: self.guard.errors, lambda self, v: setattr(self.guard, "errors", v))
    hour = property(lambda self: self.guard.hour, lambda self, v: setattr(self.guard, "hour", v))
    paused_until = property(
        lambda self: self.guard.paused_until, lambda self, v: setattr(self.guard, "paused_until", v)
    )
    last_decision_t = property(
        lambda self: self.guard.last_decision_t, lambda self, v: setattr(self.guard, "last_decision_t", v)
    )

    def on(self):
        return bool(self.cfg.reid_enabled and self.cfg.reid_consent)

    def status(self):
        now = self.clock()
        refs = self.store.valid(now, self.cfg.reid_ttl)
        deadlines = [
            self.store.active_expiries.get(r.conversation_id, r.t + self.cfg.reid_ttl)
            for r in refs
            if not (self.store.active_connected and r.conversation_id == self.store.active_cid)
        ]
        return {
            "enabled": self.on(),
            "session": self.session,
            "key_set": key_status()["set"],
            "phase": self.phase,
            "refs": len(set(r.conversation_id for r in refs)),
            "ttl_left": round(max(0, min(deadlines) - now), 0) if deadlines else None,
            "calls": self.calls,
            "tokens": self.tokens,
            "cost_est": round(self.calls * self.cfg.reid_cost_per_call, 4),
            "conversation_id": self.conversation_id,
            "last": self.last,
        }

    def _cancel(self):
        self.epoch += 1
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        for task in (self.task, self.deadline_task):
            if task and not task.done() and task is not current:
                task.cancel()
        self.task = self.deadline_task = None
        self.queries.clear()
        self.expected.clear()

    def _store_wiped(self):
        self._cancel()
        self.members.clear()
        self.member_ids.clear()
        self.view_probe = self.view_ref = None
        self.phase = "idle"
        self.committed = True

    def wipe(self, reason):
        n = self.store.wipe()
        self.emit({"type": "reid_view_clear"})
        if n:
            self.event("reid_store", action="wipe", reason=reason, count=n)

    def _refs_for_members(self, now):
        return [
            Ref(self.session, self.conversation_id, jpeg, now, self.member_ids.get(track, f"{self.session}:{track}"))
            for track, (_quality, jpeg, _t) in self.members.items()
        ]

    def _commit_best(self, now):
        if not self.committed and self.conversation_id:
            self.store.commit(self._refs_for_members(now), int(self.cfg.reid_max_refs), now)
            self.event("reid_store", action="add", session=self.session, count=len(self.store.refs))
        self.committed = True
        self.members.clear()

    def _decide(self, decision, reason, now, ref=None, score=None):
        if self.phase == "idle" and self.last and self.last["session"] == self.session:
            return
        previous = self.conversation_id
        if decision == "resume" and ref:
            self.conversation_id = ref.conversation_id
            self.store.activate(ref.conversation_id)
        self.phase = "idle"
        self.guard.decided(now)
        if self.deadline_task:
            self.deadline_task.cancel()
        d = {
            "type": "handoff",
            "session": self.session,
            "conversation_id": self.conversation_id,
            "previous_conversation_id": previous,
            "decision": decision,
            "resumed_from_session": ref.session if decision == "resume" and ref else None,
            "score": None if score is None else round(score, 3),
            "reason": reason,
            "calls": self.calls,
        }
        self.last = {k: d[k] for k in ("session", "decision", "resumed_from_session", "score", "reason")}
        self.event("reid_decision", **{k: v for k, v in d.items() if k != "type"})
        self.emit(d)
        probe, vr = self.view_probe, self.view_ref
        self.view_probe = self.view_ref = None
        if self.cfg.reid_show_crops:
            r = ref if ref is not None else (vr[1] if vr else None)
            self.emit(
                {
                    "type": "reid_view",
                    "session": self.session,
                    "decision": decision,
                    "reason": reason,
                    "score": d["score"] if d["score"] is not None else (round(vr[0], 3) if vr else None),
                    "threshold": self.cfg.reid_threshold,
                    "resumed_from_session": d["resumed_from_session"],
                    "ref_session": r.session if r else None,
                    "conversation_id": self.conversation_id,
                    "t": time.time(),
                    "probe": thumb(probe),
                    "ref": thumb(r.jpeg if r else None),
                }
            )

    def _budget(self, now):
        key_missing = not key_status()["set"] and isinstance(self.client, JevClient)
        return self.guard.block_reason(now, self.cfg.reid_cap_per_hour, self.cfg.reid_cap_total, key_missing)

    def _alive(self, session, epoch, generation):
        return (
            session == self.session
            and epoch == self.epoch
            and generation == self.store.generation
            and not self.ended
            and self.on()
            and self.phase != "idle"
        )

    def _remaining(self):
        return self.cfg.reid_decision_timeout - (self.clock() - self.started)

    async def _deadline(self, session, epoch, generation):
        await asyncio.sleep(max(0, self._remaining()))
        if self._alive(session, epoch, generation):
            if self.task and not self.task.done():
                self.task.cancel()
            self._decide("new", "decision_timeout", self.clock())

    def observe(self, snap, now=None):
        now = self.clock() if now is None else now
        on = self.on()
        if self.was_on and not on:
            self.wipe("disabled")
        self.was_on = on
        if not on:
            return []
        n = self.store.expire(now, self.cfg.reid_ttl)
        if n:
            self.event("reid_store", action="expire", count=n)
        sess, state = int(snap.get("session") or 0), snap.get("state")
        if state == "ENDED":
            if sess == self.session and not self.ended:
                self._cancel()
                self.phase = "idle"
                self.ended = True
                self._commit_best(now)
            return []
        if sess >= 1 and sess != self.session:
            self._cancel()
            # A tracker reset without ENDED is interruption, never a fabricated normal ending.
            if not self.committed:
                self.interrupt()
            self.members = {}
            self.member_ids = {}
            self.session, self.committed, self.ended = sess, False, False
            self.started, self.first_part_t, self.wait_why = now, None, {}
            self.retried = False
            self.conversation_id = uuid.uuid4().hex
            self.phase = "await_query"
            refs = self.store.valid(now, self.cfg.reid_ttl)
            if not refs:
                self._decide("new", "no_reference", now)
            elif self.guard.cooling_down(now, self.cfg.reid_cooldown):
                self._decide("new", "cooldown", now)
            else:
                self.event("reid_start", session=sess, refs=len(refs))
                self.deadline_task = asyncio.get_running_loop().create_task(
                    self._deadline(sess, self.epoch, self.store.generation)
                )
        if self.ended or sess < 1:
            return []
        parts = [t for t in snap.get("tracks", []) if t.get("participant")][:3]
        if parts and self.first_part_t is None:
            self.first_part_t = now
        good = [(face_quality(t), t) for t in parts]
        good = [(q, t) for q, t in good if q > 0 and t["bbox"][2] - t["bbox"][0] >= self.cfg.reid_min_face]
        if self.phase != "idle" and self._remaining() <= 0:
            if self.task and not self.task.done():
                self.task.cancel()
            self.event(
                "reid_wait",
                session=self.session,
                result="decision_timeout",
                waited_s=round(now - self.started, 1),
                participant_seen=self.first_part_t is not None,
                **(self.wait_why or {"why": "no_participant"}),
            )
            self._decide("new", "decision_timeout", now)
        participant_ids = {str(track["id"]) for track in parts}
        if self.phase == "calling" and participant_ids - self.expected:
            # Membership changed while remote comparisons were in flight. Invalidate
            # that result set and compare the expanded group under the original deadline.
            queries, expected = dict(self.queries), set(self.expected)
            self._cancel()
            self.queries, self.expected = queries, expected | participant_ids
            self.view_probe = self.view_ref = None
            self.phase = "await_query"
            self.deadline_task = asyncio.get_running_loop().create_task(
                self._deadline(self.session, self.epoch, self.store.generation)
            )
            self.event("reid_members_changed", session=self.session, participants=len(self.expected))
        waiting = self.phase in ("await_query", "await_retry")
        if waiting:
            self.expected.update(str(t["id"]) for t in parts)
            if parts and len(good) < len(parts):
                t = next((t for t in parts if t not in [g[1] for g in good]), parts[0])
                b = t.get("bbox") or [0, 0, 0, 0]
                why = (
                    "facing_away"
                    if t.get("facing") == "away"
                    else "no_face"
                    if t.get("has_face") is False
                    else "predicted"
                    if t.get("visible_kind") == "predicted"
                    else "face_too_small"
                    if b[2] - b[0] < self.cfg.reid_min_face
                    else "low_quality"
                )
                self.wait_why = {"why": why, "face_w": round(b[2] - b[0], 3)}
            face_wait = self.first_part_t is not None and now - self.first_part_t >= self.cfg.reid_face_wait
            if face_wait and not self.queries and not good:
                self.event(
                    "reid_wait",
                    session=self.session,
                    result="no_face_timeout",
                    waited_s=round(now - self.started, 1),
                    participant_seen=True,
                    **self.wait_why,
                )
                self._decide("new", "no_face_timeout", now)
                waiting = False
            elif face_wait and self.queries and not ({str(t["id"]) for _q, t in good} - self.queries.keys()):
                self._launch()
                waiting = False
        reqs = []
        for q, track in good:
            tid = str(track["id"])
            if waiting and tid not in self.queries:
                reqs.append((f"query:{self.session}:{self.epoch}:{tid}", track["bbox"], q))
            prev = self.members.get(tid)
            if not self.committed and (prev is None or (now - prev[2] >= 1 and q > prev[0] * 1.1)):
                reqs.append((f"best:{self.session}:{self.epoch}:{tid}", track["bbox"], q))
        return reqs

    def on_crop(self, kind, jpeg, now=None, quality=0.0):
        now = self.clock() if now is None else now
        if not jpeg or not self.on() or self.ended or self.committed:
            return
        bits = kind.split(":", 3)
        if len(bits) == 4:
            name, session, epoch, tid = bits
            if name not in ("best", "query"):
                return
            if session != str(self.session) or epoch != str(self.epoch):
                return
        elif kind in ("best", "query"):
            name, tid = kind, "1"
        else:
            return
        if tid not in self.members and len(self.members) >= 3:
            return
        previous = self.members.get(tid)
        if previous is None or quality >= previous[0]:
            self.members[tid] = (quality, jpeg, now)
        if name == "query" and self.phase in ("await_query", "await_retry"):
            if self._remaining() <= 0:
                self._decide("new", "decision_timeout", now)
                return
            self.queries[tid] = jpeg
            if self.expected <= self.queries.keys():
                self._launch()

    def _launch(self):
        self.phase = "calling"
        refs = self.store.valid(self.clock(), self.cfg.reid_ttl)
        self.task = asyncio.get_running_loop().create_task(
            self._run(dict(self.queries), refs, self.session, self.epoch, self.store.generation)
        )

    async def _run(self, queries, refs, session, epoch, generation):
        tally = MatchTally(self.cfg.reid_threshold, self.cfg.reid_uncertain_low)
        for tid, query in queries.items():
            for ref in refs:
                if not self._alive(session, epoch, generation):
                    return
                now = self.clock()
                remaining = self._remaining()
                if remaining <= 0:
                    return self._finish(session, epoch, generation, "new", "decision_timeout", now)
                why = self._budget(now)
                if why:
                    self.event("reid_skip", reason=why, session=session)
                    return self._finish(session, epoch, generation, "new", why, now)
                t0 = time.perf_counter()
                timeout = min(self.cfg.reid_timeout, remaining)
                # Reserve the call before awaiting: cancellation cannot evade caps.
                self.guard.reserve(now)
                try:
                    score, meta = await asyncio.wait_for(
                        self.client.compare(ref.jpeg, query, model=self.cfg.reid_model, timeout=timeout), timeout
                    )
                except asyncio.CancelledError:
                    return
                except Exception as error:
                    if not self._alive(session, epoch, generation):
                        return
                    kind = (
                        "timeout" if isinstance(error, TimeoutError) else getattr(error, "kind", type(error).__name__)
                    )
                    opened = self.guard.failed(now)
                    self.event("reid_error", "warning", kind=kind, session=session, errors=self.errors)
                    if opened:
                        self.event("reid_breaker", "warning", action="open", pause_s=self.guard.breaker_pause)
                    return self._finish(session, epoch, generation, "new", "error", self.clock())
                if not self._alive(session, epoch, generation):
                    return
                self.guard.succeeded(meta.get("in_tok") or 0)
                if self._remaining() <= 0:
                    return self._finish(session, epoch, generation, "new", "decision_timeout", self.clock())
                if self.view_ref is None or score > self.view_ref[0]:
                    self.view_probe, self.view_ref = query, (score, ref)
                self.event(
                    "reid_call",
                    session=session,
                    ref_session=ref.session,
                    score=round(score, 3),
                    latency_ms=round((time.perf_counter() - t0) * 1000),
                    model=meta.get("model"),
                    in_tok=meta.get("in_tok"),
                    calls=self.calls,
                    cost_est=round(self.calls * self.cfg.reid_cost_per_call, 4),
                )
                tally.add(tid, ref, score)
        now = self.clock()
        verdict = decide(tally, self.retried)
        if verdict.decision == "retry":
            # Only uncertain scores: compare once more with fresh crops (same deadline).
            if self._alive(session, epoch, generation):
                self.retried, self.phase = True, "await_retry"
                self.queries.clear()
                self.event("reid_retry", session=session, score=round(verdict.score, 3))
            return
        if verdict.decision == "resume":
            self.member_ids = verdict.identities
        return self._finish(
            session, epoch, generation, verdict.decision, verdict.reason, now, verdict.ref, verdict.score
        )

    def _finish(self, session, epoch, generation, decision, reason, now, ref=None, score=None):
        if self._alive(session, epoch, generation):
            if decision == "resume" and not any(
                candidate is ref for candidate in self.store.valid(now, self.cfg.reid_ttl)
            ):
                self.member_ids.clear()
                return self._decide("new", "reference_expired", now)
            self._decide(decision, reason, now, ref, score)

    def interrupt(self):
        """Preserve current participant references without inventing an ENDED timestamp."""
        self._cancel()
        if not self.committed and self.conversation_id:
            cid = self.conversation_id
            # Keep the original retention lease across repeated interruptions.
            existing = {r.participant: r for r in self.store.active if r.conversation_id == cid}
            for ref in self._refs_for_members(self.clock()):
                previous = existing.get(ref.participant)
                if previous is not None:
                    previous.jpeg = b""
                existing[ref.participant] = ref
            selected = list(existing.values())[-3:]
            self.store._clear([r for r in existing.values() if r not in selected])
            self.store.active = [r for r in self.store.active if r.conversation_id != cid] + selected
            self.store.interrupt(cid, self.clock(), self.cfg.reid_ttl)
            self.store.trim(int(self.cfg.reid_max_refs))
        self.members.clear()
        self.ended, self.committed, self.phase = True, True, "idle"
        self.view_probe = self.view_ref = None
        self.emit({"type": "reid_view_clear"})

    def close(self):
        self.wipe("conn_close")

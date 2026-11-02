"""Accepted PoC identity contracts, using deterministic in-memory cloud doubles."""

import asyncio
import time

from presencekit.reid import CrossSessionReid, Ref, ReferenceStore
from presencekit.tracker import Config


def face(tid, usable=True):
    return {
        "id": tid,
        "participant": True,
        "visible": True,
        "has_face": usable,
        "bbox": [0.2, 0.2, 0.5, 0.6],
        "conf": 0.9,
    }


def snap(session, tracks=(), state="ACTIVE"):
    return {"session": session, "state": state, "tracks": list(tracks)}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Backend:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = []

    async def compare(self, ref, query, **kwargs):
        self.calls.append((ref, query, kwargs))
        result = next(self.scores)
        if isinstance(result, Exception):
            raise result
        return result, {}


def setup(scores=(), store=None, clock=None):
    cfg = Config()
    cfg.update({"reid_enabled": True, "reid_consent": True, "reid_cooldown": 0, "reid_show_crops": False})
    out = []
    driver = CrossSessionReid(cfg, Backend(scores), emit=out.append, clock=clock or Clock(), store=store)
    return driver, out


async def feed(driver, snapshot, wait=True):
    for kind, _box, quality in driver.observe(snapshot):
        driver.on_crop(kind, kind.encode(), quality=quality)
    if wait and driver.task:
        await driver.task


def test_nonrepresentative_member_can_resume_and_all_comparisons_complete():
    async def go():
        driver, out = setup([0.1, 0.95])
        await feed(driver, snap(1, [face(1), face(2)]))
        cid = driver.conversation_id
        await feed(driver, snap(1, state="ENDED"))
        assert len(driver.store.refs) == 2
        await feed(driver, snap(2, [face(3)]))
        assert out[-1]["decision"] == "resume" and driver.conversation_id == cid
        assert len(driver.client.calls) == 2
        assert out[-1]["previous_conversation_id"] != cid
        before = len(out)
        await feed(driver, snap(2, [face(3)]))
        assert len(out) == before  # exactly one handoff
        driver.close()

    asyncio.run(go())


def test_all_queries_are_compared_and_conflicting_conversations_do_not_merge():
    async def go():
        clock = Clock()
        store = ReferenceStore()
        store.add(Ref(1, "A", b"A", clock(), "alice"), 3)
        store.add(Ref(2, "B", b"B", clock(), "bob"), 3)
        # Newest first: query 3 matches B; query 4 matches A. First match cannot win.
        driver, out = setup([0.99, 0.0, 0.0, 0.99], store, clock)
        await feed(driver, snap(3, [face(3), face(4)]))
        assert len(driver.client.calls) == 4
        assert out[-1]["reason"] == "ambiguous" and out[-1]["decision"] == "new"
        assert out[-1]["conversation_id"] == out[-1]["previous_conversation_id"]
        driver.close()

    asyncio.run(go())


def test_match_followed_by_error_never_chooses_partial_result():
    async def go():
        clock = Clock()
        store = ReferenceStore()
        store.add(Ref(1, "A", b"A", clock()), 3)
        store.add(Ref(2, "B", b"B", clock()), 3)
        driver, out = setup([0.99, RuntimeError("backend failed")], store, clock)
        await feed(driver, snap(3, [face(3)]))
        assert out[-1]["decision"] == "new" and out[-1]["reason"] == "error"
        driver.close()

    asyncio.run(go())


def test_late_crop_rejected_after_new_session_end_and_wipe():
    async def go():
        driver, _out = setup()
        old = driver.observe(snap(1, [face(1)]))
        await feed(driver, snap(1, state="ENDED"))
        await feed(driver, snap(2, [face(2)]))
        before = dict(driver.members)
        for kind, _box, q in old:
            driver.on_crop(kind, b"stale", quality=q)
        assert driver.members == before
        current = driver.observe(snap(2, [face(3)]))
        driver.wipe("clear")
        for kind, _box, q in current:
            driver.on_crop(kind, b"stale", quality=q)
        assert not driver.members and not driver.store.refs
        driver.close()

    asyncio.run(go())


def test_history_cap_counts_conversations_and_wipes_evicted_faces():
    store = ReferenceStore()
    evicted = [Ref(1, "A", b"A1", 10, "1"), Ref(1, "A", b"A2", 10, "2")]
    store.commit(evicted, 3, 10)
    for i, cid in enumerate(("B", "C", "D"), 2):
        store.commit([Ref(i, cid, cid.encode(), 10 + i, str(i))], 3, 10 + i)
    assert set(r.conversation_id for r in store.refs) == {"B", "C", "D"}
    assert all(not r.jpeg for r in evicted)
    expired = list(store.refs)
    store.expire(1000, 600)
    assert not store.refs and all(not r.jpeg for r in expired)


def test_resume_refreshes_all_members_only_at_normal_end():
    async def go():
        clock = Clock()
        store = ReferenceStore()
        store.commit([Ref(1, "A", b"a", clock(), "alice"), Ref(1, "A", b"b", clock(), "bob")], 3, clock())
        clock.t += 590
        driver, _out = setup([0.95, 0.0], store, clock)
        await feed(driver, snap(2, [face(5)]))
        assert not store.refs and len(store.active) == 2
        assert all(r.t == 1000 for r in store.active)
        clock.t += 100  # active conversation survives old history expiry
        await feed(driver, snap(2, [face(5)]))
        assert len(store.active) == 2
        await feed(driver, snap(2, state="ENDED"))
        assert len(store.refs) == 2 and all(r.t == clock() for r in store.refs)
        assert {r.participant for r in store.refs} == {"alice", "bob"}
        driver.close()

    asyncio.run(go())


def test_interrupt_reconnect_retains_faces_and_does_not_refresh_history():
    async def go():
        clock = Clock()
        driver, _out = setup(clock=clock)
        driver.store.add(Ref(9, "old", b"old", 999, "old-member"), 3)
        driver.client = Backend([0.0])
        await feed(driver, snap(10, [face(1)]))
        cid = driver.conversation_id
        driver.interrupt()
        store = driver.store
        assert store.refs[0].t == 999 and len(store.active) == 1
        new, out = setup([0.99, 0.0], store, clock)
        await feed(new, snap(11, [face(1)]))
        assert out[-1]["decision"] == "resume" and new.conversation_id == cid
        assert store.refs[0].t == 999
        new.close()

    asyncio.run(go())


def test_deadline_cancels_hanging_backend_without_more_observations():
    class Hanging:
        async def compare(self, *_args, **_kwargs):
            await asyncio.sleep(60)

    async def go():
        clock = time.monotonic
        store = ReferenceStore()
        store.add(Ref(1, "A", b"A", clock()), 3)
        driver, out = setup(store=store, clock=clock)
        driver.cfg.reid_decision_timeout = 0.04
        driver.cfg.reid_timeout = 10
        driver.client = Hanging()
        started = time.monotonic()
        await feed(driver, snap(2, [face(1)]), wait=False)
        await asyncio.sleep(0.09)
        assert out[-1]["decision"] == "new" and out[-1]["reason"] == "decision_timeout"
        assert time.monotonic() - started < 0.3
        driver.close()

    asyncio.run(go())


def test_cancel_resistant_backend_cannot_handoff_after_wipe_or_end():
    class Late:
        def __init__(self):
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def compare(self, *_args, **_kwargs):
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()
            return 0.99, {}

    async def go(wipe):
        clock = Clock()
        store = ReferenceStore()
        store.add(Ref(1, "A", b"A", clock()), 3)
        driver, out = setup(store=store, clock=clock)
        driver.client = client = Late()
        await feed(driver, snap(2, [face(1)]), wait=False)
        task = driver.task
        await client.entered.wait()
        if wipe:
            store.wipe()  # global reset takes exactly this route
        else:
            driver.observe(snap(2, state="ENDED"))
        client.release.set()
        await task
        assert not [d for d in out if d.get("decision") == "resume"]
        assert driver.view_probe is None and driver.view_ref is None
        driver.close()

    asyncio.run(go(True))
    asyncio.run(go(False))


def test_unusable_member_wait_is_bounded_before_evaluating_available_faces():
    async def go():
        clock = Clock()
        store = ReferenceStore()
        store.add(Ref(1, "A", b"A", clock()), 3)
        driver, out = setup([0.95], store, clock)
        await feed(driver, snap(2, [face(1), face(2, usable=False)]))
        assert not driver.client.calls and driver.phase == "await_query"
        clock.t += driver.cfg.reid_face_wait
        await feed(driver, snap(2, [face(1), face(2, usable=False)]))
        assert out[-1]["decision"] == "resume"
        driver.close()

    asyncio.run(go())


def test_call_cap_after_partial_match_keeps_provisional_identity():
    async def go():
        clock = Clock()
        store = ReferenceStore()
        store.commit([Ref(1, "A", b"a", clock(), "1"), Ref(1, "A", b"b", clock(), "2")], 3, clock())
        driver, out = setup([0.99], store, clock)
        driver.cfg.reid_cap_total = 1
        await feed(driver, snap(2, [face(1)]))
        assert len(driver.client.calls) == 1
        assert out[-1]["reason"] == "cap" and out[-1]["decision"] == "new"
        assert out[-1]["conversation_id"] == out[-1]["previous_conversation_id"]
        driver.close()

    asyncio.run(go())


def test_nine_historical_faces_fit_three_conversations_and_current_is_bounded():
    async def go():
        clock = Clock()
        store = ReferenceStore()
        for session in (1, 2, 3):
            store.commit([Ref(session, str(session), b"face", clock(), str(t)) for t in (1, 2, 3)], 3, clock())
        assert len(store.refs) == 9
        driver, _out = setup([0.0] * 27, store, clock)
        await feed(driver, snap(4, [face(t) for t in (1, 2, 3, 4)]))
        assert len(driver.members) == 3 and len(driver.client.calls) == 27
        assert driver.status()["refs"] == 3
        driver.close()

    asyncio.run(go())


def test_uncertain_other_conversation_blocks_resume_but_same_conversation_does_not():
    async def go(other):
        clock = Clock()
        store = ReferenceStore()
        store.commit([Ref(1, "A", b"a", clock(), "alice")], 3, clock())
        store.commit([Ref(2, "B" if other else "A", b"b", clock(), "bob")], 3, clock())
        driver, out = setup([0.95, 0.6], store, clock)
        await feed(driver, snap(3, [face(1)]))
        assert out[-1]["decision"] == ("new" if other else "resume")
        assert out[-1]["reason"] == ("ambiguous" if other else "match")
        driver.close()

    asyncio.run(go(True))
    asyncio.run(go(False))


def test_participant_join_during_call_invalidates_inflight_unique_match():
    class DelayedFirst(Backend):
        def __init__(self):
            super().__init__([0.0, 0.99, 0.99, 0.0])
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.initial = True

        async def compare(self, ref, query, **kwargs):
            if self.initial:
                self.initial = False
                self.entered.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    await self.release.wait()
                return 0.99, {}
            return await super().compare(ref, query, **kwargs)

    async def go():
        clock = Clock()
        store = ReferenceStore()
        store.add(Ref(1, "A", b"A", clock(), "alice"), 3)
        store.add(Ref(2, "B", b"B", clock(), "bob"), 3)
        driver, out = setup(store=store, clock=clock)
        client = driver.client = DelayedFirst()
        await feed(driver, snap(3, [face(1)]), wait=False)
        initial_task, original_start = driver.task, driver.started
        await client.entered.wait()
        clock.t += 3.5  # membership changes after face-wait; original 10 s deadline still applies
        await feed(driver, snap(3, [face(1), face(2)]), wait=False)
        assert driver.started == original_start
        assert driver.queries.keys() == {"1", "2"}
        await driver.task
        client.release.set()
        await initial_task
        assert len(client.calls) == 4
        assert driver.calls == 5  # cancelled remote work still consumes budget
        assert out[-1]["decision"] == "new" and out[-1]["reason"] == "ambiguous"
        count = driver.calls
        await feed(driver, snap(3, [face(1), face(2), face(3)]))
        assert driver.calls == count  # a decision is final for this session
        driver.close()

    asyncio.run(go())


def test_retry_can_resume_when_other_conversations_are_definite_nonmatches():
    async def go():
        clock = Clock()
        store = ReferenceStore()
        for session in (1, 2, 3):
            store.add(Ref(session, str(session), b"face", clock(), "member"), 3)
        driver, out = setup([0.6, 0.0, 0.0, 0.95, 0.0, 0.0], store, clock)
        await feed(driver, snap(4, [face(1)]))
        assert driver.phase == "await_retry"
        await feed(driver, snap(4, [face(1)]))
        assert out[-1]["decision"] == "resume" and driver.calls == 6
        driver.close()

    asyncio.run(go())


def test_reference_expiring_during_compare_cannot_resume_even_if_result_is_high():
    class Delayed:
        def __init__(self):
            self.entered, self.release = asyncio.Event(), asyncio.Event()

        async def compare(self, *_args, **_kwargs):
            self.entered.set()
            await self.release.wait()
            return 0.99, {}

    async def go(explicit_expire):
        clock = Clock()
        store = ReferenceStore()
        ref = Ref(1, "A", b"A", 1000, "alice")
        store.add(ref, 3)
        clock.t = 1599
        driver, out = setup(store=store, clock=clock)
        backend = driver.client = Delayed()
        await feed(driver, snap(2, [face(1)]), wait=False)
        await backend.entered.wait()
        clock.t = 1601
        if explicit_expire:
            store.expire(clock(), 600)
            assert not ref.jpeg
        backend.release.set()
        await driver.task
        assert out[-1]["decision"] == "new" and out[-1]["reason"] == "reference_expired"
        assert out[-1]["conversation_id"] == out[-1]["previous_conversation_id"]
        assert not driver.member_ids
        driver.close()

    asyncio.run(go(True))
    asyncio.run(go(False))


def test_repeated_resume_interrupt_never_renews_last_normal_end_lease():
    async def go():
        clock = Clock()
        store = ReferenceStore()
        store.add(Ref(1, "A", b"A", 1000, "alice"), 3)
        for session, now in ((2, 1500), (3, 1590)):
            clock.t = now
            driver, out = setup([0.99], store, clock)
            await feed(driver, snap(session, [face(1)]))
            assert out[-1]["decision"] == "resume"
            driver.interrupt()
            assert store.ended_at["A"] == 1000
            assert store.active_expiries["A"] == 1600
        retained = list(store.active)
        clock.t = 1601
        driver, out = setup([], store, clock)
        await feed(driver, snap(4, [face(1)]))
        assert out[-1]["reason"] == "no_reference"
        assert all(not ref.jpeg for ref in retained)
        driver.close()

    asyncio.run(go())


def test_never_ended_conversation_gets_one_fixed_first_interruption_lease():
    async def go():
        clock = Clock()
        driver, _out = setup(clock=clock)
        await feed(driver, snap(1, [face(1)]))
        cid, store = driver.conversation_id, driver.store
        clock.t = 1001
        driver.interrupt()
        assert cid not in store.ended_at
        assert store.active_expiries[cid] == 1601
        clock.t = 1599
        next_driver, out = setup([0.99], store, clock)
        await feed(next_driver, snap(2, [face(1)]))
        assert out[-1]["decision"] == "resume"
        next_driver.interrupt()
        assert store.active_expiries[cid] == 1601
        retained = list(store.active)
        store.expire(1602, 600)
        assert not store.active and not store.active_expiries
        assert all(not ref.jpeg for ref in retained)
        next_driver.close()

    asyncio.run(go())


def test_status_uses_fixed_interruption_expiry_and_omits_active_countdown():
    clock = Clock()
    driver, _ = setup(clock=clock)
    ref = Ref(1, "old", b"opaque", clock.t, "p")
    driver.store.add(ref, 3)
    driver.store.activate("old")
    assert driver.status()["ttl_left"] is None
    clock.t += 590
    ref.t = clock.t  # A fresh crop does not renew the normal-END lease.
    driver.store.interrupt("old", clock.t, driver.cfg.reid_ttl)
    assert driver.status()["ttl_left"] == 10
    clock.t += 11
    assert driver.status()["refs"] == 0


def test_status_carries_session_to_disambiguate_browser_decisions():
    async def go():
        driver, _ = setup()
        try:
            assert driver.status()["session"] == 0
            await feed(driver, snap(7, [face(1)]))
            assert driver.status()["session"] == 7
            driver.wipe("consent_revoked")
            assert driver.status()["session"] == 7
        finally:
            driver.close()

    asyncio.run(go())

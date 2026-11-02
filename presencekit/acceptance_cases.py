"""PoC contracts exercised against production logic, without devices, files or cloud calls.

Synthetic detector output tests behavior, not camera accuracy. Mock scores test re-ID
decisions, not Jev quality. Failed cases are intentional evidence of unimplemented contracts.
"""

from __future__ import annotations

import asyncio
import contextlib

import numpy as np

from .body import Body
from .config import Config
from .reid import CloudReid, CrossSessionReid, ReidError
from .tracker import Detection, PresenceTracker
from .web.continuity import ObservationClock, PageRegistry

PROFILE_CONFIG = {
    "tracker_mode": "kalman",
    "presence_model": "hmm",
    "intent_model": "score",
    "priority_mode": "vision",
    "join_seconds": 1.5,
    "grace_seconds": 10.0,
    "face_roi": (0.08, 0.02, 0.92, 0.98),
    "pose_roi": (0.0, 0.0, 1.0, 1.0),
    "audio_pipeline": "puresound",
    "asr_provider": "none",
    "reid_enabled": True,
    "reid_consent": True,
    "reid_show_crops": False,
    "reid_cooldown": 0.0,
    "reid_ttl": 600.0,
    "reid_max_refs": 3,
    "reid_decision_timeout": 10.0,
}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def result(name, passed, expected, actual, **details):
    return {
        "id": name,
        "suite": "functional",
        "status": "passed" if passed else "failed",
        "expected": expected,
        "actual": actual,
        "details": details,
    }


def _presence(seed, name, *, participants=1, remaining=3):
    clock = Clock()
    tracker = PresenceTracker(Config(**PROFILE_CONFIG), clock=clock)
    rng = np.random.default_rng(seed)
    boxes = {i: (0.16 + (i - 1) * 0.25, 0.3, 0.29 + (i - 1) * 0.25, 0.6) for i in (1, 2, 3)}

    def step(ids, speech=False):
        clock.t += 0.5
        dets = []
        for person in ids:
            box = tuple(np.array(boxes[person]) + rng.uniform(-0.0005, 0.0005, 4))
            embedding = np.eye(3)[person - 1]
            dets.append(Detection(box, 0.95, embedding, frontal=0.95))
        return tracker.update(dets, speaking=speech)

    group = [1, 2, 3] if name == "remaining_participant" else list(range(1, participants + 1))
    for _ in range(20):
        snap = step(group, True)
    joined = sum(t["participant"] for t in snap["tracks"])
    session = snap["session"]
    premature = 0
    elapsed = 0.0
    if name == "quiet_120s":
        sequence = [(group, False)] * 240
    elif name == "occlusion_3s":
        sequence = [([], False)] * 6 + [(group, False)] * 12
    elif name == "remaining_participant":
        sequence = [([remaining], False)] * 240
    elif name == "leave_20s":
        sequence = [([], False)] * 40
    else:
        raise ValueError(name)
    ended_at = None
    for ids, speech in sequence:
        elapsed += 0.5
        snap = step(ids, speech)
        if snap["state"] == "ENDED" and ended_at is None:
            ended_at = elapsed
        if name != "leave_20s" and (snap["state"] == "ENDED" or snap["session"] != session):
            premature += 1
    passed = joined == len(group) and (ended_at is not None if name == "leave_20s" else premature == 0)
    variant = f".people_{participants}" if name == "quiet_120s" and participants > 1 else ""
    if name == "remaining_participant" and remaining != 3:
        variant = f".remaining_{remaining}"
    return result(
        f"presence.{name}{variant}.seed_{seed}",
        passed,
        {"joined": len(group), "ended_within_s": 20} if name == "leave_20s" else {"false_endings": 0},
        {"joined": joined, "false_endings": premature, "ended_at_s": ended_at},
        seed=seed,
        simulated_seconds=elapsed + 10,
        initial_participants=len(group),
        remaining_participant=remaining if name == "remaining_participant" else None,
        input="synthetic detections and speech flags; production tracker",
    )


def _body_detection(person, cx, *, face=True, view="facing", jitter=0):
    """The same controlled shoulder/face cues exercised by test_intent.py, not image inference."""
    cy, width = 0.4, 0.25
    if view == "facing":
        left, right = cx + width * 0.35, cx - width * 0.35
    elif view == "away":
        left, right = cx - width * 0.35, cx + width * 0.35
    else:
        left, right = cx - 0.01, cx + 0.01
    points = {
        "l_shoulder": (left, cy + 0.12, 0.9),
        "r_shoulder": (right, cy + 0.12, 0.9),
        "l_hip": (cx, cy + 0.4, 0.8),
        "r_hip": (cx, cy + 0.4, 0.8),
    }
    if face:
        points.update(nose=(cx, cy, 0.9), l_eye=(cx + 0.015, cy - 0.01, 0.9), r_eye=(cx - 0.015, cy - 0.01, 0.9))
    body = Body((cx - width / 2, cy - 0.08, cx + width / 2, cy + 0.6), 0.8, points)
    box = tuple(value + jitter for value in (cx - 0.04, cy - 0.055, cx + 0.04, cy + 0.055))
    return Detection(
        box,
        0.9 if face else 0.75,
        np.eye(3)[person - 1] if face else None,
        face_bbox=box if face else None,
        body=body,
        frontal=0.95 if face and view == "facing" else 0.05,
    )


def _presence_cues(seed, name):
    clock, rng = Clock(), np.random.default_rng(seed)
    tracker = PresenceTracker(Config(**{**PROFILE_CONFIG, "pose_enabled": True}), clock=clock)

    def step(people, speech=False):
        clock.t += 0.5
        return tracker.update(people, speaking=speech)

    def person(identity, cx, **kwargs):
        return _body_detection(identity, cx, jitter=rng.uniform(-0.0005, 0.0005), **kwargs)

    for _ in range(20):
        snapshot = step([person(1, 0.4)], True)
    original_session = snapshot["session"]
    original_ids = {track["id"] for track in snapshot["tracks"] if track["participant"]}
    joined = len(original_ids) == 1
    is_departure = name in ("unjoined_bystander", "unjoined_passerby", "background_speech", "tv_audio_flag")
    # Expose the low-intent candidate alongside an established participant before that participant leaves.
    if name in ("unjoined_bystander", "unjoined_passerby"):
        for _ in range(12):
            snapshot = step([person(1, 0.4), person(2, 0.72, face=False, view="away")])
    candidate_joined = any(track["participant"] and track["id"] not in original_ids for track in snapshot["tracks"])
    ended_at, false_endings, new_sessions, pose_seen = None, 0, 0, False
    duration = 20 if is_departure else 3
    for index in range(int(duration * 2)):
        if name == "unjoined_bystander":
            detections = [person(2, 0.72, face=False, view="away")]
        elif name == "unjoined_passerby":
            detections = [person(2, 0.72 + index * 0.005, face=False, view="side")]
        elif is_departure:
            detections = []
        else:
            detections = [person(1, 0.4, face=name == "turned_head_3s", view="away")]
        # Continuous background flag and repeated TV-like onsets are both adversarial
        # tracker inputs; no audio content/source classification is being simulated.
        speech = name == "background_speech" or (name == "tv_audio_flag" and index % 6 < 3)
        snapshot = step(detections, speech)
        new_sessions += snapshot["session"] != original_session
        if is_departure:
            candidate_joined |= any(
                track["participant"] and track["id"] not in original_ids for track in snapshot["tracks"]
            )
            if snapshot["state"] == "ENDED" and ended_at is None:
                ended_at = (index + 1) * 0.5
        else:
            false_endings += snapshot["state"] == "ENDED"
            pose_seen |= any(track.get("has_face") is False and track["participant"] for track in snapshot["tracks"])
    if is_departure:
        passed = joined and not candidate_joined and ended_at is not None and new_sessions == 0
        expected = {"ended_within_s": 20, "unjoined_candidate_promotions": 0, "new_sessions": 0}
        actual = {
            "initial_joined": joined,
            "ended_at_s": ended_at,
            "candidate_joined": candidate_joined,
            "new_sessions": new_sessions,
        }
    else:
        # Return to frontal detection as well, to catch fragmentation that appears only on recovery.
        for _ in range(6):
            snapshot = step([person(1, 0.4)])
            false_endings += snapshot["state"] == "ENDED"
            new_sessions += snapshot["session"] != original_session
        passed = joined and false_endings == 0 and new_sessions == 0
        if name == "pose_only_3s":
            passed &= pose_seen
        expected = {"false_endings": 0, "new_sessions": 0, "same_participant_recovered": True}
        recovered = any(track["id"] in original_ids and track["participant"] for track in snapshot["tracks"])
        passed &= recovered
        actual = {
            "initial_joined": joined,
            "false_endings": false_endings,
            "new_sessions": new_sessions,
            "same_participant_recovered": recovered,
            "pose_only_participant_observed": pose_seen,
        }
    return result(
        f"presence.{name}.seed_{seed}",
        passed,
        expected,
        actual,
        seed=seed,
        input="production tracker; synthetic face/shoulder cues and speech boolean, no detector or audio classifier",
        config_overrides={"pose_enabled": True},
        limitation="tv_audio_flag is adversarial speaking=True with no participant, not TV/audio classification",
    )


class MockIdentity(CloudReid):
    """Opaque identity tokens in memory; never a network backend or actual face image."""

    def __init__(self, behavior="identity"):
        self.behavior = behavior
        self.release = asyncio.Event()

    async def compare(self, ref, query, **kwargs):
        if self.behavior == "stall":
            await self.release.wait()
        if self.behavior == "error":
            raise ReidError("synthetic_error")
        score = 0.6 if self.behavior == "uncertain" else (0.99 if ref == query else 0.01)
        return score, {}


def reid_snapshot(session, people, state="ACTIVE"):
    return {
        "session": session,
        "state": state,
        "tracks": [
            {
                "id": p,
                "bbox": [p / 10, 0.2, p / 10 + 0.15, 0.5],
                "conf": 0.95,
                "visible": True,
                "participant": True,
                "has_face": True,
            }
            for p in people
        ],
    }


async def _reid(name):
    clock, events, backend = Clock(), [], MockIdentity()
    driver = CrossSessionReid(
        Config(**PROFILE_CONFIG), backend, emit=events.append, event=lambda *args, **kwargs: None, clock=clock
    )

    async def drive(session, people, state="ACTIVE", wait=True):
        clock.t += 1
        for kind, box, quality in driver.observe(reid_snapshot(session, people, state)):
            person = round(box[0] * 10)
            driver.on_crop(kind, f"person-{person}".encode(), quality=quality)
        if wait and driver.task:
            await driver.task
        await asyncio.sleep(0)

    def handoffs(session):
        return [e for e in events if e.get("type") == "handoff" and e.get("session") == session]

    try:
        first_people = [1, 2] if name == "member_overlap" else [1]
        await drive(1, first_people)
        await drive(1, first_people)
        first_id = driver.conversation_id
        await drive(1, [], "ENDED")
        if name == "ambiguous_overlap":
            await drive(2, [2])
            await drive(2, [2])
            await drive(2, [], "ENDED")
            session, people = 3, [1, 2]
        else:
            session = 2
            people = [2] if name in ("different_person", "member_overlap") else [1]
        if name in ("error", "uncertain", "deadline"):
            backend.behavior = {"deadline": "stall"}.get(name, name)
        if name == "expired":
            clock.t += 600
        await drive(session, people, wait=name != "deadline")
        if name == "uncertain":
            await drive(session, people)
        if name == "deadline":
            clock.t += 11
            driver.observe(reid_snapshot(session, people))
            await asyncio.sleep(0)
        h = handoffs(session)
        resumed = [e for e in h if e.get("decision") == "resume"]
        if name in ("same_person", "member_overlap"):
            passed = len(resumed) == 1 and resumed[0]["conversation_id"] == first_id
            expected = {"resume_original_conversation": True, "new_session": session}
        else:
            passed = not resumed and bool(h) and driver.phase == "idle"
            expected = {"resume": False, "decision_completed": True}
        return result(
            f"reid.{name}",
            passed,
            expected,
            {"handoffs": len(h), "resumed": len(resumed), "phase": driver.phase},
            input="production re-ID driver with in-memory mock scores; no cloud or images",
        )
    finally:
        driver.close()
        if driver.task:
            with contextlib.suppress(asyncio.CancelledError):
                await driver.task


async def _same_page_reconnect():
    """Real registry leases and real re-ID driver, with opaque local identity tokens."""
    clock, events = Clock(), []
    registry = PageRegistry(clock)
    drivers = []

    def attach(page):
        driver = CrossSessionReid(
            Config(**PROFILE_CONFIG),
            MockIdentity(),
            emit=events.append,
            event=lambda *args, **kwargs: None,
            clock=clock,
            store=page.store,
        )
        page.interrupt = driver.interrupt
        drivers.append(driver)
        return driver

    def next_session(page):
        # Exercise the same counter restore and tracker session-start behavior as the transport.
        tracker = PresenceTracker(Config(**PROFILE_CONFIG), clock=clock)
        tracker.session = page.session
        person = Detection((0.4, 0.3, 0.55, 0.6), 0.95, np.array([1.0, 0, 0]), frontal=0.95)
        for _ in range(20):
            clock.t += 0.5
            snapshot = tracker.update([person], speaking=True)
        page.session = max(page.session, snapshot["session"])
        return snapshot["session"]

    async def drive(driver, session, people, state="ACTIVE"):
        clock.t += 1
        for kind, box, quality in driver.observe(reid_snapshot(session, people, state)):
            driver.on_crop(kind, f"person-{round(box[0] * 10)}".encode(), quality=quality)
        if driver.task:
            await driver.task
        await asyncio.sleep(0)

    try:
        page, old_lease = registry.acquire("a" * 32)
        first = attach(page)
        original_session = next_session(page)
        await drive(first, original_session, [1])
        await drive(first, original_session, [1])
        original_conversation = first.conversation_id
        await drive(first, original_session, [], "ENDED")

        # A separate ended conversation must retain its own expiry across reconnect/handoff.
        historical_session = next_session(page)
        await drive(first, historical_session, [2])
        await drive(first, historical_session, [2])
        other_conversation = first.conversation_id
        await drive(first, historical_session, [], "ENDED")
        historic_refs = list(page.store.refs)
        ended_times = {ref.conversation_id: ref.t for ref in historic_refs}
        # Resume the original conversation and disconnect while it is still active.
        # This must preserve it without turning interruption into a normal END.
        active_session = next_session(page)
        await drive(first, active_session, [1])
        await drive(first, active_session, [1])
        active_before_disconnect = not first.ended and first.conversation_id == original_conversation
        registry.release(page, old_lease)
        interrupted_reference_retained = any(ref.conversation_id == original_conversation for ref in page.store.active)
        no_fabricated_history = not any(ref.conversation_id == original_conversation for ref in page.store.refs)
        clock.t += 30
        reconnected_page, new_lease = registry.acquire("a" * 32)
        normal_end_times_unchanged = all(ref.t == ended_times[ref.conversation_id] for ref in historic_refs)
        second = attach(reconnected_page)
        new_session = next_session(reconnected_page)
        await drive(second, new_session, [1])
        await drive(second, new_session, [1])
        resumed = [
            event
            for event in events
            if event.get("type") == "handoff"
            and event.get("session") == new_session
            and event.get("decision") == "resume"
        ]
        current_state = (second.session, second.ended, second.phase, page.store.generation)
        stale_release_rejected = not registry.release(page, old_lease)
        current_owner_unchanged = current_state == (second.session, second.ended, second.phase, page.store.generation)
        current_owner_unchanged &= page.detached_at is None and page.generation == new_lease

        isolated_page, _ = registry.acquire("b" * 32)
        isolated_store_empty = not isolated_page.store.valid(clock.t, 600)
        isolated = attach(isolated_page)
        isolated_session = next_session(isolated_page)
        await drive(isolated, isolated_session, [1])
        await drive(isolated, isolated_session, [1])
        isolated_decision_new = isolated.last["decision"] == "new" and isolated.conversation_id != original_conversation

        clock.t = ended_times[other_conversation] + 600
        registry.prune()
        unrelated_history_expired = not any(ref.conversation_id == other_conversation for ref in page.store.refs)
        expired_media_cleared = all(not ref.jpeg for ref in historic_refs if ref.conversation_id == other_conversation)
        preserved_original = second.conversation_id == original_conversation and len(resumed) == 1
        actual = {
            "new_session": new_session > active_session > historical_session > original_session,
            "same_page_store": reconnected_page is page,
            "resumed_original_conversation": preserved_original,
            "stale_release_rejected": stale_release_rejected,
            "current_owner_unchanged": current_owner_unchanged,
            "other_page_isolated": isolated_store_empty and isolated_decision_new,
            "normal_end_ttl_not_refreshed": normal_end_times_unchanged and unrelated_history_expired,
            "expired_media_cleared": expired_media_cleared,
            "active_interruption_preserved": active_before_disconnect and interrupted_reference_retained,
            "interruption_not_normal_end": no_fabricated_history,
        }
        return result(
            "transport.same_page_reconnect",
            all(actual.values()),
            {key: True for key in actual},
            actual,
            input="production PageRegistry, PresenceTracker and CrossSessionReid; mock identities, virtual time",
        )
    finally:
        for driver in drivers:
            driver.close()
        await asyncio.sleep(0)


def _sensor_gap():
    wall = Clock()
    observed = ObservationClock(wall)
    tracker = PresenceTracker(Config(**PROFILE_CONFIG), clock=observed)
    person = Detection((0.4, 0.3, 0.55, 0.6), 0.95, np.array([1.0, 0, 0]), frontal=0.95)
    for _ in range(20):
        wall.t += 0.5
        snapshot = tracker.update([person], speaking=True)
    session = snapshot["session"]
    joined = any(track["participant"] for track in snapshot["tracks"])
    observed.pause()
    frozen = observed()
    ended = 0
    for _ in range(240):
        wall.t += 0.5
        snapshot = tracker.tick()
        ended += snapshot["state"] == "ENDED"
    frozen_during_gap = observed() == frozen
    observed.resume()
    wall.t += 0.5
    recovered = tracker.update([person], speaking=False)
    actual = {
        "participant_joined": joined,
        "false_departures": ended,
        "same_session_after_recovery": recovered["session"] == session,
        "not_ended_after_recovery": recovered["state"] != "ENDED",
        "observation_time_frozen_during_gap": frozen_during_gap,
        "observation_time_resumed": observed() == frozen + 0.5,
    }
    passed = ended == 0 and all(value for key, value in actual.items() if key != "false_departures")
    return result(
        "transport.sensor_gap",
        passed,
        {"false_departures": 0, "same_session_after_recovery": True},
        actual,
        input="production ObservationClock and PresenceTracker; virtual 120 s missing-input interval",
    )


def functional_cases(seeds=20):
    rows = [
        _presence(seed, scenario)
        for scenario in ("quiet_120s", "occlusion_3s", "remaining_participant", "leave_20s")
        for seed in range(seeds)
    ]
    rows.extend(_presence(seed, "quiet_120s", participants=people) for people in (2, 3) for seed in range(seeds))
    rows.extend(
        _presence(seed, "remaining_participant", remaining=person) for person in (1, 2) for seed in range(seeds)
    )
    rows.extend(
        _presence_cues(seed, scenario)
        for scenario in (
            "unjoined_bystander",
            "unjoined_passerby",
            "background_speech",
            "tv_audio_flag",
            "turned_head_3s",
            "pose_only_3s",
        )
        for seed in range(seeds)
    )
    rows.append(
        {
            "id": "presence.tv_visual_rejection",
            "suite": "functional",
            "status": "not_run",
            "expected": "A person depicted on a television must not join or keep the original conversation",
            "actual": None,
            "details": {
                "reason": "The production detector/intent interface has no screen-person semantic label. "
                "Synthetic detections cannot establish television-image rejection; "
                "requires a defined visual policy and labeled real input.",
                "covered_separately": "Background speech booleans cannot retain an absent participant in vision mode.",
            },
        }
    )
    for scenario in (
        "same_person",
        "different_person",
        "member_overlap",
        "ambiguous_overlap",
        "uncertain",
        "error",
        "expired",
        "deadline",
    ):
        rows.append(asyncio.run(_reid(scenario)))
    rows.append(asyncio.run(_same_page_reconnect()))
    rows.append(_sensor_gap())
    return rows

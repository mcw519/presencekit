"""Transport interruptions, capabilities and reference expiry are independent of departure."""

import numpy as np
from fastapi.testclient import TestClient

from presencekit.acceptance_cases import PROFILE_CONFIG, Clock
from presencekit.config import Config
from presencekit.reid import Ref
from presencekit.tracker import Detection, PresenceTracker
from presencekit.web.continuity import ObservationClock, PageRegistry


def test_observation_pause_does_not_turn_missing_input_into_departure():
    wall = Clock()
    observed = ObservationClock(wall)
    tracker = PresenceTracker(Config(**PROFILE_CONFIG), clock=observed)
    person = Detection((0.4, 0.3, 0.55, 0.6), 0.95, np.array([1.0, 0, 0]), frontal=0.95)
    for _ in range(20):
        wall.t += 0.5
        tracker.update([person], speaking=True)
    assert tracker.snapshot()["tracks"][0]["participant"]
    session = tracker.session
    observed.pause()
    frozen = observed()
    for _ in range(240):
        wall.t += 0.5
        tracker.tick()
    assert observed() == frozen
    assert tracker.session == session and tracker.state.value != "ENDED"
    observed.resume()
    wall.t += 0.5
    assert tracker.update([person], speaking=False)["state"] != "ENDED"
    assert observed() == frozen + 0.5


def test_same_page_retains_store_counter_without_refreshing_history_ttl():
    clock = Clock()
    registry = PageRegistry(clock)
    page, first = registry.acquire("a" * 32)
    page.session = 7
    ref = Ref(7, "conversation", b"opaque", clock.t, "person")
    page.store.add(ref, 3)
    registry.release(page, first)
    clock.t += 500
    resumed, second = registry.acquire("a" * 32)
    assert resumed is page and resumed.session == 7 and second != first
    assert ref.t == 1000
    assert not registry.release(page, first)  # late cleanup must not detach new owner
    assert page.detached_at is None
    clock.t += 100
    registry.prune()
    assert page.store.refs == [] and ref.jpeg == b""


def test_orphan_page_expiry_wipes_media_and_different_page_is_isolated():
    clock = Clock()
    registry = PageRegistry(clock)
    page, lease = registry.acquire("a" * 32)
    ref = Ref(1, "private", b"opaque", clock.t, "p")
    page.store.add(ref, 3)
    other, _ = registry.acquire("b" * 32)
    assert other.store.refs == []
    registry.release(page, lease)
    clock.t += 600
    registry.prune()
    assert "a" * 32 not in registry.pages and ref.jpeg == b""
    assert "b" * 32 in registry.pages


def test_replacing_transport_interrupts_old_work_and_late_release_is_ignored():
    registry = PageRegistry()
    page, old = registry.acquire("a" * 32)
    calls = []
    page.interrupt = lambda: calls.append("interrupt")
    page.disconnect = lambda: calls.append("disconnect")
    same, new = registry.acquire("a" * 32)
    assert calls == ["interrupt", "disconnect"] and same is page
    assert new > old and not registry.release(page, old)


def test_refresh_beacon_wipes_only_its_page_without_revealing_capability(monkeypatch):
    from presencekit import server
    from presencekit.web import continuity

    registry = PageRegistry()
    monkeypatch.setattr(continuity, "PAGES", registry)
    page, _ = registry.acquire("a" * 32)
    other, _ = registry.acquire("b" * 32)
    ref = Ref(1, "private", b"opaque", 0, "p")
    page.store.refs.append(ref)
    suffix = f"?token={server.TOKEN}" if server.TOKEN else ""
    with TestClient(server.app) as client:
        response = client.post("/api/reid_page_close" + suffix, json={"page_token": "a" * 32})
        assert response.json() == {"ok": True}
        assert client.post("/api/reid_page_close" + suffix, json={"page_token": "invalid"}).json() == {"ok": True}
        assert "a" * 32 not in registry.pages and "b" * 32 in registry.pages
        assert ref.jpeg == b"" and other.store.refs == []
    assert registry.pages == {}


def test_registry_capacity_does_not_evict_live_page():
    registry = PageRegistry(limit=1)
    page, lease = registry.acquire("a" * 32)
    import pytest

    with pytest.raises(ValueError, match="capacity"):
        registry.acquire("b" * 32)
    registry.release(page, lease)
    other, _ = registry.acquire("b" * 32)
    assert other is not page and "a" * 32 not in registry.pages


def test_delayed_sensor_pause_excludes_host_sleep_without_rewinding():
    wall = Clock()
    observed = ObservationClock(wall)
    tracker = PresenceTracker(Config(**PROFILE_CONFIG), clock=observed)
    from presencekit.tracker import State

    tracker.state = State.AWAY
    tracker.state_since = observed()
    last_frame = wall.t
    wall.t += 120
    observed.pause(at=last_frame + 3)
    assert tracker.tick()["state"] != "ENDED"
    assert observed() == last_frame + 3
    observed.resume()
    assert observed() == last_frame + 3


def test_app_cleans_orphan_without_another_connection(monkeypatch):
    import time

    from presencekit import server
    from presencekit.web import continuity

    clock = Clock()
    registry = PageRegistry(clock)
    monkeypatch.setattr(continuity, "PAGES", registry)
    page, lease = registry.acquire("a" * 32)
    ref = Ref(1, "private", b"opaque", clock.t, "p")
    page.store.add(ref, 3)
    registry.release(page, lease)
    with TestClient(server.app):
        clock.t += 601
        deadline = time.monotonic() + 3
        while ref.jpeg and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ref.jpeg == b"" and not registry.pages


def test_initializing_old_transport_cannot_replace_new_owner(monkeypatch):
    import asyncio
    import threading

    from presencekit.web import session

    registry = PageRegistry()
    monkeypatch.setattr(session, "PAGES", registry)
    monkeypatch.setattr(session, "TOKEN", "")
    entered, finish = threading.Event(), threading.Event()

    def pipeline():
        entered.set()
        assert finish.wait(3)
        return object()

    monkeypatch.setattr(session, "pipeline", pipeline)

    class Socket:
        query_params = {}
        headers = {"sec-websocket-protocol": "presence-v1,page-" + "a" * 32}
        client = None
        closed = None

        async def accept(self, **kwargs):
            pass

        async def close(self, code):
            self.closed = code

    async def run():
        sock = Socket()
        task = asyncio.create_task(session.ws(sock))
        assert await asyncio.to_thread(entered.wait, 3)
        page, lease = registry.acquire("a" * 32)

        def callback():
            pass

        page.interrupt = callback
        finish.set()
        await asyncio.wait_for(task, 3)
        assert sock.closed == 1000
        assert page.generation == lease and page.interrupt is callback
        assert page.detached_at is None

    asyncio.run(run())


def test_sensor_lease_caps_all_clock_readers_before_ticker_runs():
    wall = Clock()
    observed = ObservationClock(wall)
    tracker = PresenceTracker(Config(**PROFILE_CONFIG), clock=observed)
    from presencekit.tracker import State

    tracker.state = State.AWAY
    tracker.state_since = observed()
    observed.observed()
    wall.t += 120
    # snapshot/audio/config readers can run before the ticker after host sleep.
    tracker.snapshot()
    tracker.set_speech(False)
    assert observed.paused_at is not None
    assert tracker.tick()["state"] != "ENDED"
    assert observed() == 1003
    observed.observed()
    assert observed() == 1003
    wall.t += 0.5
    assert observed() == 1003.5


def test_new_frame_first_after_host_sleep_honors_previous_sensor_lease():
    wall = Clock()
    observed = ObservationClock(wall)
    observed.observed()
    wall.t += 120
    observed.observed()
    assert observed() == 1003
    wall.t += 0.5
    assert observed() == 1003.5


def test_page_subprotocol_is_masked_even_in_debug_handshake_logs():
    import logging

    from presencekit.diag import Redact

    cap = "a" * 32
    record = logging.LogRecord(
        "uvicorn.error", logging.DEBUG, "", 0, "< Sec-WebSocket-Protocol: %s", ("presence-v1, page-" + cap,), None
    )
    Redact().filter(record)
    assert cap not in record.getMessage()
    assert "page-***" in record.getMessage()


def test_websocket_subprotocol_reconnect_keeps_page_session_counter(monkeypatch):
    from presencekit import server
    from presencekit.web import continuity, session

    registry = PageRegistry()
    monkeypatch.setattr(continuity, "PAGES", registry)
    monkeypatch.setattr(session, "PAGES", registry)
    suffix = f"?token={server.TOKEN}" if server.TOKEN else ""
    protocols = ["presence-v1", "page-" + "a" * 32]

    def start(ws):
        ws.send_json({"type": "config", "config": {"priority_mode": "balanced"}})
        ws.send_json({"type": "start"})
        ws.send_json({"type": "vad", "speech": True})
        for _ in range(30):
            msg = ws.receive_json()
            if msg.get("type") == "state" and msg.get("speaking"):
                return msg["session"]
        raise AssertionError("No active speech session")

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws" + suffix, subprotocols=protocols) as ws:
            assert ws.accepted_subprotocol == "presence-v1"
            first = start(ws)
        with client.websocket_connect("/ws" + suffix, subprotocols=protocols) as ws:
            second = start(ws)
        assert first == 1 and second == 2

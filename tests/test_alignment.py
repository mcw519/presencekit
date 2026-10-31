"""Decision-aligned vision timestamps and conservative freshness without clock mixing."""

import base64
import json
import math
import struct
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from presencekit.web.alignment import VisionTiming, client_timestamp, decision_timing


@pytest.mark.parametrize("value", [None, True, False, -1, math.inf, -math.inf, math.nan, "100"])
def test_invalid_client_timestamps_are_unavailable(value):
    assert client_timestamp(value) is None


def test_low_fps_capture_phase_not_inference_delay():
    vision = VisionTiming(1000, received=50, submitted=50.001, completed=50.02)
    age, timing = decision_timing(vision, 2370, 51.37, 1)
    assert age == pytest.approx(1.37)
    assert timing["capture_gap_ms"] == 1370
    assert timing["processing_ms"] == pytest.approx(19)
    assert timing["stale"]


def test_slow_completed_frame_does_not_become_fresh_again():
    vision = VisionTiming(1000, received=20, submitted=20.2, completed=22)
    age, timing = decision_timing(vision, 1100, 22.01, 1)
    assert age == pytest.approx(2.01)
    assert timing["processed_age_ms"] == pytest.approx(10)
    assert timing["receive_to_processed_ms"] == 2000
    assert timing["queue_wait_ms"] == pytest.approx(200)
    assert timing["stale"]


def test_missing_timestamp_falls_back_to_server_residence_without_clock_subtraction():
    age, timing = decision_timing(VisionTiming(None, 500, 500, 500.01), 987654321, 501, 1)
    assert age == 1
    assert timing["capture_gap_ms"] is None
    assert not timing["timestamp_available"]
    assert not timing["stale"]  # same strict > boundary as acoustic cross-check


def test_future_video_for_backlogged_audio_cannot_cancel_server_age():
    age, timing = decision_timing(VisionTiming(3000, 100, 100.1, 100.2), 1000, 101.5, 1)
    assert timing["capture_gap_ms"] == -2000
    assert age == 1.5 and timing["stale"]


def test_no_vision_is_stale_and_json_serializable():
    age, timing = decision_timing(None, 1, 2, 1)
    assert age == math.inf
    assert timing["stale"] and timing["effective_age_ms"] is None
    json.dumps(timing, allow_nan=False)


def test_audio_metadata_retains_decision_frame_when_new_vision_finishes(monkeypatch):
    from presencekit.server import app
    from presencekit.web import session

    started, release = threading.Event(), threading.Event()
    seen = []

    class FakeStream:
        def __init__(self, cfg):
            self.cfg = cfg

        def on_vision(self, snap, now):
            pass

        def process(self, pcm, snap, now, vis_age):
            seen.append((snap, vis_age))
            started.set()
            assert release.wait(5), "test did not release audio executor"
            return pcm, False

        def meta(self):
            return {"rtf": 0.1, "counts": {}}

    monkeypatch.setattr(session, "pipeline", lambda: SimpleNamespace(name="mock", info=lambda: {}))
    monkeypatch.setattr(session, "_vision_job", lambda *args: ([], {"skipped": False}))
    monkeypatch.setattr(session, "AudioStream", FakeStream)
    monkeypatch.setattr(session, "puresound_available", lambda: (True, ""))

    def wait_for(ws, predicate):
        for _ in range(40):
            message = ws.receive()
            if message.get("text"):
                data = json.loads(message["text"])
                assert data.get("type") != "error", data
                if predicate(data):
                    return data
        raise AssertionError("expected message not received")

    frame = base64.b64encode(b"\xff\xd8mock\xff\xd9").decode()
    try:
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "config", "config": {"audio_pipeline": "puresound"}})
            ws.send_json({"type": "start"})
            ws.send_json({"type": "frame", "data": frame, "t": 1000})
            wait_for(ws, lambda d: d.get("type") == "state" and "detections" in d)
            ws.send_bytes(b"\x01" + struct.pack("<d", 2370) + bytes(640))
            assert started.wait(5)
            ws.send_json({"type": "frame", "data": frame, "t": 2300})
            wait_for(ws, lambda d: d.get("type") == "state" and "detections" in d)
            release.set()
            meta = wait_for(ws, lambda d: d.get("type") == "audio" and d.get("available"))
            assert meta["t_vision_client"] == 1000
            assert meta["vision_timing"]["capture_gap_ms"] == 1370
            assert meta["vision_timing"]["effective_age_ms"] >= 1370
            assert seen[0][1] >= 1.37
    finally:
        release.set()

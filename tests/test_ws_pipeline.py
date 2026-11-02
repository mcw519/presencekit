"""Synthetic end-to-end test of the WebSocket pipeline with a generated image."""

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from presencekit.server import app


def blank_jpeg():
    img = np.full((480, 640, 3), 80, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def test_index_served():
    r = TestClient(app).get("/")
    assert r.status_code == 200 and "PresenceKit" in r.text


def test_ws_frame_and_vad_roundtrip():
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "config": {"priority_mode": "balanced"}})
        ws.send_json({"type": "start"})
        ws.send_json({"type": "vad", "speech": True})
        for _ in range(30):
            s = ws.receive_json()
            if s.get("speaking"):
                break
        assert s["state"] == "ACTIVE" and s["speaking"]
        ws.send_json({"type": "frame", "speech": False, "data": blank_jpeg()})
        for _ in range(5):
            s = ws.receive_json()
            if s.get("type") == "state" and "detections" in s:
                break
        assert s["detections"] == [] and "pipeline" in s
        ws.send_json({"type": "config", "config": {"empty_timeout": 7}})
        for _ in range(5):
            s = ws.receive_json()
            if s.get("config_ack"):
                break
        assert s["config"]["empty_timeout"] == 7.0


FACE = Path(__file__).parent / "data" / "face.jpg"


@pytest.mark.skipif(not FACE.exists(), reason="sample face image missing")
def test_ws_real_face_tracked_then_leaves():
    img = cv2.resize(cv2.imread(str(FACE)), (480, 480))
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, 80:560] = img
    face = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", frame)[1].tobytes()).decode()

    def frames(ws, data, n):
        out = None
        for _ in range(n):
            ws.send_json({"type": "frame", "speech": False, "data": data})
            while True:
                out = ws.receive_json()
                if out.get("type") == "state" and "detections" in out:
                    break
        return out

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "start"})
        s = frames(ws, face, 4)
        assert s["detections"], "no face detected"
        assert len(s["tracks"]) == 1 and s["state"] == "ACTIVE" and s["present"]
        s = frames(ws, blank_jpeg(), 4)
        assert s["empty"] and not s["present"]


def recv_until(ws, pred, n=30):
    for _ in range(n):
        m = ws.receive_json()
        if pred(m):
            return m
    raise AssertionError("message not received")


def test_stats_pushed_only_while_running_and_stop_resets():
    with TestClient(app).websocket_connect("/ws") as ws:
        s = recv_until(ws, lambda m: m.get("type") == "state")
        assert s["running"] is False and s["state"] == "ENDED"
        ws.send_json({"type": "start"})
        ws.send_json({"type": "vad", "speech": True})
        ws.send_json({"type": "frame", "speech": True, "data": blank_jpeg()})
        st = recv_until(ws, lambda m: m.get("type") == "stats")
        assert {"proc_cpu", "proc_rss", "sys_cpu", "mem_total", "load", "per_core"} <= set(st["system"])
        c = st["conn"]
        assert c["frames"]["received"] == 1 and c["frames"]["processed"] == 1
        assert {
            "decode",
            "motion",
            "detect",
            "embed",
            "pose",
            "tracker",
            "total",
            "queue_wait",
            "executor_wait",
            "lock_wait",
            "appearance",
        } == set(c["latency_ms"])
        assert {"skipped_pct", "embed_computed", "embed_reused", "recommended_fps"} <= set(c["optim"])
        assert c["latency_ms"]["detect"]["p95"] >= 0 and c["frame_size"] == [640, 480]
        assert st["info"]["opencv"] and st["info"]["pipeline"]
        ws.send_json({"type": "ping", "t": 1, "rtt": 12.5})
        assert recv_until(ws, lambda m: m.get("type") == "pong")["t"] == 1
        ws.send_json({"type": "stop"})
        s = recv_until(ws, lambda m: m.get("type") == "state" and m["running"] is False)
        assert s["state"] == "ENDED" and s["tracks"] == []
        ws.send_json({"type": "frame", "data": blank_jpeg()})
        ws.send_json({"type": "vad", "speech": True})
        s = recv_until(ws, lambda m: m.get("type") == "state")
        assert s["state"] == "ENDED"


def test_adaptive_fps_feedback_on_all_state_paths():
    with TestClient(app).websocket_connect("/ws") as ws:
        initial = recv_until(ws, lambda m: m.get("type") == "state")
        assert "recommended_fps" in initial
        ws.send_json({"type": "config", "config": {"adaptive_fps": True, "fps_low": 1.5, "fps_high": 4}})
        ack = recv_until(ws, lambda m: m.get("config_ack"))
        assert ack["recommended_fps"] == 1.5
        for command in ("start", "reset", "stop"):
            ws.send_json({"type": command})
            state = recv_until(ws, lambda m: m.get("type") == "state")
            assert state["recommended_fps"] == 1.5
        ws.send_json({"type": "config", "config": {"adaptive_fps": False}})
        assert recv_until(ws, lambda m: m.get("config_ack"))["recommended_fps"] == 0
        ws.send_json({"type": "start"})
        state = recv_until(ws, lambda m: m.get("type") == "state" and m.get("running"))
        assert state["recommended_fps"] == 0
        stats = recv_until(ws, lambda m: m.get("type") == "stats")
        assert stats["conn"]["optim"]["recommended_fps"] == 0
        # The next ticker state also contains the current recommendation, without a frame.
        assert recv_until(ws, lambda m: m.get("type") == "state")["recommended_fps"] == 0


def test_binary_jpeg_and_legacy_share_worker_and_wire_accounting():
    import json
    import struct

    jpeg = base64.b64decode(blank_jpeg().split(",")[-1])
    binary = struct.pack("<BdB", 4, 123.25, 0) + jpeg
    legacy = json.dumps({"type": "frame", "speech": False, "t": 124, "data": blank_jpeg()})
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "start"})
        recv_until(ws, lambda m: m.get("type") == "state" and m.get("running"))
        # All malformed videos are ignored, never routed through the audio tag.
        for invalid in [
            b"\x04",
            struct.pack("<BdB", 4, float("nan"), 0) + jpeg,
            struct.pack("<BdB", 4, 1, 2) + jpeg,
            struct.pack("<BdB", 4, 1, 0) + b"bad jpeg",
            struct.pack("<BdB", 4, 1, 0) + b"\xff\xd8bad\xff\xd9",
        ]:
            ws.send_bytes(invalid)
        ws.send_bytes(binary)
        state = recv_until(ws, lambda m: m.get("type") == "state" and "detections" in m)
        assert state["detections"] == []
        ws.send_text(legacy)
        state = recv_until(ws, lambda m: m.get("type") == "state" and "detections" in m)
        assert state["detections"] == []
        stats = recv_until(ws, lambda m: m.get("type") == "stats" and m["conn"]["frames"]["processed"] >= 2)
        # The malformed JPEG with valid envelope may be admitted but fails image decoding safely.
        assert stats["conn"]["frames"]["processed"] == 2
        assert stats["conn"]["frames"]["received"] == 3
        invalid_wire = len(struct.pack("<BdB", 4, 1, 0) + b"\xff\xd8bad\xff\xd9")
        assert stats["conn"]["frame_bytes_avg"] == round((len(binary) + len(legacy.encode()) + invalid_wire) / 3)
        ws.send_json({"type": "ping", "t": 9})
        assert recv_until(ws, lambda m: m.get("type") == "pong")["t"] == 9

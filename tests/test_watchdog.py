"""Watchdog and robustness: enabling pose mid-session while audio streams, concurrent connections, supervised task
restart, non-blocking outbox, /debug/events protection and log redaction."""

import asyncio
import base64
import logging
import struct
import threading

import cv2
import numpy as np
from fastapi.testclient import TestClient

from presencekit import diag, server


def jpeg():
    ok, buf = cv2.imencode(".jpg", np.full((480, 640, 3), 90, np.uint8))
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def pcm(i):
    t = np.arange(320) / 16000
    return b"\x01" + struct.pack("<d", float(i)) + (np.sin(2 * np.pi * 220 * t) * 8000).astype("<i2").tobytes()


def session(results, idx):
    img = jpeg()
    states_after = 0
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "config": {"pose_enabled": False}})
        ws.send_json({"type": "start"})
        for k in range(60):
            ws.send_bytes(pcm(k * 320))
            if k % 3 == 0:
                ws.send_json({"type": "frame", "speech": False, "data": img})
            if k == 20:
                ws.send_json({"type": "config", "config": {"pose_enabled": True}})
        # after pose was enabled the session must keep processing new frames (each one answered)
        for _ in range(3):
            ws.send_json({"type": "frame", "speech": False, "data": img})
            for _ in range(400):
                m = ws.receive()
                if m.get("text") and '"detections"' in m["text"]:
                    states_after += 1
                    break
        ws.send_json({"type": "stop"})
    results[idx] = states_after


def test_pose_enabled_mid_session_two_concurrent_connections():
    server.pipeline()
    res = [0, 0]
    ts = [threading.Thread(target=session, args=(res, i)) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(90)
    assert not any(t.is_alive() for t in ts), "a session hung"
    assert res[0] >= 1 and res[1] >= 1


def test_supervised_task_crash_is_logged_and_restarted():
    runs, crashes = [], []

    async def task():
        runs.append(1)
        if len(runs) < 3:
            raise RuntimeError("boom")

    asyncio.run(diag.supervise("t", task, "c-test", on_crash=lambda n, e: crashes.append(e)))
    assert len(runs) == 3 and len(crashes) == 2 and "boom" in crashes[0]
    ev = [e for e in diag.recent(50, "c-test") if e["event"] == "task_crash"]
    assert ev and ev[-1]["level"] == "error"


def test_outbox_post_never_blocks_and_drops_audio():
    class Slow:
        async def send_text(self, *_):
            await asyncio.sleep(10)

        async def send_bytes(self, *_):
            await asyncio.sleep(10)

    async def go():
        ob = server.Outbox(Slow())
        task = asyncio.create_task(ob.run())
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for _ in range(server.SEND_AUDIO_MAX + 30):
            ob.post(b"\x02" + b"\0" * 648, audio=True)
        assert loop.time() - t0 < 0.1
        task.cancel()
        return ob.dropped_audio

    assert asyncio.run(go()) > 0


def test_debug_events_token_protected_and_secret_free(monkeypatch):
    monkeypatch.setattr(server, "TOKEN", "tok123")
    diag.event("config_change", conn="c1", asr_key="sk-SECRET", other=1)
    c = TestClient(server.app)
    assert c.get("/debug/events").status_code in (401, 403)
    r = c.get("/debug/events?token=tok123")
    assert r.status_code == 200
    assert "sk-SECRET" not in r.text and "tok123" not in r.text


def test_log_redaction_masks_token_and_key():
    rec = logging.LogRecord("x", 20, "", 0, "GET /ws?token=abc&key=def", None, None)
    diag.Redact().filter(rec)
    msg = rec.getMessage()
    assert "abc" not in msg and "def" not in msg

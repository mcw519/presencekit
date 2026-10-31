"""Cross-session re-ID: mock Jev server; decisions, caps, TTL/wipe, privacy."""

import asyncio
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

from presencekit import reid
from presencekit.tracker import Config

SECRET = "sk-test-SECRET-123456"


class Mock:
    def __init__(self):
        self.scores = []  # queue of scores (float) or ("http", code)
        self.reqs = []
        self.srv = HTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1/systemone"

    def _handler(mock):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.reqs.append({"headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                s = mock.scores.pop(0) if mock.scores else 0.0
                if isinstance(s, tuple):
                    self.send_response(s[1])
                    self.end_headers()
                    return
                if s == "slow":
                    import time

                    time.sleep(1.0)
                    s = 0.99
                out = json.dumps(
                    {
                        "model": "jev-1.13.0",
                        "answers": {"same": {"type": "noul", "noul": s}},
                        "usage": {"input_tokens": 950},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)

        return H


@pytest.fixture
def mock():
    m = Mock()
    reid.set_key(SECRET)
    yield m
    m.srv.shutdown()
    reid.set_key("")


def face(x=0.4, w=0.2, part=True, **kw):
    return {
        "id": 1,
        "bbox": [x, 0.3, x + w, 0.6],
        "visible": True,
        "participant": part,
        "conf": 0.9,
        "has_face": True,
        **kw,
    }


def snap(session, state="ACTIVE", tracks=None):
    return {"session": session, "state": state, "tracks": tracks if tracks is not None else [face()]}


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def make(mock_url, **cfg):
    c = Config()
    c.update({"reid_enabled": True, "reid_consent": True, "reid_cooldown": 0, **cfg})
    out, ev, clk = [], [], Clock()
    r = reid.CrossSessionReid(
        c, reid.JevClient(url=mock_url), emit=out.append, event=lambda n, *a, **k: ev.append((n, k)), clock=clk
    )
    return r, out, ev, clk


async def drive(r, s, clk, dt=1.0):
    clk.t += dt
    for req in r.observe(s):
        kind, bbox, q = req
        r.on_crop(kind, b"JPEG-" + kind.encode(), quality=q)
    if r.task:
        await r.task
        r.task = None


async def end_session(r, clk, n):
    """Session n with a participant, then ENDED (stores its reference)."""
    await drive(r, snap(n), clk)
    await drive(r, snap(n), clk, 1.5)
    await drive(r, snap(n, "ENDED", []), clk)


def run(coro):
    return asyncio.run(coro)


def test_off_by_default_and_needs_consent(mock):
    c = Config()
    assert not c.reid_enabled and not c.reid_consent and c.reid_threshold == 0.9 and c.reid_ttl == 600
    r = reid.CrossSessionReid(c, reid.JevClient(url=mock.url))
    assert r.observe(snap(1)) == []
    c.update({"reid_enabled": True})
    assert r.observe(snap(2)) == []  # no consent -> nothing
    assert mock.reqs == []


def test_first_session_is_new_without_call(mock):
    async def go():
        r, out, ev, clk = make(mock.url)
        await drive(r, snap(1), clk)
        assert out[-1]["decision"] == "new" and out[-1]["reason"] == "no_reference"
        await drive(r, snap(1), clk)  # inside a session: never a call
        await drive(r, snap(1), clk)
        assert mock.reqs == []

    run(go())


def test_same_person_resumes_previous_conversation(mock):
    async def go():
        r, out, ev, clk = make(mock.url)
        await end_session(r, clk, 1)
        first = out[-1]["conversation_id"]
        mock.scores = [0.95]
        await drive(r, snap(2), clk)
        h = out[-1]
        assert h["decision"] == "resume" and h["conversation_id"] == first and h["resumed_from_session"] == 1
        assert len(mock.reqs) == 1
        req = mock.reqs[0]
        assert req["headers"]["typesafe-danger-zone-images"] == "true"
        assert req["headers"]["authorization"] == f"Bearer {SECRET}"
        b = req["body"]
        assert b["model"] == "jev-latest" and b["questions"]["same"]["type"] == "noul"
        assert set(b["state"]) == {"reference", "query"} and len(b["assets"]) == 2
        for _ in range(3):  # no further calls inside the session
            await drive(r, snap(2), clk)
        assert len(mock.reqs) == 1 and r.tokens == 950

    run(go())


def test_different_person_new_conversation(mock):
    async def go():
        r, out, ev, clk = make(mock.url)
        await end_session(r, clk, 1)
        first = out[-1]["conversation_id"]
        mock.scores = [0.1]
        await drive(r, snap(2), clk)
        assert out[-1]["decision"] == "new" and out[-1]["reason"] == "below_threshold"
        assert out[-1]["conversation_id"] != first

    run(go())


def test_uncertain_retries_once_then_new_or_resume(mock):
    async def go():
        r, out, ev, clk = make(mock.url)
        await end_session(r, clk, 1)
        mock.scores = [0.6, 0.7]
        await drive(r, snap(2), clk)
        assert r.phase == "await_retry" and out[-1]["session"] == 1
        await drive(r, snap(2), clk)
        assert out[-1]["decision"] == "new" and out[-1]["reason"] == "uncertain" and len(mock.reqs) == 2
        await end_session(r, clk, 3)
        mock.scores = [0.6, 0.0, 0.0, 0.93, 0.0, 0.0]
        await drive(r, snap(4), clk)
        await drive(r, snap(4), clk)
        assert out[-1]["decision"] == "resume"

    run(go())


def test_ttl_expiry_means_new_without_call(mock):
    async def go():
        r, out, ev, clk = make(mock.url, reid_ttl=600)
        await end_session(r, clk, 1)
        clk.t += 601
        await drive(r, snap(2), clk)
        assert out[-1]["reason"] == "no_reference" and mock.reqs == [] and r.store.refs == []
        assert any(n == "reid_store" and k.get("action") == "expire" for n, k in ev)

    run(go())


def test_ttl_capped_at_ten_minutes():
    c = Config()
    c.update({"reid_ttl": 3600})
    assert c.reid_ttl == 600


def test_errors_caps_and_breaker_fall_back_to_new(mock):
    async def go():
        r, out, ev, clk = make(mock.url, reid_cap_per_hour=2)
        await end_session(r, clk, 1)
        mock.scores = [("http", 500)]
        await drive(r, snap(2), clk)
        assert out[-1]["decision"] == "new" and out[-1]["reason"] == "error"
        await end_session(r, clk, 3)
        mock.scores = [0.1]
        await drive(r, snap(4), clk)
        await end_session(r, clk, 5)
        await drive(r, snap(6), clk)  # cap of 2 per hour reached -> new, no call
        assert out[-1]["reason"] == "cap" and len(mock.reqs) == 2

    run(go())


def test_timeout_falls_back(mock):
    async def go():
        r, out, ev, clk = make(mock.url, reid_timeout=0.3)
        await end_session(r, clk, 1)
        mock.scores = ["slow"]
        await drive(r, snap(2), clk)
        assert out[-1]["decision"] == "new" and out[-1]["reason"] == "error"
        assert any(n == "reid_error" and k["kind"] == "timeout" for n, k in ev)

    run(go())


def test_no_participant_hits_overall_cap(mock):
    async def go():
        r, out, ev, clk = make(mock.url, reid_decision_timeout=2)
        await end_session(r, clk, 1)
        await drive(r, snap(2, tracks=[]), clk)
        await drive(r, snap(2, tracks=[]), clk, 3)
        assert out[-1]["reason"] == "decision_timeout" and mock.reqs == []
        w = [k for n, k in ev if n == "reid_wait"][-1]
        assert w["why"] == "no_participant" and w["participant_seen"] is False

    run(go())


def test_face_wait_starts_at_first_participant(mock):
    async def go():
        r, out, ev, clk = make(mock.url, reid_decision_timeout=10, reid_face_wait=3)
        await end_session(r, clk, 1)
        small = [face(w=0.03)]  # participant visible but face too small to use
        await drive(r, snap(2, tracks=[]), clk)  # 1 s: nobody yet
        await drive(r, snap(2, tracks=[]), clk, 2)  # 3 s waited: would have timed out under the old rule
        assert r.phase == "await_query"
        await drive(r, snap(2, tracks=small), clk)  # participant appears -> 3 s face wait starts
        await drive(r, snap(2, tracks=small), clk, 2)
        assert r.phase == "await_query"
        await drive(r, snap(2, tracks=small), clk, 2)
        assert out[-1]["reason"] == "no_face_timeout"
        w = [k for n, k in ev if n == "reid_wait"][-1]
        assert w["why"] == "face_too_small" and w["participant_seen"] is True

    run(go())


def test_wall_deadline_continues_while_away(mock):
    async def go():
        r, out, ev, clk = make(mock.url, reid_decision_timeout=3)
        await end_session(r, clk, 1)
        await drive(r, snap(2, tracks=[]), clk)
        for _ in range(5):  # AWAY does not suspend the overall deadline
            await drive(r, snap(2, "AWAY", []), clk)
        assert r.phase == "idle" and out[-1]["reason"] == "decision_timeout"
        await drive(r, snap(2), clk)
        assert out[-1]["decision"] == "new" and len(mock.reqs) == 0

    run(go())


def test_wipe_paths(mock):
    async def go():
        r, out, ev, clk = make(mock.url)
        await end_session(r, clk, 1)
        assert len(r.store.refs) == 1
        r.wipe("ui")
        assert r.store.refs == []
        await end_session(r, clk, 2)
        assert reid.wipe_all() >= 1 and r.store.refs == []
        await end_session(r, clk, 3)
        r.cfg.update({"reid_enabled": False})
        r.observe(snap(3, "ENDED", []))
        assert r.store.refs == []
        r.cfg.update({"reid_enabled": True})
        await end_session(r, clk, 4)
        r.close()
        assert r.store.refs == []

    run(go())


def test_max_refs_newest_first(mock):
    async def go():
        r, out, ev, clk = make(mock.url, reid_max_refs=2)
        for n in (1, 2, 3):
            mock.scores = [0.0, 0.0]
            await end_session(r, clk, n)
        assert [x.session for x in r.store.refs] == [2, 3]

    run(go())


def test_key_never_logged_or_returned(mock, caplog):
    caplog.set_level(logging.DEBUG)

    async def go():
        r, out, ev, clk = make(mock.url)
        await end_session(r, clk, 1)
        mock.scores = [0.95]
        await drive(r, snap(2), clk)
        return out, ev, r.status()

    out, ev, st = run(go())
    blob = json.dumps([out, [list(e) for e in ev], st], default=str) + caplog.text
    assert SECRET not in blob and "JPEG-" not in blob and "data_base64" not in blob
    assert reid.key_status() == {"set": True, "source": "ui"}


def test_face_crop_in_memory(tmp_path):
    img = np.full((240, 320, 3), 128, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    before = set(os.listdir(tmp_path))
    out = reid.face_crop(buf.tobytes(), [0.3, 0.3, 0.6, 0.7])
    assert out and out[:2] == b"\xff\xd8"
    d = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert d.shape[:2] == (reid.CROP_MAX, reid.CROP_MAX)  # small faces upscaled (API minimum size)
    assert set(os.listdir(tmp_path)) == before


def test_module_never_writes_files():
    src = (Path(reid.__file__)).read_text()
    for bad in (" open(", "imwrite", "tofile", "NamedTemporaryFile", "write_bytes"):
        assert bad not in src, bad


@pytest.mark.skipif(not os.environ.get("REID_LIVE"), reason="live smoke test (set REID_LIVE=1 and TYPESAFE_API_KEY)")
def test_live_smoke():
    reid.set_key(os.environ.get("TYPESAFE_API_KEY", ""))  # earlier tests clear the in-memory key
    img = np.full((160, 160, 3), 200, np.uint8)
    cv2.circle(img, (80, 80), 50, (40, 90, 200), -1)  # synthetic non-person image
    jpg = reid.face_crop(cv2.imencode(".jpg", img)[1].tobytes(), [0.1, 0.1, 0.9, 0.9])
    score, meta = asyncio.run(reid.JevClient().compare(jpg, jpg, model="jev-latest", timeout=10))
    assert 0.0 <= score <= 1.0 and meta.get("model")


def test_endpoints_never_echo_key(caplog):
    from fastapi.testclient import TestClient

    from presencekit import server

    caplog.set_level(logging.DEBUG)
    c = TestClient(server.app)
    q = f"?token={server.TOKEN}" if server.TOKEN else ""
    r = c.post("/api/reid_key" + q, json={"key": SECRET})
    assert r.json() == {"set": True, "source": "ui"} and SECRET not in r.text
    assert SECRET not in c.get("/api/reid_status" + q).text
    assert c.post("/api/reid_reset" + q).json()["wiped"] >= 0
    assert SECRET not in caplog.text
    c.post("/api/reid_key" + q, json={"key": ""})
    assert reid.key_status()["set"] is False


# ---- v0.11.2: display-only comparison thumbnails (reid_show_crops)
def _jpeg(v):
    ok, b = cv2.imencode(".jpg", np.full((320, 320, 3), v, np.uint8))
    return b.tobytes()


async def drive_img(r, s, clk, dt=1.0, v=120):
    clk.t += dt
    for kind, _bbox, q in r.observe(s):
        r.on_crop(kind, _jpeg(v), quality=q)
    if r.task:
        await r.task
        r.task = None


def _views(out):
    return [o for o in out if o.get("type") == "reid_view"]


def test_view_thumbnails_sent_to_browser_only(mock, caplog):
    async def go(r, clk):
        for st in ("ACTIVE", "ACTIVE", "ENDED"):
            await drive_img(r, snap(1, st, [] if st == "ENDED" else None), clk, v=60)
        for _ in range(3):
            await drive_img(r, snap(2), clk, v=200)

    mock.scores = [0.95]
    r, out, ev, clk = make(mock.url)
    caplog.set_level(logging.DEBUG)
    run(go(r, clk))
    v = _views(out)[-1]
    assert v["decision"] == "resume" and v["score"] == 0.95 and v["threshold"] == 0.9 and v["ref_session"] == 1
    for k in ("probe", "ref"):
        img = cv2.imdecode(np.frombuffer(__import__("base64").b64decode(v[k]), np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[:2] == (reid.THUMB_PX, reid.THUMB_PX)
    assert abs(int(cv2.imdecode(np.frombuffer(__import__("base64").b64decode(v["ref"]), np.uint8), 0).mean()) - 60) < 8
    blob = json.dumps(ev, default=str) + caplog.text
    assert v["probe"][:40] not in blob and v["ref"][:40] not in blob  # never in log/diag events
    out.clear()
    r.wipe("ui")
    assert out == [{"type": "reid_view_clear"}]


def test_view_off_sends_no_thumbnails(mock):
    async def go(r, clk):
        for st in ("ACTIVE", "ACTIVE", "ENDED"):
            await drive_img(r, snap(1, st, [] if st == "ENDED" else None), clk)
        for _ in range(3):
            await drive_img(r, snap(2), clk)

    mock.scores = [0.95]
    r, out, _ev, clk = make(mock.url, reid_show_crops=False)
    run(go(r, clk))
    assert any(o["type"] == "handoff" for o in out) and not _views(out)


def test_view_new_conversation_without_reference(mock):
    async def go(r, clk):
        for _ in range(3):
            await drive_img(r, snap(1), clk)

    r, out, _ev, clk = make(mock.url)
    run(go(r, clk))
    v = _views(out)[-1]
    assert v["decision"] == "new" and v["reason"] == "no_reference" and v["ref"] is None and v["probe"] is None

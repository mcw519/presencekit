"""Cloud ASR (AssemblyAI v3) against a local mock WebSocket server; key handling never leaks the key."""

import asyncio
import json
import logging
import os
from urllib.parse import parse_qs, urlparse

import pytest

from presence_demo import asr
from presence_demo.tracker import Config

KEY = "test-key-0123456789abcdef"


@pytest.fixture(autouse=True)
def _key():
    old = dict(asr._key)
    asr.set_key(KEY)
    yield
    asr._key.update(old)


def test_params_per_model():
    c = Config()
    p = asr.build_params(c)
    assert p["speech_model"] == "universal-3-6-pro" and p["sample_rate"] == 16000 and p["encoding"] == "pcm_s16le"
    assert json.loads(p["language_codes"]) == ["zh", "en"] and "format_turns" not in p
    c.update(
        {
            "asr_speech_model": "universal-streaming-english",
            "asr_keyterms": "Acme, Widget",
            "asr_eot_confidence": 0.7,
        }
    )
    p = asr.build_params(c)
    assert p["format_turns"] == "true" and p["end_of_turn_confidence_threshold"] == "0.70" and "language_codes" not in p
    assert json.loads(p["keyterms_prompt"]) == ["Acme", "Widget"]
    assert c.update({"asr_language_codes": 5}) == ["invalid value for asr_language_codes"]


def test_stream_against_mock_server(caplog):
    caplog.set_level(logging.DEBUG)
    from websockets.asyncio.server import serve

    seen = {"chunks": [], "auth": None, "query": None, "terminate": False}

    async def handler(ws):
        seen["auth"] = ws.request.headers.get("Authorization")
        seen["query"] = parse_qs(urlparse(ws.request.path).query)
        await ws.send(json.dumps({"type": "Begin", "id": "x", "expires_at": 0, "configuration": {"model": "m"}}))
        async for m in ws:
            if isinstance(m, bytes):
                seen["chunks"].append(len(m))
                if len(seen["chunks"]) == 3:
                    w = [{"text": "你好", "start": 100, "end": 400, "confidence": 0.9, "word_is_final": True}]
                    base = {"type": "Turn", "turn_order": 0, "end_of_turn_confidence": 0.5, "words": w}
                    await ws.send(
                        json.dumps({**base, "end_of_turn": False, "turn_is_formatted": False, "transcript": "你"})
                    )
                    await ws.send(
                        json.dumps({**base, "end_of_turn": True, "turn_is_formatted": True, "transcript": "你好"})
                    )
            elif json.loads(m).get("type") == "Terminate":
                seen["terminate"] = True
                await ws.send(json.dumps({"type": "Termination", "audio_duration_seconds": 1}))
                return

    events = []

    async def main():
        async with serve(handler, "127.0.0.1", 0, logger=asr._ws_log) as srv:  # mock server must not log headers either
            port = srv.sockets[0].getsockname()[1]
            s = asr.AssemblyAIStream(Config(), events.append, url=f"ws://127.0.0.1:{port}/v3/ws")
            s.start()
            for _ in range(25):  # 0.5 s of 20 ms blocks
                s.feed(bytes(640))
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.2)
            await s.close()

    asyncio.run(main())
    assert seen["auth"] == KEY and seen["query"]["speech_model"] == ["universal-3-6-pro"]
    assert set(seen["chunks"]) == {3200}  # 100 ms per message
    assert seen["terminate"]
    kinds = [e["kind"] for e in events]
    assert "partial" in kinds and "final" in kinds and kinds[-1] == "status"
    fin = next(e for e in events if e["kind"] == "final")
    assert fin["text"] == "你好" and fin["start_ms"] == 100 and fin["t0"] is not None
    assert KEY not in caplog.text


def test_key_endpoints_never_echo_key(caplog):
    from fastapi.testclient import TestClient

    from presence_demo import server

    caplog.set_level(logging.DEBUG)
    c = TestClient(server.app)
    r = c.post("/api/asr_key", json={"key": "secret-xyz-987"})
    assert r.json() == {"set": True, "source": "ui"} and "secret-xyz-987" not in r.text
    assert "secret-xyz-987" not in c.get("/api/asr_status").text
    assert "secret-xyz-987" not in c.get("/api/config").text
    assert c.post("/api/asr_key", json={"key": ""}).json()["set"] is False
    assert "secret-xyz-987" not in caplog.text


def test_unreachable_server_reports_error_without_key(caplog):
    caplog.set_level(logging.DEBUG)
    events = []

    async def main():
        s = asr.AssemblyAIStream(Config(), events.append, url="ws://127.0.0.1:9/v3/ws", max_retries=0)
        s.start()
        await asyncio.sleep(0.5)
        await s.close()

    asyncio.run(main())
    assert events and events[0]["kind"] == "error"
    assert KEY not in caplog.text and KEY not in json.dumps(events)


def test_close_after_giving_up_with_full_queue_does_not_hang():
    async def main():
        s = asr.AssemblyAIStream(Config(), lambda ev: None, url="ws://127.0.0.1:9/v3/ws", max_retries=0)
        s.start()
        await asyncio.wait_for(s._task, 5.0)  # _run gave up: nobody drains the queue any more
        for _ in range(100):  # the audio worker keeps feeding until the session gate closes the stream
            s.feed(bytes(3200))
        assert s._q.full()
        await asyncio.wait_for(s.close(), 1.0)  # used to block forever on the full queue

    asyncio.run(main())


@pytest.mark.skipif(not os.environ.get("ASSEMBLYAI_API_KEY"), reason="no ASSEMBLYAI_API_KEY (live test)")
def test_live_connects():
    asr.set_key(os.environ["ASSEMBLYAI_API_KEY"])
    events = []

    async def main():
        s = asr.AssemblyAIStream(Config(), events.append)
        s.start()
        for _ in range(20):
            s.feed(bytes(3200))
            await asyncio.sleep(0.1)
        await s.close()

    asyncio.run(main())
    assert any(e.get("status") == "begin" for e in events)

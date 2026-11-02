"""Browser adapter contracts without launching Chrome or accessing devices."""

import asyncio
import json
from pathlib import Path

import pytest

from presencekit import acceptance_browser as browser


def test_numeric_results_drop_payloads_and_secrets():
    assert browser._numbers(
        {
            "session": 2,
            "errors": 0,
            "videoSent": float("nan"),
            "pcm": b"sensitive",
            "page_token": "secret",
            "url": "secret",
        }
    ) == {"session": 2, "errors": 0}


@pytest.mark.parametrize("seconds", [0, 2, 3601, float("nan"), float("inf"), None, "3"])
def test_duration_validation(seconds):
    with pytest.raises(ValueError):
        browser.run_browser(seconds=seconds)


def test_missing_chrome_is_not_run(monkeypatch, tmp_path):
    monkeypatch.setattr(browser, "CHROME", tmp_path / "missing")
    result = browser.run_browser()
    assert result[0]["status"] == "not_run"
    assert result[0]["suite"] == "browser"


def test_missing_offline_models_is_not_run(monkeypatch):
    monkeypatch.setattr(browser, "CHROME", Path(__file__))
    monkeypatch.setattr(browser.pipeline, "_preflight", lambda scene: "Local models missing")
    assert browser.run_browser()[0]["status"] == "not_run"


def test_execution_error_is_failed_without_exception_payload(monkeypatch):
    monkeypatch.setattr(browser, "CHROME", Path(__file__))
    monkeypatch.setattr(browser.pipeline, "_preflight", lambda scene: None)

    async def fail(seconds):
        raise RuntimeError("private page token / media / payload")

    monkeypatch.setattr(browser, "_isolated", fail)
    result = browser.run_browser(seconds=3)
    assert result[0]["status"] == "failed"
    assert result[0]["actual"] == {"error_type": "RuntimeError"}
    assert "private" not in json.dumps(result)


def test_common_api_returns_cases(monkeypatch):
    monkeypatch.setattr(browser, "CHROME", Path(__file__))
    monkeypatch.setattr(browser.pipeline, "_preflight", lambda scene: None)

    async def successful(seconds):
        assert seconds == 3
        return [browser._case("test", "passed", {"errors": 0}, {"errors": 0})]

    monkeypatch.setattr(browser, "_isolated", successful)
    case = browser.run_browser(seeds=20, seconds=3)[0]
    assert case["id"] == "browser.test"
    assert case["details"]["method"]["profile"].startswith("rules/vision")


class Socket:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.queue.get()

    async def send(self, data):
        message = json.loads(data)
        self.sent.append(message)
        result = {"result": {"value": {"session": 1, "page_token": "secret"}}}
        await self.queue.put(json.dumps({"id": message["id"], "result": result}))

    async def close(self):
        pass


def test_cdp_observable_snapshot_is_allowlisted():
    async def check():
        socket = Socket()
        cdp = browser.CDP(socket)
        try:
            assert await cdp.snapshot() == {"session": 1}
            assert (await cdp.wait(lambda state: state["session"] == 1))["session"] == 1
            with pytest.raises(TimeoutError):
                await cdp.wait(lambda state: False, timeout=0)
        finally:
            await cdp.close()

    asyncio.run(check())


def test_remote_browser_requests_are_blocked():
    async def check():
        socket = Socket()
        cdp = browser.CDP(socket)
        try:
            for ident, url in enumerate(("https://external.example/test", "http://127.0.0.1:1234/static/app.js")):
                await socket.queue.put(
                    json.dumps(
                        {"method": "Fetch.requestPaused", "params": {"requestId": str(ident), "request": {"url": url}}}
                    )
                )
            for _ in range(20):
                if len(socket.sent) == 2:
                    break
                await asyncio.sleep(0.01)
            assert [message["method"] for message in socket.sent] == ["Fetch.failRequest", "Fetch.continueRequest"]
        finally:
            await cdp.close()

    asyncio.run(check())


def test_fixture_never_calls_native_devices_or_persists_media():
    source = (browser.ROOT / "scripts/acceptance_browser_fixture.js").read_text()
    assert "Object.defineProperty(navigator.mediaDevices, 'getUserMedia'" in source
    assert "canvas.captureStream" in source
    assert "sourceContextsLive" in source
    assert "localStorage" not in source
    assert "MediaRecorder" not in source
    assert "download" not in source


def test_media_gate_requires_fresh_progress_and_session_continuity():
    before = dict(errors=0, session=1, ended=0, framesProcessed=3, pcmSent=100, audioReceived=90)
    after = dict(before, framesProcessed=5, pcmSent=130, audioReceived=120)
    assert browser._media_progress(before, after)
    for key in ("framesProcessed", "pcmSent", "audioReceived"):
        assert not browser._media_progress(before, dict(after, **{key: before[key]}))
    assert not browser._media_progress(before, dict(after, session=2))
    assert not browser._media_progress(before, dict(after, ended=1))
    assert not browser._media_progress(before, dict(after, errors=1))


def test_frozen_gate_rejects_elapsed_observation_time_even_without_ended():
    before = dict(session=1, ended=0, errors=0, inState=10.0)
    assert browser._frozen_observation(before, dict(before, inState=10.1))
    assert not browser._frozen_observation(before, dict(before, inState=32.0))
    assert not browser._frozen_observation(before, dict(before, session=2))


def test_reconnect_requires_current_socket_processing_not_old_totals():
    before = dict(connections=1, session=1)
    after = dict(
        connections=2,
        session=2,
        ended=0,
        sameCapability=1,
        currentFramesProcessed=1,
        currentPcmSent=20,
        currentVideoSent=2,
        currentAudioReceived=10,
    )
    assert browser._reconnected(before, after)
    for key in ("currentFramesProcessed", "currentPcmSent", "currentVideoSent", "currentAudioReceived"):
        assert not browser._reconnected(before, dict(after, **{key: 0}, framesProcessed=100, audioReceived=1000))
    assert not browser._reconnected(before, dict(after, sameCapability=0))

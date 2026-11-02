"""Isolated headless Chrome using synthetic media and the production web app.

No browser profile, physical device, remote service or media recording is used.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from . import acceptance_pipeline as pipeline

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
ROOT = pipeline.ROOT
METHOD = {
    "transport": "production app.js + AudioWorklet + binary JPEG/PCM over isolated loopback WebSocket",
    "media": "canvas with bundled test face and fixed-seed synthetic noise; native getUserMedia never called",
    "profile": "rules/vision, full face ROI, leave_distance 10m, 2fps, pose off, ASR none, re-ID disabled",
    "isolation": "temporary headless Chrome profile; local models only; remote browser requests blocked",
    "limitations": "No physical Chrome devices, Jev, TV semantics, audio quality or RTF performance gate",
}
NUMERIC_KEYS = frozenset(
    (
        "connections",
        "sameCapability",
        "videoSent",
        "pcmSent",
        "framesProcessed",
        "audioReceived",
        "errors",
        "interrupted",
        "session",
        "ended",
        "requests",
        "running",
        "streamLive",
        "audioLive",
        "frameTimerLive",
        "tracksLive",
        "sourceContextsLive",
        "productionContextLive",
        "inState",
        "currentFramesProcessed",
        "currentPcmSent",
        "currentVideoSent",
        "currentAudioReceived",
    )
)


def _case(name, status, expected, actual=None, **details):
    return {
        "id": "browser." + name,
        "suite": "browser",
        "status": status,
        "expected": expected,
        "actual": actual,
        "details": {"method": METHOD, **details},
    }


def _numbers(value):
    """Drop everything except explicitly permitted finite numeric counters."""
    return {
        key: number
        for key, number in value.items()
        if key in NUMERIC_KEYS and isinstance(number, (int, float)) and math.isfinite(number)
    }


def _media_progress(before, after):
    return (
        after["errors"] == 0
        and after["session"] == before["session"]
        and not after["ended"]
        and all(after[key] > before[key] for key in ("framesProcessed", "pcmSent", "audioReceived"))
    )


def _frozen_observation(before, after):
    return (
        after["session"] == before["session"]
        and not after["ended"]
        and after["errors"] == 0
        and abs(after["inState"] - before["inState"]) <= 0.200001
    )


def _reconnected(before, after):
    return (
        after["connections"] > before["connections"]
        and after["session"] > before["session"]
        and not after["ended"]
        and after["sameCapability"] == 1
        and all(
            after[key] > 0
            for key in ("currentFramesProcessed", "currentPcmSent", "currentVideoSent", "currentAudioReceived")
        )
    )


def _json_url(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=2) as response:
        return json.load(response)


class CDP:
    def __init__(self, ws):
        self.ws, self.serial, self.pending = ws, 0, {}
        self.reader = asyncio.create_task(self._read())
        self.actions = set()

    async def _read(self):
        try:
            async for raw in self.ws:
                message = json.loads(raw)
                if "id" in message:
                    future = self.pending.pop(message["id"], None)
                    if future and not future.done():
                        if "error" in message:
                            future.set_exception(RuntimeError("CDP command rejected"))
                        else:
                            future.set_result(message.get("result", {}))
                elif message.get("method") == "Fetch.requestPaused":
                    item = message["params"]
                    parsed = urlparse(item["request"]["url"])
                    local = parsed.scheme in ("data", "blob", "about") or (
                        parsed.scheme in ("http", "ws") and parsed.hostname == "127.0.0.1"
                    )
                    command = "Fetch.continueRequest" if local else "Fetch.failRequest"
                    params = {"requestId": item["requestId"]}
                    if not local:
                        params["errorReason"] = "BlockedByClient"
                    task = asyncio.create_task(self.call(command, params))
                    self.actions.add(task)
                    task.add_done_callback(self.actions.discard)
        except Exception:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("CDP connection closed"))

    async def call(self, method, params=None):
        self.serial += 1
        ident = self.serial
        future = asyncio.get_running_loop().create_future()
        self.pending[ident] = future
        try:
            await self.ws.send(json.dumps({"id": ident, "method": method, "params": params or {}}))
            return await asyncio.wait_for(future, 12)
        finally:
            self.pending.pop(ident, None)

    async def evaluate(self, expression):
        result = await self.call(
            "Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True}
        )
        if "exceptionDetails" in result:
            raise RuntimeError("Browser fixture evaluation failed")
        return result.get("result", {}).get("value")

    async def snapshot(self):
        return _numbers(await self.evaluate("acceptanceFixture.snapshot()"))

    async def wait(self, predicate, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = await self.snapshot()
            if predicate(value):
                return value
            await asyncio.sleep(0.15)
        raise TimeoutError("Browser observable contract timed out")

    async def close(self):
        self.reader.cancel()
        for task in tuple(self.actions):
            task.cancel()
        await asyncio.gather(self.reader, *self.actions, return_exceptions=True)
        await self.ws.close()


def _terminate(process):
    if process is None:
        return
    # The process group also owns Chrome renderers and utility processes.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


async def _browser_exercise(cdp, port, seconds):
    cases = []
    await cdp.evaluate("acceptanceFixture.configure(); acceptanceFixture.begin()")
    before = await cdp.wait(
        lambda s: (
            s["session"] > 0
            and not s["ended"]
            and s["framesProcessed"] > 0
            and s["audioReceived"] > 0
            and s["pcmSent"] > 0
            and s["videoSent"] > 0
        )
    )
    await asyncio.sleep(seconds)
    stable = await cdp.snapshot()
    cases.append(
        _case(
            "production_media",
            "passed" if _media_progress(before, stable) else "failed",
            "Production binary JPEG and AudioWorklet PCM processed without browser errors",
            stable,
            steady_stream_seconds=seconds,
        )
    )
    pause_start = time.monotonic()
    await cdp.evaluate("acceptanceFixture.pauseVideo()")
    paused = await cdp.wait(lambda s: s["interrupted"] == 1)
    interruption_ok = True
    frozen_start = time.monotonic()
    while time.monotonic() - frozen_start < 22:
        sample = await cdp.snapshot()
        interruption_ok = interruption_ok and _frozen_observation(paused, sample)
        await asyncio.sleep(0.2)
    after_pause = await cdp.snapshot()
    interruption_ok = interruption_ok and _frozen_observation(paused, after_pause)
    interruption_ok = interruption_ok and after_pause["audioReceived"] > paused["audioReceived"]
    interruption_ok = interruption_ok and after_pause["videoSent"] == paused["videoSent"]
    cases.append(
        _case(
            "sensor_interruption",
            "passed" if interruption_ok else "failed",
            "22-second video gap freezes presence without ENDED while audio continues",
            after_pause,
            paused_seconds=round(time.monotonic() - pause_start, 2),
            frozen_observation_seconds=round(time.monotonic() - frozen_start, 2),
            in_state_delta_seconds=round(after_pause["inState"] - paused["inState"], 3),
        )
    )
    await cdp.evaluate("acceptanceFixture.resumeVideo()")
    resumed = await cdp.wait(lambda s: s["interrupted"] == 0)
    cases.append(
        _case(
            "sensor_resume",
            "passed" if resumed["session"] == before["session"] and not resumed["ended"] else "failed",
            "Video resumes the same session",
            resumed,
        )
    )
    await cdp.evaluate("acceptanceFixture.reconnect()")
    reconnected = await cdp.wait(lambda s: _reconnected(before, s), timeout=15)
    cases.append(
        _case(
            "same_page_reconnect",
            "passed" if reconnected["sameCapability"] == 1 else "failed",
            "Automatic production reconnect retains page capability and opens a new session",
            reconnected,
        )
    )
    await cdp.evaluate("acceptanceFixture.finish()")
    stopped = await cdp.wait(lambda s: s["productionContextLive"] == 0)
    await asyncio.sleep(0.3)
    settled = await cdp.snapshot()
    clean = all(
        stopped[key] == 0
        for key in (
            "running",
            "streamLive",
            "audioLive",
            "frameTimerLive",
            "tracksLive",
            "sourceContextsLive",
            "productionContextLive",
        )
    )
    clean = clean and all(settled[key] == stopped[key] for key in ("videoSent", "pcmSent"))
    cases.append(
        _case(
            "stop_cleanup",
            "passed" if clean else "failed",
            "Capture tracks, app AudioContext, frame timer and synthetic source contexts closed",
            stopped,
        )
    )
    cases.append(
        _case(
            "browser_errors",
            "passed" if settled["errors"] == 0 else "failed",
            {"errors": 0},
            {"errors": settled["errors"]},
        )
    )
    before_pagehide = await asyncio.to_thread(_json_url, f"http://127.0.0.1:{port}/_acceptance/pages")
    await cdp.call("Page.navigate", {"url": "about:blank"})
    deadline = time.monotonic() + 3
    while True:
        page_count = await asyncio.to_thread(_json_url, f"http://127.0.0.1:{port}/_acceptance/pages")
        if page_count["count"] == 0 or time.monotonic() > deadline:
            break
        await asyncio.sleep(0.1)
    cases.append(
        _case(
            "pagehide_cleanup",
            "passed" if before_pagehide["count"] == 1 and page_count["count"] == 0 else "failed",
            {"before_retained_pages": 1, "retained_pages": 0},
            {"before_retained_pages": before_pagehide["count"], "retained_pages": page_count["count"]},
        )
    )
    health = await asyncio.to_thread(pipeline._health, port)
    cases.append(
        _case(
            "server_errors",
            "passed" if health["errors"] == 0 else "failed",
            {"errors": 0},
            {"errors": health["errors"]},
        )
    )
    return cases


async def _isolated(seconds, *, server_module=__name__, extra_fixture=None, exercise=_browser_exercise):
    from websockets.asyncio.client import connect

    server = chrome = cdp = None
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    cases = []
    try:
        with tempfile.TemporaryDirectory(prefix="presence-browser-") as profile:
            try:
                server = subprocess.Popen(
                    [sys.executable, "-m", server_module, "--serve-fd", str(listener.fileno())],
                    cwd=ROOT,
                    pass_fds=(listener.fileno(),),
                    env=pipeline._child_environment(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 60
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("Model server startup failed")
                    try:
                        health = await asyncio.to_thread(pipeline._health, port)
                        if health["pid"] != server.pid:
                            raise RuntimeError("Unexpected server identity")
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Model server startup timed out") from None
                        await asyncio.sleep(0.1)
                chrome = subprocess.Popen(
                    [
                        str(CHROME),
                        "--headless=new",
                        "--remote-debugging-port=0",
                        "--remote-debugging-address=127.0.0.1",
                        "--user-data-dir=" + profile,
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--disable-background-networking",
                        "--disable-component-update",
                        "--disable-sync",
                        "--disable-extensions",
                        "--disable-default-apps",
                        "--disable-renderer-backgrounding",
                        "--disable-background-timer-throttling",
                        "--autoplay-policy=no-user-gesture-required",
                        "--mute-audio",
                        "--proxy-server=http://127.0.0.1:9",
                        "--proxy-bypass-list=127.0.0.1",
                        "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1",
                        "about:blank",
                    ],
                    env=pipeline._child_environment(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                active = Path(profile) / "DevToolsActivePort"
                deadline = time.monotonic() + 12
                while not active.exists():
                    if chrome.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Isolated Chrome startup failed")
                    await asyncio.sleep(0.1)
                debug_port = int(active.read_text().splitlines()[0])
                targets = await asyncio.to_thread(_json_url, f"http://127.0.0.1:{debug_port}/json/list")
                target = next(item for item in targets if item["type"] == "page")
                cdp = CDP(await connect(target["webSocketDebuggerUrl"], max_size=8 << 20, proxy=None))
                await cdp.call("Page.enable")
                await cdp.call("Network.enable")
                await cdp.call("Network.setCacheDisabled", {"cacheDisabled": True})
                await cdp.call("Fetch.enable", {"patterns": [{"urlPattern": "*"}]})
                await cdp.call(
                    "Page.addScriptToEvaluateOnNewDocument",
                    {"source": (ROOT / "scripts/acceptance_browser_fixture.js").read_text()},
                )
                if extra_fixture:
                    await cdp.call("Page.addScriptToEvaluateOnNewDocument", {"source": extra_fixture.read_text()})
                await cdp.call("Page.navigate", {"url": f"http://127.0.0.1:{port}/"})
                deadline = time.monotonic() + 12
                while not await cdp.evaluate(
                    "typeof start==='function' && typeof cfg==='object' && ws?.readyState===1"
                ):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Production page startup timed out")
                    await asyncio.sleep(0.15)
                cases = await exercise(cdp, port, seconds)
            finally:
                try:
                    if cdp:
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(cdp.close(), timeout=3)
                finally:
                    try:
                        _terminate(chrome)
                    finally:
                        _terminate(server)
    finally:
        listener.close()
    return cases


def run_browser(seeds=20, seconds=10):
    """Fixed-seed media; seconds denotes the steady streaming phase, not all faults."""
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 3 <= seconds <= 3600:
        raise ValueError("seconds must be finite and in 3..3600")
    missing = (
        "Local Google Chrome is unavailable; no installation attempted."
        if not CHROME.is_file()
        else (pipeline._preflight("twofaces"))
    )
    if missing:
        return [_case("availability", "not_run", "Local Chrome and offline model prerequisites", reason=missing)]
    try:
        return asyncio.run(_isolated(seconds))
    except Exception as exc:
        # Exceptions can contain CDP payloads or URLs; retain only the type.
        return [
            _case(
                "execution",
                "failed",
                "All browser steps complete within their deadlines",
                {"error_type": type(exc).__name__},
            )
        ]


def _serve(fd):
    from fastapi.responses import FileResponse

    from .server import app
    from .web.continuity import PAGES

    @app.get("/_acceptance/face")
    def face():
        return FileResponse(ROOT / "tests/data/face.jpg", headers={"Cache-Control": "no-store"})

    @app.get("/_acceptance/pages")
    def pages():
        return {"count": len(PAGES.pages)}

    pipeline._serve(fd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-fd", type=int, required=True)
    _serve(parser.parse_args().serve_fd)

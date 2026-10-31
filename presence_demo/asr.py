"""Optional cloud ASR: AssemblyAI Universal-Streaming v3 over WebSocket (``asr_provider = assemblyai``).

PRIVACY: this is an EXTERNAL cloud service -- audio sent here leaves the device. Nothing is sent unless the
user selects ``assemblyai`` and an API key is set. The key lives in process memory only (env var
``ASSEMBLYAI_API_KEY`` or set from the UI); it is never logged, written to disk or echoed to the browser.
Transcripts are kept in memory only (forwarded to the page, never stored).

Docs (verified 2026-10-15): https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from urllib.parse import urlencode

log = logging.getLogger("presence.asr")
# websockets logs request headers (incl. Authorization = the key) at DEBUG: give it a logger that never does
_ws_log = logging.getLogger("presence.asr.ws")
_ws_log.setLevel(logging.WARNING)

URLS = {
    "global": "wss://streaming.assemblyai.com/v3/ws",
    "us": "wss://streaming.us.assemblyai.com/v3/ws",
    "eu": "wss://streaming.eu.assemblyai.com/v3/ws",
}
MODELS = ("universal-3-6-pro", "universal-3-5-pro", "universal-streaming-english", "universal-streaming-multilingual")
PRO = MODELS[:2]
URL_OVERRIDE: str | None = os.environ.get("ASSEMBLYAI_WS_URL") or None  # tests / proxies
SEND_MS = 100  # AssemblyAI accepts 50-1000 ms per message; we batch 20 ms blocks into 100 ms

_key: dict = {"value": os.environ.get("ASSEMBLYAI_API_KEY", "") or "", "source": "env"}
if not _key["value"]:
    _key["source"] = None


def set_key(value: str | None) -> dict:
    """Set (or clear with empty) the API key in memory. Returns the public status only."""
    v = (value or "").strip()
    _key["value"], _key["source"] = v, ("ui" if v else None)
    return key_status()


def key_status() -> dict:
    return {"set": bool(_key["value"]), "source": _key["source"]}


def _key_value() -> str:
    return _key["value"]


def build_params(cfg) -> dict:
    """Connection query parameters for the selected model (only the ones that model supports)."""
    m = cfg.asr_speech_model
    p: dict = {"sample_rate": 16000, "encoding": "pcm_s16le", "speech_model": m}
    p["min_turn_silence"] = int(cfg.asr_min_turn_silence)
    p["max_turn_silence"] = int(cfg.asr_max_turn_silence)
    if m in PRO:
        langs = [x.strip() for x in cfg.asr_language_codes.split(",") if x.strip()]
        if langs:
            p["language_codes"] = json.dumps(langs)
        p["language_detection"] = "true"
    else:  # Universal-Streaming English / Multilingual only
        p["format_turns"] = "true" if cfg.asr_format_turns else "false"
        p["end_of_turn_confidence_threshold"] = f"{cfg.asr_eot_confidence:.2f}"
        if m == "universal-streaming-multilingual":
            p["language_detection"] = "true"
    terms = [x.strip() for x in cfg.asr_keyterms.split(",") if x.strip()][:100]
    if terms:
        p["keyterms_prompt"] = json.dumps(terms)
    return p


class AssemblyAIStream:
    """One streaming session. ``feed`` PCM16 16 kHz bytes; messages go to ``on_event(dict)``.

    Reconnects with backoff (up to ``max_retries``) while open; ``close`` sends ``Terminate``.
    ``url`` can be overridden (tests use a local mock server).
    """

    def __init__(self, cfg, on_event, url: str | None = None, max_retries: int = 3):
        self.cfg = cfg
        self.on_event = on_event
        self.url = url or URLS.get(cfg.asr_region, URLS["global"])
        self.max_retries = max_retries
        self._buf = bytearray()
        self._q: asyncio.Queue = asyncio.Queue(maxsize=50)  # 5 s of 100 ms messages
        self._task: asyncio.Task | None = None
        self._closing = False
        self.t0: float | None = None  # monotonic time of the first audio byte of the current connection
        self.connected = False
        self.retries = 0
        self.sent_ms = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def feed(self, pcm: bytes) -> None:
        if self._closing:
            return
        self._buf += pcm
        n = 16000 * 2 * SEND_MS // 1000
        while len(self._buf) >= n:
            chunk, self._buf = bytes(self._buf[:n]), self._buf[n:]
            if self._q.full():
                self._q.get_nowait()  # drop oldest rather than grow (network stall)
            self._q.put_nowait(chunk)

    async def close(self) -> None:
        self._closing = True
        # never await the queue: once _run has given up (no key, retries exhausted) nobody drains it, and a full
        # queue would hang the audio worker and the connection close
        if self._q.full():
            self._q.get_nowait()
        self._q.put_nowait(None)
        if self._task:
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._task, 3.0)
            if not self._task.done():
                self._task.cancel()

    async def _run(self) -> None:
        from websockets.asyncio.client import connect

        while not self._closing:
            key = _key_value()
            if not key:
                self.on_event({"kind": "error", "error": "API key not set"})
                return
            url = self.url + "?" + urlencode(build_params(self.cfg))
            try:
                async with connect(
                    url, additional_headers={"Authorization": key}, open_timeout=10, logger=_ws_log
                ) as ws:
                    self.connected, self.t0 = True, None
                    self.on_event({"kind": "status", "status": "connected"})
                    rx = asyncio.create_task(self._recv(ws))
                    try:
                        while True:
                            chunk = await self._q.get()
                            if chunk is None:
                                await ws.send(json.dumps({"type": "Terminate"}))
                                with contextlib.suppress(asyncio.TimeoutError):
                                    await asyncio.wait_for(rx, 2.0)
                                return
                            if self.t0 is None:
                                self.t0 = time.monotonic() - SEND_MS / 1000
                            await ws.send(chunk)
                            self.sent_ms += SEND_MS
                            if rx.done():
                                rx.result()  # surface server close / errors
                                break
                    finally:
                        rx.cancel()
                        self.connected = False
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - report type only, never the URL/headers (key)
                self.connected = False
                self.retries += 1
                msg = type(e).__name__ + (f" {getattr(e, 'rcvd', None).code}" if getattr(e, "rcvd", None) else "")
                log.warning("assemblyai stream error: %s (retry %d)", msg, self.retries)
                self.on_event({"kind": "error", "error": msg, "retry": self.retries})
                if self.retries > self.max_retries:
                    return
                await asyncio.sleep(min(5.0, 0.5 * 2**self.retries))
            else:
                if not self._closing:  # server ended the session: reconnect
                    self.retries += 1
                    if self.retries > self.max_retries:
                        return

    async def _recv(self, ws) -> None:
        async for raw in ws:
            if not isinstance(raw, str):
                continue
            m = json.loads(raw)
            t = m.get("type")
            if t == "Begin":
                self.retries = 0
                self.on_event(
                    {"kind": "status", "status": "begin", "model": (m.get("configuration") or {}).get("model")}
                )
            elif t == "Turn":
                words = m.get("words") or []
                self.on_event(
                    {
                        "kind": "final" if m.get("end_of_turn") else "partial",
                        "text": m.get("transcript", ""),
                        "turn_order": m.get("turn_order"),
                        "formatted": m.get("turn_is_formatted"),
                        "start_ms": words[0]["start"] if words else None,
                        "end_ms": words[-1]["end"] if words else None,
                        "language": m.get("language_code"),
                        "t0": self.t0,
                    }
                )
            elif t == "Termination":
                self.on_event({"kind": "status", "status": "terminated", "audio_s": m.get("audio_duration_seconds")})
                return
            elif "error" in m:
                self.on_event({"kind": "error", "error": str(m.get("error"))[:200]})

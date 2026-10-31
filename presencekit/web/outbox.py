"""Per-connection outbound queue to the browser with back-pressure: audio and droppable JSON are bounded so a
slow client never blocks the processing workers."""

from __future__ import annotations

import asyncio
import collections
import json
import time

from fastapi import WebSocket, WebSocketDisconnect

from .. import diag

SEND_AUDIO_MAX = 50  # queued outbound audio messages (~1 s); more -> new audio is dropped (slow client)
SEND_AUDIO_STALE = 0.3  # an audio message waiting longer than this is skipped if newer audio is queued
SEND_JSON_MAX = 60  # queued droppable JSON messages; more -> drop (state/meta are superseded by the next one)


class Outbox:
    """Non-blocking outbound queue for one WebSocket: processing never awaits the network (a slow client used to
    block the vision/audio workers on ``send``). Audio and droppable JSON are dropped when the backlog grows."""

    def __init__(self, sock: WebSocket, conn: str | None = None):
        self.conn = conn
        self.closed = False
        self.sock, self.q = sock, collections.deque()
        self.ev = asyncio.Event()
        self.n_audio = self.n_json = 0
        self.sent = self.dropped_audio = self.dropped_json = 0
        self.send_ms_max = 0.0
        self.last_send = time.monotonic()

    def post(self, item, *, audio: bool = False, droppable: bool = False) -> None:
        if audio and self.n_audio >= SEND_AUDIO_MAX:
            self.dropped_audio += 1
            return
        if droppable and self.n_json >= SEND_JSON_MAX:
            self.dropped_json += 1
            return
        self.n_audio += audio
        self.n_json += droppable
        self.q.append((item, audio, droppable, time.monotonic()))
        self.ev.set()

    def backlog(self) -> int:
        return len(self.q)

    async def run(self) -> None:
        while True:
            if not self.q:
                self.ev.clear()
                await self.ev.wait()
                continue
            item, audio, droppable, t_post = self.q.popleft()
            self.n_audio -= audio
            self.n_json -= droppable
            if audio and self.n_audio and time.monotonic() - t_post > SEND_AUDIO_STALE:
                self.dropped_audio += 1  # catch up: playback latency matters more than completeness
                continue
            t0 = time.perf_counter()
            try:
                if isinstance(item, bytes):
                    await self.sock.send_bytes(item)
                else:
                    await self.sock.send_text(json.dumps(item, separators=(",", ":")))
            except (WebSocketDisconnect, RuntimeError) as e:  # client went away: a normal close, not a crash
                self.closed = True
                diag.event("sender_closed", conn=self.conn, reason=type(e).__name__)
                return
            ms = (time.perf_counter() - t0) * 1000
            self.send_ms_max = max(self.send_ms_max, ms)
            self.sent += 1
            self.last_send = time.monotonic()

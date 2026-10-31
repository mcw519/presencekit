"""One WebSocket connection = one session. This module is the *transport*: token check, page capability
handshake, envelope receive, and close. Everything else (tracker, vision/audio workers, watchdog, re-ID driver,
message handling) is :class:`presencekit.runtime.PresenceRuntime`, which streams state back through the
:class:`Outbox`.

The runtime's dependencies are read from this module's globals at connect time (``pipeline``, ``_vision_job``,
``AudioStream``, ``puresound_available``, ``PAGES``, ...), so replacing one of them here keeps working as in v0.12.
"""

from __future__ import annotations

import asyncio
import secrets
import time

from fastapi import WebSocket, WebSocketDisconnect

from .. import diag
from ..audio import AudioStream, puresound_available
from ..optimizer import FrameProcessor  # noqa: F401  (re-exported, v0.12 location)
from ..runtime.context import AppContext
from ..runtime.session import HEARTBEAT_S, STALL_S, WARN_FRAME_MS, PresenceRuntime  # noqa: F401  (re-exported)
from ..stats import AudioStats, ConnStats  # noqa: F401  (re-exported, v0.12 location)
from .continuity import PAGE_TOKEN, PAGES
from .outbox import Outbox
from .runtime import AUDIO_POOL, TOKEN, VISION_LOCK, _pose_cache, _vision_job, calibrate, pipeline, pose_detector


def app_context() -> AppContext:
    """Dependencies for a new connection, as currently bound in this module."""
    return AppContext(
        pages=PAGES,
        vision_lock=VISION_LOCK,
        audio_pool=AUDIO_POOL,
        pipeline=pipeline,
        pose_detector=pose_detector,
        pose_cache=_pose_cache,
        vision_job=_vision_job,
        calibrate=calibrate,
        audio_stream=AudioStream,
        puresound_available=puresound_available,
    )


async def ws(sock: WebSocket):
    # bytes: compare_digest raises TypeError on non-ASCII str (would be a 500 instead of a refusal)
    if TOKEN and not secrets.compare_digest(sock.query_params.get("token", "").encode(), TOKEN.encode()):
        await sock.close(code=1008)
        return
    protocols = [p.strip() for p in sock.headers.get("sec-websocket-protocol", "").split(",")]
    capabilities = [p[5:] for p in protocols if p.startswith("page-")]
    page_token = capabilities[0] if len(capabilities) == 1 else None
    if capabilities and (len(capabilities) != 1 or not PAGE_TOKEN.fullmatch(page_token)):
        await sock.close(code=1008)
        return
    if "presence-v1" in protocols:
        await sock.accept(subprotocol="presence-v1")
    else:
        await sock.accept()
    ctx = app_context()
    page, page_generation = None, None
    if page_token:
        try:
            page, page_generation = ctx.pages.acquire(page_token)
        except ValueError:
            await sock.close(code=1008)
            return
    cid = secrets.token_hex(3)
    peer = sock.client.host if sock.client else "?"
    diag.event("conn_open", conn=cid, peer=peer)
    t_open = time.monotonic()
    t = time.perf_counter()
    try:
        pipe = await asyncio.to_thread(ctx.pipeline)
    except BaseException:
        if page:
            ctx.pages.release(page, page_generation)
        raise
    if page and page.generation != page_generation:
        await sock.close(code=1000)
        return
    rt = PresenceRuntime(
        ctx,
        Outbox(sock, cid),
        cid=cid,
        pipe=pipe,
        page=page,
        page_generation=page_generation,
        t_open=t_open,
        ready_perf=t,
        close_transport=lambda: asyncio.create_task(sock.close(code=1000)),
    )
    rt.start()
    reason = "client"
    try:
        while True:
            raw = await sock.receive()
            if rt.stale():
                break
            if raw.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(raw.get("code", 1000))
            if raw.get("bytes") is not None:
                await rt.on_bytes(raw["bytes"])
            else:
                await rt.on_text(raw.get("text"))
    except WebSocketDisconnect as e:
        reason = f"disconnect code={e.code}"
    except Exception as e:  # noqa: BLE001
        reason = f"error {type(e).__name__}: {e}"
        diag.log.exception("receive loop failed conn=%s", cid)
    finally:
        await rt.close(reason)

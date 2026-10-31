"""PresenceRuntime runs without any web framework: a fake outbox and stub vision are enough (v0.13 layering)."""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from presencekit import diag
from presencekit.core.tracks import Detection
from presencekit.runtime import CLIENT_MESSAGE_TYPES, AppContext, FrameProcessor, PresenceRuntime
from presencekit.runtime import session as rsession
from presencekit.runtime.continuity import PageRegistry
from presencekit.runtime.protocol import VIDEO_HEADER, VIDEO_TAG


class FakeOutbox:
    def __init__(self):
        self.items, self.flags, self.dropped_audio, self.dropped_json = [], [], 0, 0
        self.send_ms_max, self.last_send = 0.0, time.monotonic()

    def post(self, item, *, audio=False, droppable=False):
        self.items.append(item)
        self.flags.append((audio, droppable))

    def backlog(self):
        return 0

    async def run(self):
        await asyncio.Event().wait()

    def of(self, kind):
        return [i for i in self.items if isinstance(i, dict) and i.get("type") == kind]


class StubPipe:
    name = "stub"

    def info(self):
        return {"pipeline": self.name}


def _ctx(calls):
    def vision_job(proc, data, cfg, stable, conn):
        calls.append((data, conn))
        return [Detection(bbox=(0.4, 0.3, 0.6, 0.7), conf=0.9, face_bbox=(0.45, 0.32, 0.55, 0.45))], {"detect": 1.0}

    return AppContext(
        pages=PageRegistry(),
        vision_lock=threading.Lock(),
        audio_pool=ThreadPoolExecutor(max_workers=1),
        pipeline=StubPipe,
        pose_detector=lambda: None,
        pose_cache={},
        vision_job=vision_job,
        calibrate=lambda tr, meters: ["no person to calibrate"],
        audio_stream=lambda cfg: None,
        puresound_available=lambda: (False, "stub"),
    )


def test_runtime_handles_the_protocol_without_a_web_framework():
    async def scenario():
        calls, out = [], FakeOutbox()
        rt = PresenceRuntime(_ctx(calls), out, cid="abc123", pipe=StubPipe())
        rt.start()
        try:
            await rt.on_text(json.dumps({"type": "ping", "t": 7}))
            await rt.on_text(json.dumps({"type": "config", "config": {"min_conf": 0.5}}))
            await rt.on_text(json.dumps({"type": "start"}))
            jpeg = b"\xff\xd8 private \xff\xd9"
            await rt.on_bytes(VIDEO_HEADER.pack(VIDEO_TAG, 12.0, 0) + jpeg)
            for _ in range(100):
                if rt.mon["frames_done"]:
                    break
                await asyncio.sleep(0.01)
            await rt.on_bytes(b"\x01" + struct.pack("<d", 1.0) + bytes(640))  # ignored: energy_vad pipeline
            await rt.on_text(json.dumps({"type": "stop"}))
        finally:
            await rt.close("client")
        return rt, out, calls

    rt, out, calls = asyncio.run(scenario())
    assert out.of("pong") == [{"type": "pong", "t": 7}]
    acks = [s for s in out.of("state") if s.get("config_ack")]
    assert acks and acks[0]["config"]["min_conf"] == 0.5
    assert calls == [(b"\xff\xd8 private \xff\xd9", "abc123")]
    assert rt.mon["frames_done"] == 1 and rt.aud["recv"] == 0
    assert any(s.get("detections") for s in out.of("state"))
    assert out.of("state")[-1]["running"] is False and out.of("state")[-1]["detections"] == []
    assert all(t.done() for t in rt.tasks.values())


def test_client_message_types_cover_every_handled_kind():
    import inspect

    src = inspect.getsource(PresenceRuntime.handle_message) + inspect.getsource(PresenceRuntime.on_text)
    for kind in CLIENT_MESSAGE_TYPES:
        assert f'"{kind}"' in src, kind


def test_v012_module_paths_reexport_the_same_objects():
    from presencekit import optimizer, runtime
    from presencekit.runtime import alignment, continuity, protocol, resources
    from presencekit.web import alignment as w_alignment
    from presencekit.web import continuity as w_continuity
    from presencekit.web import protocol as w_protocol
    from presencekit.web import runtime as w_runtime

    assert runtime.FrameProcessor is optimizer.FrameProcessor is FrameProcessor
    assert w_protocol.binary_frame is protocol.binary_frame and w_protocol.VIDEO_TAG == protocol.VIDEO_TAG
    assert w_alignment.decision_timing is alignment.decision_timing
    assert w_continuity.PAGES is continuity.PAGES and w_continuity.PageRegistry is continuity.PageRegistry
    for name in ("AUDIO_POOL", "VISION_LOCK", "VisionBusy", "_vision_job", "calibrate", "pipeline", "pose_detector"):
        assert getattr(w_runtime, name) is getattr(resources, name), name
    assert w_runtime._pose_cache is resources._pose_cache


def test_default_context_references_the_process_singletons():
    from presencekit.runtime import continuity, resources

    ctx = AppContext.default()
    assert ctx.pages is continuity.PAGES and ctx.vision_lock is resources.VISION_LOCK
    assert ctx.audio_pool is resources.AUDIO_POOL and ctx.pose_cache is resources._pose_cache
    assert set(ctx.reid_key_status()) == {"set", "source"}


class StubAudio:
    """PureSound stand-in: echoes the input as the clean output; recording tracks are the same samples."""

    def __init__(self):
        self.cfg, self.calibrated, self.last = None, None, b""

    def on_vision(self, snap, now):
        pass

    def process(self, pcm, snap, now, vis_age):
        self.last = pcm
        return pcm, False

    def recording_tracks(self):
        return self.last, self.last

    def meta(self):
        return {}


def test_recording_packets_are_bounded_like_audio():
    async def scenario():
        out = FakeOutbox()
        ctx = replace(_ctx([]), audio_stream=lambda cfg: StubAudio(), puresound_available=lambda: (True, None))
        rt = PresenceRuntime(ctx, out, cid="rec", pipe=StubPipe())
        rt.start()
        try:
            await rt.on_text(json.dumps({"type": "config", "config": {"audio_pipeline": "puresound"}}))
            await rt.on_text(json.dumps({"type": "start"}))
            await rt.on_text(json.dumps({"type": "record", "on": True}))
            for i in range(3):
                await rt.on_bytes(b"\x01" + struct.pack("<d", float(i)) + bytes(640))
            for _ in range(200):
                if rt.aud["processed"] >= 3:
                    break
                await asyncio.sleep(0.01)
        finally:
            await rt.close("client")
        return out

    out = asyncio.run(scenario())
    rec = [f for i, f in zip(out.items, out.flags, strict=True) if isinstance(i, bytes) and i[0] == 3]
    # 0x03 used to be posted with no flag at all, so the outbox never bounded it (slow client -> unbounded growth)
    assert len(rec) == 3 and all(audio for audio, _ in rec)


def test_undecodable_frames_do_not_trip_the_vision_watchdog(monkeypatch):
    monkeypatch.setattr(rsession, "STALL_S", 0.3)
    ctx = replace(_ctx([]), vision_job=lambda proc, data, cfg, stable, conn: None)  # every JPEG fails to decode

    async def scenario():
        rt = PresenceRuntime(ctx, FakeOutbox(), cid="badjpeg", pipe=StubPipe())
        rt.start()
        try:
            await rt.on_text(json.dumps({"type": "start"}))
            vision = rt.tasks["vision"]
            for _ in range(15):  # 1.5 s of corrupt frames: the watchdog checks every second
                await rt.on_bytes(VIDEO_HEADER.pack(VIDEO_TAG, 1.0, 0) + b"\xff\xd8 corrupt \xff\xd9")
                await asyncio.sleep(0.1)
            return rt, vision is rt.tasks["vision"]
        finally:
            await rt.close("client")

    rt, same_task = asyncio.run(scenario())
    assert same_task and rt.mon["frames_bad"] > 0 and rt.mon["frames_done"] == 0
    events = [e["event"] for e in diag.recent(200, "badjpeg")]
    assert "vision_stalled" not in events and "bad_frame" in events

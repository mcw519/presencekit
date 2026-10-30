"""PresenceRuntime runs without any web framework: a fake outbox and stub vision are enough (v0.13 layering)."""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from presence_demo.core.tracks import Detection
from presence_demo.runtime import CLIENT_MESSAGE_TYPES, AppContext, FrameProcessor, PresenceRuntime
from presence_demo.runtime.continuity import PageRegistry
from presence_demo.runtime.protocol import VIDEO_HEADER, VIDEO_TAG


class FakeOutbox:
    def __init__(self):
        self.items, self.dropped_audio, self.dropped_json = [], 0, 0
        self.send_ms_max, self.last_send = 0.0, time.monotonic()

    def post(self, item, *, audio=False, droppable=False):
        self.items.append(item)

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
    from presence_demo import optimizer, runtime
    from presence_demo.runtime import alignment, continuity, protocol, resources
    from presence_demo.web import alignment as w_alignment
    from presence_demo.web import continuity as w_continuity
    from presence_demo.web import protocol as w_protocol
    from presence_demo.web import runtime as w_runtime

    assert runtime.FrameProcessor is optimizer.FrameProcessor is FrameProcessor
    assert w_protocol.binary_frame is protocol.binary_frame and w_protocol.VIDEO_TAG == protocol.VIDEO_TAG
    assert w_alignment.decision_timing is alignment.decision_timing
    assert w_continuity.PAGES is continuity.PAGES and w_continuity.PageRegistry is continuity.PageRegistry
    for name in ("AUDIO_POOL", "VISION_LOCK", "VisionBusy", "_vision_job", "calibrate", "pipeline", "pose_detector"):
        assert getattr(w_runtime, name) is getattr(resources, name), name
    assert w_runtime._pose_cache is resources._pose_cache


def test_default_context_references_the_process_singletons():
    from presence_demo.runtime import continuity, resources

    ctx = AppContext.default()
    assert ctx.pages is continuity.PAGES and ctx.vision_lock is resources.VISION_LOCK
    assert ctx.audio_pool is resources.AUDIO_POOL and ctx.pose_cache is resources._pose_cache
    assert set(ctx.reid_key_status()) == {"set", "source"}

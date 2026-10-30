"""Per-connection presence runtime, independent of any web framework.

One :class:`PresenceRuntime` = one client connection: it owns the tracker, the vision worker (latest frame wins),
the audio worker (isolation + ASR gate), the ticker, the watchdog and the cross-session re-ID driver, and streams
state through an *outbox*. The transport (``presence_demo.web.session``) only does the handshake, decodes
nothing but the envelope, and calls :meth:`PresenceRuntime.on_bytes` / :meth:`PresenceRuntime.on_text`.

The outbox is duck-typed: ``post(item, *, audio=False, droppable=False)``, ``async run()``, ``backlog()``,
``last_send``, ``send_ms_max``, ``dropped_audio``, ``dropped_json`` (see ``presence_demo.web.outbox.Outbox``).

PRIVACY: audio/video payloads stay in memory and are never logged or written.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import struct
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from .. import asr as asrmod
from .. import diag
from .. import reid as reidmod
from ..audio import SR
from ..core.tracker import Config, PresenceTracker
from ..optimizer import FrameProcessor, recommend_fps
from ..stats import AudioStats, ConnStats, build_stats
from .alignment import VisionTiming, client_timestamp, decision_timing
from .context import AppContext
from .continuity import ObservationClock
from .protocol import AUDIO_IN_TAG, AUDIO_OUT_TAG, RECORDING_TAG, VIDEO_TAG, binary_frame, legacy_frame

WARN_FRAME_MS = float(os.environ.get("PRESENCE_WARN_FRAME_MS", "500"))
HEARTBEAT_S = float(os.environ.get("PRESENCE_HEARTBEAT_S", "10"))
STALL_S = float(os.environ.get("PRESENCE_STALL_S", "5"))


class PresenceRuntime:
    """All per-connection state and workers. Construct after the pipeline is loaded, then :meth:`start`,
    feed :meth:`on_bytes` / :meth:`on_text`, and always :meth:`close` (in a ``finally``)."""

    def __init__(
        self,
        ctx: AppContext,
        out: Any,
        *,
        cid: str,
        pipe: Any,
        page: Any = None,
        page_generation: int | None = None,
        t_open: float | None = None,
        ready_perf: float | None = None,
        close_transport: Callable[[], Any] | None = None,
    ):
        self.ctx, self.out, self.cid, self.pipe = ctx, out, cid, pipe
        self.page, self.page_generation = page, page_generation
        self.t_open = time.monotonic() if t_open is None else t_open
        self.observation_clock = ObservationClock()
        self.st: dict = {"tr": PresenceTracker(Config(), clock=self.observation_clock), "running": False}
        if page:
            self.st["tr"].session = page.session
        self.stats = ConnStats()
        self.audio_stats = AudioStats()
        self.proc = FrameProcessor(pipe, pose_loader=ctx.pose_detector)
        self.stats.counts = self.proc.counts
        self.fpsstate = {"sig": (), "changed_at": 0.0, "rec": 0.0}
        self.lock = asyncio.Lock()
        self.slot: dict = {"frame": None}
        self.wake = asyncio.Event()
        # audio (audio_pipeline=puresound): binary PCM16 chunks in, clean PCM16 + "audio" meta out; memory only
        self.aud: dict = {
            "stream": None,
            "q": asyncio.Queue(maxsize=25),  # <= 500 ms backlog; older chunks are dropped (latency over completeness)
            "last_rx": 0.0,
            "meta_t": 0.0,
            "err": None,
            "dropped": 0,
            "lat": None,
            "vision": None,  # immutable capture/receive/processing provenance of the last decision frame
            "calibrated": False,
            "asr": None,  # AssemblyAIStream while a session is active (asr_provider=assemblyai)
            "asr_sess": None,
            "sess": None,
            "sess_t0": 0.0,
            "rec": False,  # browser-side recording requested: also stream aligned vi + raw (server stores nothing)
            "recv": 0,
            "processed": 0,
            "rtf": None,
        }
        self.mon: dict = {
            "frame_rx_t": 0.0,  # last frame received (monotonic)
            "frame_done_t": time.monotonic(),
            "frames_done": 0,
            "audio_rx_t": 0.0,
            "audio_done_t": time.monotonic(),
            "loop_lag_ms": 0.0,
            "loop_lag_max": 0.0,
            "state": None,
            "session": None,
            "warn_t": {},
            "tracks": {},  # id -> {"visible": bool, "since": t}  (track lifecycle events)
            "fm": {
                "frames": 0,
                "detected": 0,
                "confs": [],
                "widths": [],
                "out_roi": 0,
                "coast": 0,
                "t0": time.monotonic(),
            },
            "wall0": time.time(),
            "mono0": time.monotonic(),
        }
        self.rd = reidmod.CrossSessionReid(
            self.st["tr"].cfg,
            emit=lambda d: out.post(d),
            event=lambda name, level="info", **kv: diag.event(name, level, cid, **kv),
            store=page.store if page else None,
        )
        if page:
            page.interrupt = self.rd.interrupt
            if close_transport is not None:
                page.disconnect = close_transport
        vision_ms = round((time.perf_counter() - ready_perf) * 1000, 1) if ready_perf is not None else None
        diag.event("conn_ready", conn=cid, vision_ms=vision_ms, pipeline=pipe.name)
        self.tasks: dict[str, asyncio.Task] = {}
        self.factories = {"ticker": self.ticker, "vision": self.worker, "audio": self.audio_worker, "sender": out.run}
        self._unsub: Callable[[], None] | None = None

    # ---------------------------------------------------------------- helpers
    @property
    def tracker(self) -> PresenceTracker:
        return self.st["tr"]

    @property
    def running(self) -> bool:
        return self.st["running"]

    def stale(self) -> bool:
        """True once a newer connection took over this page (this one must stop)."""
        return bool(self.page and self.page.generation != self.page_generation)

    def warn(self, name: str, every: float = 30.0, level: str = "warning", **kv) -> None:  # rate-limited warnings
        now = time.monotonic()
        if now - self.mon["warn_t"].get(name, -1e9) >= every:
            self.mon["warn_t"][name] = now
            diag.event(name, level, self.cid, **kv)

    def notify_ui(self, rec: dict) -> None:  # this connection's events (and global ones) -> UI 事件紀錄
        if rec.get("conn") in (self.cid, None):
            self.out.post({"type": "event", **rec}, droppable=rec.get("level") == "info")

    def on_crash(self, task: str, err: str) -> None:
        self.out.post({"type": "error", "where": task, "error": err})

    def clean_active(self) -> bool:
        return self.aud["stream"] is not None and time.monotonic() - self.aud["last_rx"] < 1.0

    def track_state(self, snap: dict) -> None:
        mon, cid = self.mon, self.cid
        s, sess = snap.get("state"), snap.get("session")
        tracks = snap.get("tracks", [])
        if s != mon["state"] or sess != mon["session"]:
            tr_ev = [e for e in snap.get("log") or [] if e.get("event") == "transition" and e.get("to") == s]
            diag.event(
                "state",
                conn=cid,
                frm=mon["state"],
                to=s,
                session=sess,
                reason=tr_ev[-1].get("reason") if tr_ev else None,
                participants=sum(1 for x in tracks if x.get("participant")),
                faces=sum(1 for x in tracks if x.get("visible") and x.get("has_face") is not False),
                speech=snap.get("speaking"),
                p_group=snap.get("p_group"),
            )
            if s == "ACTIVE" and mon["state"] == "ENDED":
                diag.event(
                    "session_open",
                    conn=cid,
                    session=sess,
                    reason=tr_ev[-1].get("reason") if tr_ev else None,
                    participants=sum(1 for x in tracks if x.get("participant")),
                    faces=sum(1 for x in tracks if x.get("visible") and x.get("has_face") is not False),
                    tracks=len(tracks),
                    speech=snap.get("speaking"),
                )
            mon["state"], mon["session"] = s, sess
        # track lifecycle: new / lost (missing) / recovered / dropped
        now_t, seen = time.monotonic(), set()
        for t in tracks:
            tid, vis = t.get("id"), bool(t.get("visible")) and t.get("visible_kind") != "predicted"
            seen.add(tid)
            prev = mon["tracks"].get(tid)
            if prev is None:
                mon["tracks"][tid] = {"visible": vis, "since": now_t}
                diag.event(
                    "track_new",
                    conn=cid,
                    track=tid,
                    participant=t.get("participant"),
                    has_face=t.get("has_face"),
                    session=sess,
                )
            elif prev["visible"] != vis:
                if vis:
                    diag.event(
                        "track_recovered",
                        conn=cid,
                        track=tid,
                        missing_s=round(now_t - prev["since"], 1),
                        participant=t.get("participant"),
                    )
                else:
                    diag.event(
                        "track_lost",
                        conn=cid,
                        track=tid,
                        kind=t.get("visible_kind") or "missing",
                        visible_s=round(now_t - prev["since"], 1),
                        participant=t.get("participant"),
                    )
                prev.update(visible=vis, since=now_t)
        for tid in [k for k in mon["tracks"] if k not in seen]:
            prev = mon["tracks"].pop(tid)
            diag.event(
                "track_dropped", conn=cid, track=tid, was_visible=prev["visible"], for_s=round(now_t - prev["since"], 1)
            )
        for e in snap.get("events") or []:
            name = e.get("event") if isinstance(e, dict) else str(e)
            if name in ("NEW_GROUP", "ENDED", "SESSION_START", "JOIN", "LEAVE"):
                diag.event("session_event", conn=cid, kind=name, session=sess)

    def send(self, snap, extra=None, droppable=True):
        page, st, fpsstate = self.page, self.st, self.fpsstate
        if page:
            if page.generation != self.page_generation:
                return
            page.session = max(page.session, int(snap.get("session") or 0))
        self.track_state(snap)
        now = time.monotonic()
        sig = tuple(sorted((t["id"], t["participant"], t["visible"]) for t in snap["tracks"]))
        if sig != fpsstate["sig"]:
            fpsstate["sig"], fpsstate["changed_at"] = sig, now
        fpsstate["rec"] = recommend_fps(snap, st["tr"].cfg, now - fpsstate["changed_at"] < 3.0)
        self.stats.recommended_fps = fpsstate["rec"]
        snap["recommended_fps"] = fpsstate["rec"]
        snap.update(
            {
                "type": "state",
                "running": st["running"],
                "pipeline": self.pipe.name,
                "sensor_status": "interrupted" if self.observation_clock.paused_at is not None else "available",
            }
        )
        if extra:
            snap.update(extra)
            if extra.get("config_ack"):
                snap["config"] = asdict(st["tr"].cfg)
        self.out.post(snap, droppable=droppable)

    # ---------------------------------------------------------------- workers
    async def ticker(self):  # keep timers moving; stats ~1/s while running; loop-lag probe
        st, mon, rd, out, pose_cache = self.st, self.mon, self.rd, self.out, self.ctx.pose_cache
        n = 0
        while True:
            t0 = time.monotonic()
            await asyncio.sleep(0.5)
            lag = (time.monotonic() - t0 - 0.5) * 1000
            mon["loop_lag_ms"] = lag
            mon["loop_lag_max"] = max(mon["loop_lag_max"], lag)
            if lag > 250:
                self.warn("loop_lag", 10, lag_ms=round(lag))
            n += 1
            async with self.lock:
                if self.stale():
                    return
                tr = st["tr"]
                if st["running"] and time.monotonic() - mon["frame_done_t"] > 3:
                    self.observation_clock.pause(at=mon["frame_done_t"] + 3)
                self.ctx.pages.prune()
                snap_t = tr.tick() if st["running"] else tr.snapshot()
                self.send(snap_t)
                if st["running"]:
                    rd.observe(snap_t)  # timeouts / TTL only; crops come from the frame worker
                if n % 2 == 0 and (rd.on() or rd.calls):
                    out.post({"type": "reid", **rd.status()}, droppable=True)
                if st["running"] and n % 2 == 0:
                    out.post(
                        build_stats(
                            self.stats,
                            {
                                **self.pipe.info(),
                                "pose_model": (
                                    pose_cache.get("det").name
                                    if pose_cache.get("det")
                                    else ("unavailable" if "det" in pose_cache else "not loaded")
                                ),
                            },
                        ),
                        droppable=True,
                    )

    async def worker(self):  # processes only the latest frame; older queued frames are dropped
        st, mon, aud, slot, rd = self.st, self.mon, self.aud, self.slot, self.rd
        while True:
            await self.wake.wait()
            self.wake.clear()
            msg = slot["frame"]
            slot["frame"] = None
            if msg is None or not st["running"]:
                continue
            t0 = time.perf_counter()
            data = msg["data"]
            stable = st["tr"].state.value not in ("MAYBE_AWAY", "AWAY")
            submitted = time.monotonic()

            def vision_job(data=data, cfg=st["tr"].cfg, stable=stable, submitted=submitted):
                started = time.monotonic()
                result = self.ctx.vision_job(self.proc, data, cfg, stable, self.cid)
                if result is not None:
                    result[1]["executor_wait"] = (started - submitted) * 1000
                return result

            r = await asyncio.to_thread(vision_job)
            if r is None:
                del data
                continue
            dets, tm = r
            tm["queue_wait"] = max(0.0, (t0 - msg["_received_perf"]) * 1000)
            reqs: list = []
            async with self.lock:
                if not st["running"]:
                    continue
                t1 = time.perf_counter()
                tr = st["tr"]
                if self.stale():
                    continue
                self.observation_clock.observed()
                tr.speech_source = "clean" if self.clean_active() else "browser"
                snap = tr.update(dets, speaking=None if self.clean_active() else msg.get("speech"))
                if not tm.get("skipped"):
                    fm = mon["fm"]
                    fm["frames"] += 1
                    faces = [d for d in dets if d.face_bbox is not None]
                    if faces:
                        fm["detected"] += 1
                        r_ = tr.cfg.face_roi
                        for d in faces:
                            fm["confs"].append(float(d.conf))
                            fm["widths"].append(float(d.face_bbox[2] - d.face_bbox[0]))
                            cx, cy = (d.face_bbox[0] + d.face_bbox[2]) / 2, (d.face_bbox[1] + d.face_bbox[3]) / 2
                            fm["out_roi"] += not (r_[0] <= cx <= r_[2] and r_[1] <= cy <= r_[3])
                    fm["coast"] += any(
                        t.get("participant") and t.get("visible_kind") == "predicted" for t in snap["tracks"]
                    )
                tm["tracker"] = (time.perf_counter() - t1) * 1000
                aud["vision"] = VisionTiming(
                    client_timestamp(msg.get("t")), msg["_received_mono"], submitted, time.monotonic()
                )
                tm["total"] = (time.perf_counter() - t0) * 1000
                now = time.monotonic()
                self.stats.on_processed(tm)
                mon["frame_done_t"] = now
                mon["frames_done"] += 1
                if tm["total"] > WARN_FRAME_MS:
                    self.warn(
                        "slow_frame",
                        10,
                        total_ms=round(tm["total"]),
                        detect_ms=round(tm.get("detect", 0)),
                        pose_ms=round(tm.get("pose", 0)),
                        embed_ms=round(tm.get("embed", 0)),
                    )
                self.send(
                    snap,
                    {
                        "detections": [{"bbox": d.bbox, "conf": d.conf} for d in dets],
                        "skipped": tm.get("skipped", False),
                    },
                )
                reqs = rd.observe(snap, now)
            # cross-session re-ID: crop faces in memory only when the driver asks (never stored to disk)
            for kind, bbox, q in reqs:
                crop = await asyncio.to_thread(reidmod.face_crop, data, bbox)
                rd.on_crop(kind, crop, quality=q)
            del data

    def asr_event(self, ev: dict, sess_t0: float, session: int):
        # session-relative turn times: stream start (monotonic) - session start + word offset
        t0 = ev.pop("t0", None)
        for k in ("start_ms", "end_ms"):
            if t0 is not None and ev.get(k) is not None:
                ev[k.replace("_ms", "_s")] = round(t0 - sess_t0 + ev[k] / 1000, 2)
        if ev.get("kind") in ("status", "error"):  # never the transcript text
            diag.event(
                "asr",
                "warning" if ev.get("kind") == "error" else "info",
                self.cid,
                status=ev.get("status"),
                error=ev.get("error"),
                session=session,
            )
        self.out.post({"type": "asr", "session": session, **ev})

    async def asr_close(self):
        aud = self.aud
        if aud["asr"] is not None:
            s_, aud["asr"] = aud["asr"], None
            await s_.close()

    async def asr_step(self, tr, raw: bytes, clean: bytes, speech: bool, now: float):
        """Cloud ASR gate: only during an active session, only if selected and a key is set."""
        aud, st = self.aud, self.st
        c = tr.cfg
        if tr.session != aud["sess"]:
            aud["sess"], aud["sess_t0"] = tr.session, now
        want = (
            c.asr_provider == "assemblyai"
            and asrmod.key_status()["set"]
            and st["running"]
            and tr.state.value != "ENDED"
        )
        if (
            c.asr_provider == "assemblyai"
            and not asrmod.key_status()["set"]
            and st["running"]
            and tr.state.value != "ENDED"
        ):
            self.warn("asr_skip", 60, reason="no_key")
        if aud["asr"] is not None and (not want or aud["asr_sess"] != tr.session):
            await self.asr_close()
        if not want:
            return
        if aud["asr"] is None:
            sess, t0 = tr.session, aud["sess_t0"]
            aud["asr"] = asrmod.AssemblyAIStream(c, lambda ev: self.asr_event(ev, t0, sess), url=asrmod.URL_OVERRIDE)
            aud["asr_sess"] = sess
            aud["asr"].start()
        src = clean if c.asr_source == "clean" else raw
        if c.asr_feed == "speech" and not speech:
            src = bytes(len(src))  # keep real-time pacing; no content leaves the device
        aud["asr"].feed(src)

    async def audio_worker(self):  # one chunk at a time, in order; isolation runs in a thread
        st, mon, aud, out, cid = self.st, self.mon, self.aud, self.out, self.cid
        loop = asyncio.get_running_loop()
        while True:
            batch = [await aud["q"].get()]
            while not aud["q"].empty() and len(batch) < 25:  # catch up after a stall: one thread hop for the backlog
                batch.append(aud["q"].get_nowait())
            aud["backlog_max"] = max(aud.get("backlog_max", 0), len(batch))
            if len(batch) > 10:
                self.warn("audio_backlog", 10, backlog_ms=len(batch) * 20)
            tr = st["tr"]
            if tr.cfg.audio_pipeline != "puresound":
                aud["stream"] = None
                continue
            if aud["stream"] is None:
                ok, why = self.ctx.puresound_available()
                if not ok:
                    if aud["err"] != why:
                        aud["err"] = why
                        diag.event("audio_unavailable", "warning", cid, reason=why)
                        out.post({"type": "audio", "available": False, "error": why})
                    continue
                t_init = time.perf_counter()
                aud["stream"] = await asyncio.to_thread(self.ctx.audio_stream, tr.cfg)
                diag.event("audio_init", conn=cid, ms=round((time.perf_counter() - t_init) * 1000, 1))
                aud["err"] = None
            a = aud["stream"]
            a.cfg = tr.cfg
            now = time.monotonic()
            snap = tr.snapshot() if st["running"] else None  # None -> acoustic_only (no vision)
            a.calibrated = aud["calibrated"]
            # No await between snapshot and provenance: later vision completions must not
            # relabel an older decision while this batch runs in the audio executor.
            vision = aud["vision"] if snap is not None else None
            stale_after = a.cfg.xcheck_stale

            rec = aud["rec"]

            def job(a=a, snap=snap, now=now, vision=vision, batch=batch, rec=rec, stale_after=stale_after):
                if snap is not None:
                    a.on_vision(snap, now)
                res = []
                for audio_client, pcm, received in batch:
                    started = time.monotonic()
                    vis_age, timing = decision_timing(vision, audio_client, started, stale_after)
                    o, sp = a.process(pcm, snap, started, vis_age)
                    done = time.monotonic()
                    res.append(
                        (o, sp, a.recording_tracks() if rec else None, done, done - started, started - received, timing)
                    )
                return res

            t0 = time.perf_counter()
            res = await loop.run_in_executor(self.ctx.audio_pool, job)
            proc_ms = (time.perf_counter() - t0) * 1000 / len(batch)
            aud["last_rx"] = time.monotonic()
            aud["processed"] += len(batch)
            mon["audio_done_t"] = time.monotonic()
            for (t_client, pcm, _), (o, speech, tracks, done, elapsed, wait, _timing) in zip(batch, res, strict=True):
                self.audio_stats.record(done, elapsed, len(pcm) / (2 * SR), wait)
                if o:
                    out.post(bytes((AUDIO_OUT_TAG,)) + struct.pack("<d", t_client) + o, audio=True)
                if o and tracks is not None:  # 0x03: t, n, out, vi, raw (same n samples each, aligned)
                    vi, rw = tracks
                    out.post(bytes((RECORDING_TAG,)) + struct.pack("<dI", t_client, len(o) // 2) + o + vi + rw)
                await self.asr_step(tr, pcm, o, speech, now)
            async with self.lock:
                tr = st["tr"]
                tr.speech_source = "clean"
                if st["running"] and self.observation_clock.paused_at is None and (speech != tr.speaking):
                    self.send(tr.set_speech(speech), droppable=False)
            if now - aud["meta_t"] >= 0.1:
                aud["meta_t"] = now
                meta = a.meta()
                aud["rtf"] = meta.get("rtf")
                aud["counts"] = meta.get("counts")
                out.post(
                    {
                        "type": "audio",
                        "available": True,
                        **meta,
                        **self.audio_stats.snapshot(),
                        "p_group": snap.get("p_group") if snap is not None else None,
                        "state": snap.get("state") if snap is not None else tr.state.value,
                        "t_server": time.time(),
                        "t_client": t_client,
                        "proc_ms": round(proc_ms, 2),
                        "t_vision_client": vision.client_ms if vision is not None else None,
                        "vision_timing": _timing,
                        "asr": {
                            "provider": tr.cfg.asr_provider,
                            "key_set": asrmod.key_status()["set"],
                            "connected": bool(aud["asr"] and aud["asr"].connected),
                            "sent_s": round(aud["asr"].sent_ms / 1000, 1) if aud["asr"] else 0,
                        },
                        "dropped": aud["dropped"],
                        "queue": aud["q"].qsize(),
                        "backlog_max": aud.get("backlog_max", 0),
                        "send_dropped": out.dropped_audio,
                    },
                    droppable=True,
                )

    def spawn(self, name: str) -> None:
        self.tasks[name] = asyncio.create_task(diag.supervise(name, self.factories[name], self.cid, self.on_crash))

    async def watchdog(self):  # heartbeat + stall detection (+ recovery) every HEARTBEAT_S / 1 s
        st, mon, aud, out, cid, rd, tasks = self.st, self.mon, self.aud, self.out, self.cid, self.rd, self.tasks
        vision_lock = self.ctx.vision_lock
        last_hb = time.monotonic()
        last_frames = last_audio = 0
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            run = st["running"]
            ps = aud["stream"] is not None and st["tr"].cfg.audio_pipeline == "puresound"
            # vision: frames arrive but none finished for STALL_S
            if run and mon["frame_rx_t"] > mon["frame_done_t"] + STALL_S:
                self.warn(
                    "vision_stalled",
                    30,
                    "error",
                    since_s=round(now - mon["frame_done_t"], 1),
                    lock_held=vision_lock.locked(),
                )
                diag.log.error("stack dump (vision_stalled conn=%s)\n%s", cid, diag.dump_stacks())
                out.post({"type": "error", "where": "vision", "error": "影像處理停滯，正在重新啟動"})
                tasks["vision"].cancel()
                self.proc.reset()
                self.spawn("vision")
                mon["frame_done_t"] = now
                self.wake.set()
            if run and ps and mon["audio_rx_t"] > mon["audio_done_t"] + STALL_S:
                self.warn(
                    "audio_stalled", 30, "error", since_s=round(now - mon["audio_done_t"], 1), queue=aud["q"].qsize()
                )
                diag.log.error("stack dump (audio_stalled conn=%s)\n%s", cid, diag.dump_stacks())
                out.post({"type": "error", "where": "audio", "error": "音訊處理停滯，正在重新啟動"})
                tasks["audio"].cancel()
                aud["stream"] = None
                self.spawn("audio")
                mon["audio_done_t"] = now
            if run and now - mon["frame_rx_t"] > 3 and now - self.t_open > 5:
                self.warn("no_frames", 30, since_s=round(now - mon["frame_rx_t"], 1) if mon["frame_rx_t"] else None)
            if (
                run
                and st["tr"].cfg.audio_pipeline == "puresound"
                and now - mon["audio_rx_t"] > 1
                and now - self.t_open > 5
            ):
                self.warn("no_audio", 30, since_s=round(now - mon["audio_rx_t"], 1) if mon["audio_rx_t"] else None)
            if out.backlog() > 50 or now - out.last_send > 5 and out.backlog():
                self.warn(
                    "send_backlog",
                    10,
                    backlog=out.backlog(),
                    dropped_audio=out.dropped_audio,
                    dropped_json=out.dropped_json,
                    last_send_s=round(now - out.last_send, 1),
                )
            wall_d, mono_d = time.time() - mon["wall0"], now - mon["mono0"]
            if wall_d - mono_d > 2.0:  # CLOCK_MONOTONIC does not advance while the host is suspended
                diag.event("host_suspend", "warning", cid, gap_s=round(wall_d - mono_d, 1))
            mon["wall0"], mon["mono0"] = time.time(), now
            fm = mon["fm"]
            if run and now - fm["t0"] >= 30 and fm["frames"]:
                rate = fm["detected"] / fm["frames"]
                cs, ws_ = fm["confs"], fm["widths"]
                diag.event(
                    "face_miss_summary",
                    "warning" if rate < 0.6 else "info",
                    cid,
                    window_s=round(now - fm["t0"]),
                    frames=fm["frames"],
                    face_frames=fm["detected"],
                    hit_rate=round(rate, 2),
                    conf_avg=round(sum(cs) / len(cs), 2) if cs else None,
                    conf_min=round(min(cs), 2) if cs else None,
                    face_w_avg=round(sum(ws_) / len(ws_), 3) if ws_ else None,
                    out_of_roi=fm["out_roi"],
                    coast_frames=fm["coast"],
                    min_conf=st["tr"].cfg.min_conf,
                )
                mon["fm"] = {"frames": 0, "detected": 0, "confs": [], "widths": [], "out_roi": 0, "coast": 0, "t0": now}
            if now - last_hb >= HEARTBEAT_S:
                stats = self.stats
                fr = {"received": stats.received, "processed": stats.processed, "dropped": stats.dropped}
                c = self.proc.counts
                snap = st["tr"].snapshot()
                diag.event(
                    "heartbeat",
                    conn=cid,
                    running=run,
                    frames_rx=fr.get("received"),
                    frames_done=fr.get("processed"),
                    frames_dropped=fr.get("dropped"),
                    fps=round((mon["frames_done"] - last_frames) / (now - last_hb), 2),
                    detect_calls=c.get("detect_calls"),
                    detect_hits=c.get("detect_hits"),
                    tracks=len(snap.get("tracks", [])),
                    state=snap.get("state"),
                    p_group=snap.get("p_group"),
                    audio_rx=aud["recv"],
                    audio_done=aud["processed"],
                    audio_dropped=aud["dropped"],
                    audio_rate=round((aud["processed"] - last_audio) / (now - last_hb), 1),
                    audio_queue=aud["q"].qsize(),
                    rtf=aud["rtf"],
                    xcheck=aud.get("counts"),
                    send_backlog=out.backlog(),
                    send_dropped=out.dropped_audio + out.dropped_json,
                    send_ms_max=round(out.send_ms_max, 1),
                    loop_lag_ms=round(mon["loop_lag_max"], 1),
                    vision_lock_held=vision_lock.locked(),
                    reid_calls=rd.calls,
                    reid_tokens=rd.tokens,
                    reid_cost_est=round(rd.calls * st["tr"].cfg.reid_cost_per_call, 4),
                    reid_refs=len(rd.store.refs),
                )
                mon["loop_lag_max"] = 0.0
                out.send_ms_max = 0.0
                last_hb, last_frames, last_audio = now, mon["frames_done"], aud["processed"]

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Subscribe to diagnostics and start the supervised workers (ticker, vision, audio, sender, watchdog)."""
        self._unsub = diag.subscribe(self.notify_ui)
        for name in self.factories:
            self.spawn(name)
        self.tasks["watchdog"] = asyncio.create_task(diag.supervise("watchdog", self.watchdog, self.cid))

    def _admit(self, frame: dict, wire_bytes: int) -> None:
        # Application wire bytes: complete JSON UTF-8 or binary header + JPEG.
        self.stats.on_received(wire_bytes)
        self.mon["frame_rx_t"] = time.monotonic()
        if self.slot["frame"] is not None:
            self.stats.dropped += 1
        frame["_received_perf"] = time.perf_counter()
        frame["_received_mono"] = self.mon["frame_rx_t"]
        self.slot["frame"] = frame
        self.wake.set()

    async def on_bytes(self, b: bytes) -> None:
        """One binary packet: 0x04 video (latest frame wins) or 0x01 PCM16 audio; anything else is ignored."""
        # PRIVACY: audio/video payloads stay in memory, never logged/written.
        st, aud, mon = self.st, self.aud, self.mon
        if b and b[0] == VIDEO_TAG:
            if not st["running"]:
                return
            try:
                frame = binary_frame(b)
            except ValueError:
                return
            self._admit(frame, len(b))
            return
        if len(b) > 9 and b[0] == AUDIO_IN_TAG and st["tr"].cfg.audio_pipeline == "puresound":
            aud["recv"] += 1
            mon["audio_rx_t"] = time.monotonic()
            if aud["q"].full():
                aud["q"].get_nowait()
                aud["dropped"] += 1
            aud["q"].put_nowait((struct.unpack("<d", b[1:9])[0], b[9:], mon["audio_rx_t"]))

    async def on_text(self, text: str | None) -> None:
        """One JSON text packet (raises on invalid JSON, like v0.12, which ends the connection)."""
        text = text or "{}"
        msg = json.loads(text)
        kind = msg.get("type")
        if kind == "frame":
            if not self.st["running"]:
                return
            try:
                frame = legacy_frame(msg)
            except ValueError:
                return
            self._admit(frame, len(text.encode("utf-8")))
            return
        await self.handle_message(kind, msg)

    async def handle_message(self, kind, msg: dict) -> None:
        """Control messages (see :data:`presence_demo.runtime.protocol.CLIENT_MESSAGE_TYPES`); unknown kinds are
        ignored."""
        st, aud, mon, rd, out, cid, stats = self.st, self.aud, self.mon, self.rd, self.out, self.cid, self.stats
        clock = self.observation_clock
        async with self.lock:
            tr = st["tr"]
            if kind == "ping":
                if msg.get("rtt") is not None:
                    stats.rtt.append(float(msg["rtt"]))
                out.post({"type": "pong", "t": msg.get("t")})
            elif kind == "vad":
                if st["running"] and clock.paused_at is None and not self.clean_active():
                    self.send(tr.set_speech(bool(msg.get("speech"))), droppable=False)
            elif kind == "config":
                new = msg.get("config", {})
                before = asdict(tr.cfg)
                warns = tr.cfg.update(new)
                after = asdict(tr.cfg)
                changed = {k: after[k] for k in after if after[k] != before.get(k)}
                if not tr.cfg.reid_enabled or not tr.cfg.reid_consent:
                    rd.wipe("disabled")
                if changed:
                    diag.event("config", conn=cid, **changed)
                if warns:
                    diag.event("config_warnings", "warning", cid, warnings="; ".join(warns)[:300])
                self.send(tr.tick() if st["running"] else tr.snapshot(), {"config_ack": True, "warnings": warns}, False)
            elif kind == "calibrate":
                warns = self.ctx.calibrate(tr, msg.get("meters"))
                aud["calibrated"] = aud["calibrated"] or any(w.startswith("calibrated") for w in warns)
                diag.event("calibrate", conn=cid, result="; ".join(warns)[:200])
                self.send(tr.tick() if st["running"] else tr.snapshot(), {"config_ack": True, "warnings": warns}, False)
            elif kind == "client":  # browser-side report: visibility / auto-pause / media ended / reconnect
                what = str(msg.get("what") or "")[:32]
                info = {
                    k: (v if isinstance(v, (int, float, bool)) else str(v)[:64])
                    for k, v in msg.items()
                    if k not in ("type", "what", "level", "conn", "_event")
                }
                diag.event(
                    "client",
                    "warning" if what in ("media_ended", "reconnect") else "info",
                    cid,
                    what=what,
                    **dict(list(info.items())[:8]),
                )
            elif kind == "reid_wipe":
                rd.wipe("ui")
                out.post({"type": "reid", **rd.status()})
            elif kind == "record":  # client-side recording on/off (PRIVACY: the server never writes audio)
                aud["rec"] = bool(msg.get("on"))
                diag.event("record_stream", conn=cid, on=aud["rec"])
            elif kind == "start":
                st["running"] = True
                mon["frame_done_t"] = time.monotonic()
                clock.observed()
                diag.event("start", conn=cid)
                self.send(tr.snapshot(), droppable=False)
            elif kind in ("stop", "reset"):
                self.proc.reset()
                stats.counts = self.proc.counts
                st["running"] = kind == "reset" and st["running"]
                if kind == "stop":
                    aud["rec"] = False
                diag.event(kind, conn=cid)
                rd.interrupt()
                previous_session = tr.session
                clock.resume()
                if st["running"]:
                    clock.observed()
                st["tr"] = PresenceTracker(tr.cfg, clock=clock)
                st["tr"].session = previous_session
                if kind == "reset":
                    rd.wipe("reset")
                self.slot["frame"] = None
                aud["vision"] = None
                self.send(st["tr"].snapshot(), {"detections": []}, False)

    async def close(self, reason: str) -> None:
        """Release the page (or the private re-ID store), cancel workers, close ASR, log ``conn_close``."""
        st = self.st
        if self._unsub is not None:
            self._unsub()
        if self.page:
            self.ctx.pages.release(self.page, self.page_generation)
        else:
            self.rd.close()
        if st["running"] and st["tr"].state.value != "ENDED":
            diag.event("session_interrupted", conn=self.cid, session=st["tr"].session, reason="transport_closed")
        for t in self.tasks.values():
            t.cancel()
        with contextlib.suppress(Exception):
            await self.asr_close()
        diag.event(
            "conn_close",
            conn=self.cid,
            reason=reason,
            duration_s=round(time.monotonic() - self.t_open, 1),
            frames=self.mon["frames_done"],
            audio_chunks=self.aud["processed"],
            send_dropped=self.out.dropped_audio + self.out.dropped_json,
        )

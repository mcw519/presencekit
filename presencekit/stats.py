"""Resource / latency statistics for the monitor panel."""

from __future__ import annotations

import os
import time
from collections import deque

import psutil


def _pct(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[min(len(s) - 1, int(round(q * (len(s) - 1))))])


class ConnStats:
    """Per-connection rolling statistics (last N samples per timing stage).

    Vision queue_wait is server receipt -> async worker start; executor_wait is
    submission -> thread start; lock_wait is shared vision lock acquisition.
    total spans async worker start -> tracker completion and excludes queue_wait.
    """

    def __init__(self, window: int = 60, clock=time.monotonic):
        self.clock = clock
        self.lat = {
            k: deque(maxlen=window)
            for k in (
                "queue_wait",
                "executor_wait",
                "lock_wait",
                "decode",
                "motion",
                "detect",
                "embed",
                "appearance",
                "pose",
                "tracker",
                "total",
            )
        }
        self.frame_times = deque(maxlen=window)
        self.bytes = deque(maxlen=window)  # (t, nbytes)
        self.received = self.processed = self.dropped = 0
        self.rtt = deque(maxlen=20)
        self.last_size = (0, 0)
        self.counts: dict = {}
        self.recommended_fps = 0.0

    def on_received(self, nbytes: int):
        self.received += 1
        self.bytes.append((self.clock(), nbytes))

    def on_processed(self, timings: dict):
        self.processed += 1
        self.frame_times.append(self.clock())
        for k in self.lat:
            if k in timings:
                self.lat[k].append(float(timings[k]))
        if timings.get("w"):
            self.last_size = (timings["w"], timings["h"])

    def fps(self, horizon=5.0) -> float:
        now = self.clock()
        n = sum(1 for t in self.frame_times if now - t <= horizon)
        return n / horizon

    def bandwidth(self, horizon=5.0) -> float:
        now = self.clock()
        return sum(b for t, b in self.bytes if now - t <= horizon) / horizon

    def _optim(self) -> dict:
        c = self.counts or {}
        frames = c.get("frames", 0)
        emb = c.get("embed_computed", 0) + c.get("embed_reused", 0)
        return {
            "skipped": c.get("skipped", 0),
            "skipped_pct": round(100 * c.get("skipped", 0) / frames, 1) if frames else 0.0,
            "detect_calls": c.get("detect_calls", 0),
            "embed_computed": c.get("embed_computed", 0),
            "embed_reused": c.get("embed_reused", 0),
            "embed_reused_pct": round(100 * c.get("embed_reused", 0) / emb, 1) if emb else 0.0,
            "recommended_fps": self.recommended_fps,
            "pose_calls": c.get("pose_calls", 0),
            "pose_reused": c.get("pose_reused", 0),
            "pose_reused_pct": round(
                100 * c.get("pose_reused", 0) / max(1, c.get("pose_calls", 0) + c.get("pose_reused", 0)), 1
            ),
            "detect_hits": c.get("detect_hits", 0),
        }

    def snapshot(self) -> dict:
        b = [x for _, x in self.bytes]
        return {
            "latency_ms": {
                k: {
                    "avg": round(sum(v) / len(v), 2) if v else 0.0,
                    "p95": round(_pct(v, 0.95), 2),
                    "last": round(v[-1], 2) if v else 0.0,
                }
                for k, v in self.lat.items()
            },
            "fps": round(self.fps(), 2),
            "frames": {"received": self.received, "processed": self.processed, "dropped": self.dropped},
            "frame_bytes_avg": round(sum(b) / len(b)) if b else 0,
            "bandwidth_Bps": round(self.bandwidth()),
            "frame_size": list(self.last_size),
            "optim": self._optim(),
            "rtt_ms": round(sum(self.rtt) / len(self.rtt), 1) if self.rtt else None,
        }


class AudioStats:
    """Last five wall-clock seconds of block timings, bounded even for tiny PCM chunks.

    Processing covers the complete AudioStream.process call. Queue wait covers server
    receipt through that block's processing start, including executor and batch waiting.
    RTF is sum(processing seconds) / sum(input audio seconds).
    """

    def __init__(self, horizon=5.0, clock=time.monotonic, limit=2000):
        self.horizon, self.clock = horizon, clock
        self.samples = deque(maxlen=limit)

    def record(self, done, process_s, audio_s, queue_wait_s):
        self.samples.append((done, process_s, audio_s, queue_wait_s))

    def snapshot(self):
        cutoff = self.clock() - self.horizon
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        audio_s = sum(s[2] for s in self.samples)
        proc = [s[1] * 1000 for s in self.samples]
        waits = [s[3] * 1000 for s in self.samples]
        return {
            "rtf_recent": round(sum(s[1] for s in self.samples) / audio_s, 3) if audio_s else None,
            "proc_p95_ms": round(_pct(proc, 0.95), 2) if proc else None,
            "queue_wait_ms_avg": round(sum(waits) / len(waits), 2) if waits else None,
            "queue_wait_ms_p95": round(_pct(waits, 0.95), 2) if waits else None,
        }


_proc = psutil.Process(os.getpid())
_proc.cpu_percent(None)
psutil.cpu_percent(None, percpu=True)


def system_stats() -> dict:
    mem = psutil.virtual_memory()
    per = psutil.cpu_percent(None, percpu=True)
    try:
        load = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        load = None
    return {
        "proc_cpu": round(_proc.cpu_percent(None), 1),
        "proc_rss": _proc.memory_info().rss,
        "proc_threads": _proc.num_threads(),
        "sys_cpu": round(sum(per) / len(per), 1) if per else 0.0,
        "per_core": per,
        "mem_total": mem.total,
        "mem_used": mem.total - mem.available,
        "mem_pct": mem.percent,
        "load": load,
    }


def build_stats(conn: ConnStats, info: dict) -> dict:
    return {
        "type": "stats",
        "t": time.time(),
        "system": system_stats(),
        "conn": conn.snapshot(),
        "info": info,
    }

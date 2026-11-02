"""Isolated offline-model WebSocket replay; no devices, downloads or cloud calls.

This exercises the Python server, not Chrome or Jev quality. The child process is
bound to a fresh loopback socket and preloads strictly local models before replay.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import math
import os
import socket
import statistics
import struct
import subprocess
import sys
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE_CAPACITY = 25  # Production ingress audio queue, 20 ms blocks.
MAX_SAMPLES = 3601  # One-second samples, maximum supported run of one hour.
METHOD = {
    "transport": "isolated loopback WebSocket; binary JPEG + mono PCM16 16 kHz, 20 ms blocks",
    "audio": "deterministic synthetic noise, seed 42; no speech recognition quality claim",
    "video": "640x480 blank or two placements of the same local test face, at 2 fps",
    "models": "local YuNet + SFace + MoveNet + PureSound CPU; downloads disabled",
    "cloud": "ASR none; re-ID off (mock behavior covered by functional suite)",
    "sampling": "one-second recent-5s RTF/ingress queue/drop counters; RSS read from child PID",
    "queue_gate": "late-quarter mean <= early-quarter mean + 1 block, and max < 25 blocks",
    "qualification": "30-minute gate requires >=1800 s measured after warmup, >=95% sampling coverage",
    "limitations": "No Chrome, physical devices, network faults, real Jev, or original RTF 0.37 reproduction",
}


def _case(suite, name, status, expected, actual, details=None):
    return {
        "id": f"{suite}.{name}",
        "suite": suite,
        "status": status,
        "expected": expected,
        "actual": actual,
        "details": {"method": METHOD, **(details or {})},
    }


def _percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)] if ordered else None


def summarize_samples(samples, *, suite, seconds, dropped_baseline=0, send_dropped_baseline=0, error_count=0):
    """Evaluate one-second measurements; missing/stale data never passes a gate.

    Samples carry t (relative seconds), rtf_recent, queue, audio_dropped,
    rss_bytes, and audio_age_seconds. Raw media is never retained.
    """
    samples = list(samples)
    expected_count = max(1, math.floor(seconds))
    valid = [
        row
        for row in samples
        if all(
            isinstance(row.get(k), (int, float)) and math.isfinite(row[k])
            for k in ("t", "rtf_recent", "queue", "audio_dropped", "rss_bytes", "audio_age_seconds")
        )
        and 0 <= row["audio_age_seconds"] <= 1.0
        and row["rtf_recent"] >= 0
        and row["rss_bytes"] > 0
        and 0 <= row["queue"] <= QUEUE_CAPACITY
    ]
    coverage = len(valid) / expected_count
    span = max((row["t"] for row in valid), default=0)
    adequate = len(valid) >= 3 and coverage >= 0.95 and span >= seconds - 1.5
    qualified = adequate and (suite != "soak" or seconds >= 1800)
    details = {
        "seconds": seconds,
        "qualification": qualified,
        "sample_count": len(samples),
        "valid_samples": len(valid),
        "coverage": round(coverage, 4),
        "sample_span_seconds": span,
    }
    results = [
        _case(
            suite,
            "measurement_coverage",
            "passed" if adequate else "failed",
            {"coverage_min": 0.95, "fresh_audio_max_age_s": 1.0},
            details,
            details,
        )
    ]
    if not adequate:
        return results + [_case(suite, "performance_gates", "not_run", "Fresh complete measurements", None, details)]
    rtfs = [row["rtf_recent"] for row in valid]
    p95 = _percentile(rtfs, 0.95)
    results.append(
        _case(suite, "rtf", "passed" if p95 <= 0.5 else "failed", {"recent_5s_rtf_p95_max": 0.5}, {"p95": p95}, details)
    )
    dropped = max(row["audio_dropped"] for row in valid) - dropped_baseline
    send_dropped = max(row.get("send_dropped", 0) for row in valid) - send_dropped_baseline
    results.append(
        _case(
            suite,
            "audio_dropped",
            "passed" if dropped == 0 and send_dropped == 0 else "failed",
            {"dropped": 0, "send_dropped": 0},
            {"dropped": dropped, "send_dropped": send_dropped},
            details,
        )
    )
    results.append(
        _case(
            suite, "errors", "passed" if error_count == 0 else "failed", {"errors": 0}, {"errors": error_count}, details
        )
    )
    quarter = max(1, len(valid) // 4)
    early = statistics.mean(row["queue"] for row in valid[:quarter])
    late = statistics.mean(row["queue"] for row in valid[-quarter:])
    peak = max(row["queue"] for row in valid)
    results.append(
        _case(
            suite,
            "queue",
            "passed" if late <= early + 1 and peak < QUEUE_CAPACITY else "failed",
            {"late_minus_early_blocks_max": 1, "peak_blocks_below": QUEUE_CAPACITY},
            {"early_mean": early, "late_mean": late, "peak": peak},
            details,
        )
    )
    if suite == "soak":
        first = [row["rss_bytes"] for row in valid if row["t"] <= 600]
        last = [row["rss_bytes"] for row in valid if row["t"] > seconds - 600]
        memory_ready = qualified and len(first) >= 570 and len(last) >= 570
        growth = (statistics.mean(last) - statistics.mean(first)) / (1 << 20) if memory_ready else None
        results.append(
            _case(
                suite,
                "rss_growth",
                "not_run" if not memory_ready else "passed" if growth <= 30 else "failed",
                {"last_vs_first_10min_growth_mib_max": 30},
                {"growth_mib": growth},
                details,
            )
        )
        results.append(
            _case(
                suite,
                "30_minute_qualification",
                "passed" if qualified else "not_run",
                {"measured_seconds_min": 1800, "coverage_min": 0.95},
                details,
                details,
            )
        )
    # One-second numeric metrics are bounded; no JPEG, PCM or server payloads are written.
    results.append(
        _case(
            suite,
            "measurements",
            "passed",
            "Bounded numeric-only samples",
            {
                "samples": [
                    {
                        key: row[key]
                        for key in (
                            "t",
                            "rtf_recent",
                            "queue",
                            "audio_dropped",
                            "rss_bytes",
                            "audio_age_seconds",
                            "send_dropped",
                        )
                        if key in row
                    }
                    for row in valid[:MAX_SAMPLES]
                ]
            },
            details,
        )
    )
    return results


def _preflight(scene):
    if os.name != "posix":
        return "This isolated inherited-socket adapter currently requires a POSIX host."
    for package in (
        "cv2",
        "numpy",
        "psutil",
        "websockets",
        "uvicorn",
        "onnxruntime",
        "puresound_streaming",
        "ai_edge_litert",
    ):
        if importlib.util.find_spec(package) is None:
            return f"Local dependency unavailable: {package}; no installation attempted."
    models = ROOT / "models"
    names = (
        "face_detection_yunet_2023mar.onnx",
        "face_recognition_sface_2021dec.onnx",
        "movenet_multipose_lightning_fp16.tflite",
        "dpcrn_curriculum_v2.onnx",
        "dpcrn_curriculum_v2.json",
    )
    missing = [name for name in names if not (models / name).is_file() or not (models / name).stat().st_size]
    if missing:
        return "Local model files missing: " + ", ".join(missing) + "; no download attempted."
    if scene == "twofaces" and not (ROOT / "tests" / "data" / "face.jpg").is_file():
        return "Local face fixture missing; no download attempted."
    return None


def _health(port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}/_acceptance/health", timeout=2) as response:
        return json.load(response)


def _child_environment():
    # No service credentials, token, user-supplied model paths or configuration reach the child.
    allowed = ("PATH", "HOME", "TMPDIR", "SYSTEMROOT", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
    return {**{key: os.environ[key] for key in allowed if key in os.environ}, "PYTHONDONTWRITEBYTECODE": "1"}


async def _replay(port, pid, seconds, warmup_seconds, scene):
    import cv2
    import numpy as np
    import psutil
    from websockets.asyncio.client import connect

    image = np.full((480, 640, 3), 80, np.uint8)
    if scene == "twofaces":
        face = cv2.imread(str(ROOT / "tests" / "data" / "face.jpg"))
        if face is None:
            raise ValueError("Local face fixture cannot be decoded")
        for left in (60, 360):
            image[60:420, left : left + 220] = cv2.resize(face, (220, 360))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    jpeg = encoded.tobytes()
    pcm = (np.random.default_rng(42).normal(0, 0.01, 320) * 32767).astype("<i2").tobytes()
    process = psutil.Process(pid)
    latest = {"audio": None, "audio_t": None, "vision_processed": 0, "errors": 0}
    stop = asyncio.Event()
    samples = deque(maxlen=MAX_SAMPLES)
    async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2**22, proxy=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "config",
                    "config": {
                        "audio_pipeline": "puresound",
                        "pose_enabled": True,
                        "adaptive_fps": False,
                        "reid_enabled": False,
                        "reid_consent": False,
                        "asr_provider": "none",
                        "tracker_mode": "kalman",
                        "presence_model": "hmm",
                        "intent_model": "score",
                        "priority_mode": "vision",
                    },
                }
            )
        )
        await ws.send(json.dumps({"type": "start"}))

        async def receive():
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "audio":
                    if msg.get("available"):
                        latest["audio"], latest["audio_t"] = msg, time.monotonic()
                    else:
                        raise RuntimeError("PureSound unavailable in isolated server")
                elif msg.get("type") == "stats":
                    latest["vision_processed"] = msg.get("conn", {}).get("frames", {}).get("processed", 0)
                elif msg.get("type") == "error":
                    latest["errors"] += 1
            if not stop.is_set():
                raise RuntimeError("WebSocket closed before replay finished")

        async def audio():
            target = time.monotonic()
            while not stop.is_set():
                await asyncio.sleep(max(0, target - time.monotonic()))
                await ws.send(b"\x01" + struct.pack("<d", time.perf_counter() * 1000) + pcm)
                # Missed deadlines skip waiting, matching a real capture stream's continuous samples.
                target += 0.02

        async def video():
            while not stop.is_set():
                await ws.send(struct.pack("<BdB", 4, time.perf_counter() * 1000, 0) + jpeg)
                await asyncio.sleep(0.5)

        tasks = [asyncio.create_task(fn()) for fn in (receive, audio, video)]
        try:
            warm_start = time.monotonic()
            deadline = warm_start + max(30, warmup_seconds + 15)
            while True:
                for task in tasks:
                    if task.done():
                        task.result()
                        raise RuntimeError("Replay worker stopped during warmup")
                now = time.monotonic()
                if now - warm_start >= warmup_seconds and latest["audio"] and latest["vision_processed"]:
                    break
                if now > deadline:
                    raise RuntimeError("Warmup failed: fresh audio and processed vision required")
                await asyncio.sleep(0.1)
            baseline = await asyncio.to_thread(_health, port)
            drop_baseline = latest["audio"].get("dropped", 0)
            send_drop_baseline = latest["audio"].get("send_dropped", 0)
            measured_start = time.monotonic()
            for index in range(1, math.ceil(seconds) + 1):
                await asyncio.sleep(max(0, measured_start + min(index, seconds) - time.monotonic()))
                for task in tasks:
                    if task.done():
                        task.result()
                        raise RuntimeError("Replay worker stopped before measurement finished")
                msg = latest["audio"] or {}
                samples.append(
                    {
                        "t": round(time.monotonic() - measured_start, 3),
                        "rtf_recent": msg.get("rtf_recent"),
                        "queue": msg.get("queue"),
                        "audio_dropped": msg.get("dropped"),
                        "send_dropped": msg.get("send_dropped"),
                        "rss_bytes": process.memory_info().rss,
                        "audio_age_seconds": round(time.monotonic() - (latest["audio_t"] or 0), 3),
                    }
                )
            final = await asyncio.to_thread(_health, port)
            errors = final["errors"] - baseline["errors"] + latest["errors"]
            return (
                list(samples),
                drop_baseline,
                send_drop_baseline,
                errors,
                {
                    "pid": pid,
                    "warmup_seconds": round(measured_start - warm_start, 3),
                    "scene": scene,
                    "error_events": final["error_events"],
                },
            )
        finally:
            stop.set()
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "stop"}))
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def _isolated_run(seconds, warmup_seconds, scene):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    child = None
    try:
        child = subprocess.Popen(
            [sys.executable, "-m", "presencekit.acceptance_pipeline", "--serve-fd", str(listener.fileno())],
            cwd=ROOT,
            pass_fds=(listener.fileno(),),
            env=_child_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 60
        while True:
            if child.poll() is not None:
                raise RuntimeError(f"Isolated model server exited during startup (code {child.returncode})")
            try:
                health = await asyncio.to_thread(_health, port)
                if health["pid"] != child.pid:
                    raise RuntimeError("Unexpected server identity")
                break
            except (OSError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Isolated model server startup timed out") from None
                await asyncio.sleep(0.1)
        return await _replay(port, child.pid, seconds, warmup_seconds, scene)
    finally:
        listener.close()
        if child is not None:
            child.terminate()
            try:
                await asyncio.to_thread(child.wait, timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                await asyncio.to_thread(child.wait, timeout=5)


def _run(suite, seconds, warmup_seconds, scene):
    if not math.isfinite(seconds) or not 3 <= seconds <= 3600:
        raise ValueError("seconds must be finite and in 3..3600")
    if not math.isfinite(warmup_seconds) or not 5 <= warmup_seconds <= 120:
        raise ValueError("warmup_seconds must be finite and in 5..120")
    if scene not in ("blank", "twofaces"):
        raise ValueError("scene must be blank or twofaces")
    missing = _preflight(scene)
    if missing:
        return [
            _case(suite, "availability", "not_run", "All local model/runtime prerequisites", None, {"reason": missing})
        ]
    try:
        samples, baseline, send_baseline, errors, info = asyncio.run(_isolated_run(seconds, warmup_seconds, scene))
        cases = summarize_samples(
            samples,
            suite=suite,
            seconds=seconds,
            dropped_baseline=baseline,
            send_dropped_baseline=send_baseline,
            error_count=errors,
        )
        for case in cases:
            case["details"].update(info)
        return cases
    except Exception as exc:
        return [
            _case(
                suite,
                "execution",
                "failed",
                "Replay completes with fresh metrics",
                type(exc).__name__,
                {"reason": str(exc), "seconds": seconds, "qualification": False},
            )
        ]


def run_pipeline(seeds=20, seconds=10, *, warmup_seconds=5, scene="twofaces"):
    """One fixed-seed workload; seeds is accepted for the common suite adapter API."""
    return _run("pipeline", seconds, warmup_seconds, scene)


def run_soak(seeds=20, seconds=1800, *, warmup_seconds=10, scene="twofaces"):
    """A shorter smoke run never passes the 30-minute or RSS qualification gates."""
    return _run("soak", seconds, warmup_seconds, scene)


def _serve(fd):
    """Child-only entry point: disable download paths before app imports."""
    import uvicorn

    from . import diag, vision
    from .audio import Isolator
    from .runtime import resources as runtime  # owner of the process-wide pipeline (v0.12: web.runtime)

    def no_download(*args, **kwargs):
        raise RuntimeError("Downloads disabled by acceptance adapter")

    vision.ensure_model = vision.ensure_movenet = no_download
    # All cloud configuration is disabled by the replay; also reject Python-level remote connects.
    original_connect = socket.socket.connect

    def local_only(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise RuntimeError("External networking disabled by acceptance adapter")
        return original_connect(sock, address)

    socket.socket.connect = local_only
    runtime._pipeline = vision.VisionPipeline(allow_download=False)
    runtime._pose_cache["det"] = vision.load_pose_detector(allow_download=False)
    if runtime._pipeline.embed_status["active"] != "fp32" or runtime._pose_cache["det"] is None:
        raise RuntimeError("Required real models did not load; fallback is not accepted")
    Isolator()  # Shared ORT session loaded before exposing the ready endpoint.
    from .server import app

    error_events = Counter()

    def observe(event):
        if event.get("level") in ("error", "critical"):
            error_events[event["event"]] += 1

    diag.subscribe(observe)

    @app.get("/_acceptance/health")
    def health():
        return {"pid": os.getpid(), "errors": sum(error_events.values()), "error_events": dict(error_events)}

    server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_config=None, ws_max_size=8 * 1024 * 1024))
    server.run(sockets=[socket.socket(fileno=fd)])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-fd", type=int, required=True)
    _serve(parser.parse_args().serve_fd)

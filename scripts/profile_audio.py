"""Stage-level synthetic audio profile: hot loops vs 20 ms real-time pacing.

No microphone, media files or cloud calls. Run with the project's PureSound extra.
Thread CPU time is measured separately from elapsed time, excluding pacing sleeps.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
import contextlib
import ctypes
import json
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presencekit.audio import SR, AudioStream  # noqa: E402
from presencekit.config import Config  # noqa: E402


def qos(value):
    if value == "default":
        return None
    if sys.platform != "darwin":
        raise RuntimeError("--qos requires macOS")
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    fn = lib.pthread_set_qos_class_self_np
    fn.argtypes, fn.restype = [ctypes.c_uint, ctypes.c_int], ctypes.c_int
    result = fn({"user-initiated": 0x19, "utility": 0x11}[value], 0)  # from sys/qos.h
    if result:
        raise RuntimeError(f"pthread_set_qos_class_self_np: {result}")
    return result


def current_qos():
    if sys.platform != "darwin":
        return None
    fn = ctypes.CDLL("/usr/lib/libSystem.B.dylib").qos_class_self
    fn.restype = ctypes.c_uint
    return hex(fn())


def scheduler_state():
    result = {"t": time.time(), "qos": current_qos()}
    if sys.platform == "darwin":
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        fn = lib.getpriority
        fn.argtypes, fn.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
        for name, selector in (("thread_background", 3), ("process_background", 4)):
            ctypes.set_errno(0)
            value = fn(selector, 0)  # PRIO_DARWIN_THREAD/PROCESS, from sys/resource.h
            result[name] = value if not ctypes.get_errno() else None
    return result


class Meter:
    def __init__(self):
        self.values = {}
        self.enabled = False
        self.samples = {}

    def wrap(self, obj, name, label):
        original = getattr(obj, name)

        def measured(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            wall, cpu = time.perf_counter(), time.thread_time()
            try:
                return original(*args, **kwargs)
            finally:
                w, c = time.perf_counter() - wall, time.thread_time() - cpu
                values = self.values.setdefault(label, [0, 0.0, 0.0])
                values[0] += 1
                values[1] += w
                values[2] += c
                self.samples.setdefault(label, collections.deque(maxlen=1000)).append((w, c))

        setattr(obj, name, measured)


class SessionMeter:
    def __init__(self, session, meter):
        self.original = session
        self.calls = 0
        self.last_done = None
        self.trace = collections.deque(maxlen=2000)
        self.meter = meter
        meter.wrap(self, "run", "ort")

    def run(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.original.run(*args, **kwargs)
        done = time.perf_counter()
        if self.meter.enabled:
            self.trace.append((self.calls % 2, done - start, start - self.last_done if self.last_done else 0))
        self.calls += 1
        self.last_done = done
        return result

    def summary(self, trace=None):
        trace = self.trace if trace is None else trace
        return {
            name: {
                "calls": len(rows),
                "mean_ms": float(np.mean([r[1] for r in rows]) * 1000),
                "p95_ms": float(np.percentile([r[1] for r in rows], 95) * 1000),
                "gap_mean_ms": float(np.mean([r[2] for r in rows]) * 1000),
            }
            for name, rows in (
                ("after_gap_gt_5ms", [r for r in trace if r[2] > 0.005]),
                ("after_gap_lte_5ms", [r for r in trace if r[2] <= 0.005]),
            )
            if rows
        }


def build(signal, seconds):
    rng = np.random.default_rng(42)
    samples = int(SR * (seconds + 0.5))
    scale = {"noise": 0.03, "quiet": 0.0003, "silence": 0.0}[signal]
    pcm = (rng.normal(0, scale, samples) * 32767).clip(-32768, 32767).astype("<i2")
    return [pcm[i : i + 320].tobytes() for i in range(0, len(pcm) - 319, 320)]


async def one(mode, signal, args):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers, initializer=lambda: qos(args.qos))
    meter = Meter()
    stream = AudioStream(Config())
    provider = stream.iso.proc.session.get_providers()
    session_meter = SessionMeter(stream.iso.proc.session, meter)
    stream.iso.proc.session = session_meter
    meter.wrap(stream.iso.proc, "process_samples", "sdk")
    meter.wrap(stream.iso, "process", "isolator")
    meter.wrap(stream.vad, "step", "vad")
    meter.wrap(stream.raw_vad, "step", "vad")
    meter.wrap(stream.adjuster, "view", "crosscheck")
    meter.wrap(stream.adjuster, "decide", "crosscheck")
    blocks = build(signal, args.seconds)
    elapsed, cpu_times = [], []
    observed_qos = set()
    observed_scheduler = []
    loop = asyncio.get_running_loop()
    worker_ids = []

    if args.prestart_workers:
        barrier = threading.Barrier(args.workers)
        await asyncio.gather(*(loop.run_in_executor(pool, barrier.wait) for _ in range(args.workers)))

    def process(i):
        if i == 25:
            observed_qos.add(current_qos())
            observed_scheduler.append(scheduler_state())
        meter.enabled = i >= 25  # half a second of input warm-up
        wall, cpu = time.perf_counter(), time.thread_time()
        stream.process(blocks[i], None, time.monotonic())
        if meter.enabled:
            worker_ids.append(threading.get_ident())
            elapsed.append(time.perf_counter() - wall)
            cpu_times.append(time.thread_time() - cpu)

    def burst():
        for i in range(len(blocks)):
            process(i)

    try:
        if mode == "worker_burst":
            await loop.run_in_executor(pool, burst)
        elif mode == "main_burst":
            burst()
        else:
            start = time.monotonic()
            for i in range(len(blocks)):
                await asyncio.sleep(max(0, start + i * 0.02 - time.monotonic()))
                if mode == "worker_paced":
                    await loop.run_in_executor(pool, process, i)
                else:
                    process(i)
    finally:
        pool.shutdown(wait=True)
    audio_s = len(elapsed) * 0.02
    stages = {}
    for key, (n, wall, cpu) in meter.values.items():
        stages[key] = {"calls": n, "wall_rtf": wall / audio_s, "thread_cpu_rtf": cpu / audio_s}
    return {
        "mode": mode,
        "signal": signal,
        "audio_seconds": audio_s,
        "providers": provider,
        "observed_worker_qos": list(observed_qos),
        "observed_worker_scheduler": observed_scheduler,
        "worker_threads_used": len(set(worker_ids)),
        "worker_switches": sum(a != b for a, b in zip(worker_ids, worker_ids[1:], strict=False)),
        "workers_configured": args.workers,
        "workers_prestarted": args.prestart_workers,
        "rtf": sum(elapsed) / audio_s,
        "thread_cpu_rtf": sum(cpu_times) / audio_s,
        "block_p50_ms": float(np.median(elapsed) * 1000),
        "block_p95_ms": float(np.percentile(elapsed, 95) * 1000),
        "stages_nested": stages,
        "ort_gap_groups_last_2000_calls": session_meter.summary(),
    }


async def main(args):
    import onnxruntime as ort

    qos(args.qos)
    results = []
    for signal in args.signal:
        for mode in args.mode:
            result = await one(mode, signal, args)
            results.append(result)
            print(json.dumps(result), flush=True)
    output = {
        "ort": ort.__version__,
        "machine": platform.machine(),
        "qos": args.qos,
        "disable_kleidiai": args.disable_kleidiai,
        "method": "synthetic PCM16, 20 ms blocks, 0.5 s input warm-up; stage times are nested, not additive",
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n")


def serve(args):
    """Profile real WebSocket jobs on a separate localhost server until Ctrl-C."""
    import onnxruntime as ort
    import uvicorn

    from presencekit.server import app

    if args.browser_synthetic:
        from fastapi.responses import FileResponse, HTMLResponse, Response

        from presencekit.server import index

        def synthetic_page():
            html = index().body.decode()
            synthetic_script = (ROOT / "scripts/benchmark_browser_stream.js").read_text()
            html = html.replace("<head>", "<head><script>" + synthetic_script + "</script>")
            return HTMLResponse(html)

        app.add_api_route("/debug/synthetic", synthetic_page, methods=["GET"])
        app.add_api_route(
            "/debug/synthetic-input.js",
            lambda: Response((ROOT / "scripts/benchmark_browser_stream.js").read_text(), media_type="text/javascript"),
            methods=["GET"],
        )
        app.add_api_route(
            "/debug/synthetic-face.jpg", lambda: FileResponse(ROOT / "tests/data/face.jpg"), methods=["GET"]
        )

    profiles = []
    original_init = AudioStream.__init__

    def instrument(stream, *init_args, **kwargs):
        original_init(stream, *init_args, **kwargs)
        meter = Meter()
        session_meter = SessionMeter(stream.iso.proc.session, meter)
        stream.iso.proc.session = session_meter
        meter.wrap(stream.iso.proc, "process_samples", "sdk")
        meter.wrap(stream.iso, "process", "isolator")
        meter.wrap(stream.vad, "step", "vad")
        meter.wrap(stream.raw_vad, "step", "vad")
        meter.wrap(stream.adjuster, "view", "crosscheck")
        meter.wrap(stream.adjuster, "decide", "crosscheck")
        meter.wrap(stream, "process", "total")
        process = stream.process
        record = {
            "audio_seconds": 0.0,
            "blocks": 0,
            "meter": meter,
            "session": session_meter,
            "durations": collections.deque(maxlen=250),
            "scheduler": collections.deque(maxlen=120),
            "lock": threading.Lock(),
            "worker_ids": set(),
            "previous_worker": None,
            "worker_switches": 0,
        }
        profiles.append(record)

        def measured_block(pcm, *process_args, **process_kwargs):
            if record["blocks"] % 50 == 25:
                record["scheduler"].append(scheduler_state())
            meter.enabled = record["blocks"] >= 25
            if meter.enabled:
                worker = threading.get_ident()
                record["worker_ids"].add(worker)
                if record["previous_worker"] is not None and record["previous_worker"] != worker:
                    record["worker_switches"] += 1
                record["previous_worker"] = worker
            record["blocks"] += 1
            if meter.enabled:
                record["audio_seconds"] += len(pcm) / (2 * SR)
                record["durations"].append(len(pcm) / (2 * SR))
            return process(pcm, *process_args, **process_kwargs)

        def measured(pcm, *process_args, **process_kwargs):
            with record["lock"]:
                return measured_block(pcm, *process_args, **process_kwargs)

        stream.process = measured

    def snapshot():
        results = []
        for record in tuple(profiles):
            with record["lock"]:
                audio_s, blocks = record["audio_seconds"], record["blocks"]
                worker_count, worker_switches = len(record["worker_ids"]), record["worker_switches"]
                scheduler = tuple(record["scheduler"])
                trace = tuple(record["session"].trace)
                durations = tuple(record["durations"])
                samples = {label: tuple(rows) for label, rows in record["meter"].samples.items()}
                values = {label: tuple(rows) for label, rows in record["meter"].values.items()}
            if not audio_s:
                continue
            results.append(
                {
                    "audio_seconds": audio_s,
                    "blocks_with_warmup": blocks,
                    "worker_threads_used": worker_count,
                    "worker_switches": worker_switches,
                    "scheduler_last_120_samples": scheduler,
                    "ort_gap_groups_last_2000_calls": record["session"].summary(trace),
                    "recent_n_blocks": len(durations),
                    "recent_250_blocks": {
                        label: {
                            "wall_rtf": sum(w for w, _ in rows[-250:]) / sum(durations),
                            "thread_cpu_rtf": sum(c for _, c in rows[-250:]) / sum(durations),
                            "p95_ms": float(np.percentile([w for w, _ in rows[-250:]], 95) * 1000),
                        }
                        for label, rows in samples.items()
                        if label in ("total", "sdk", "isolator")
                    },
                    "stages_nested": {
                        label: {"calls": n, "wall_rtf": wall / audio_s, "thread_cpu_rtf": cpu / audio_s}
                        for label, (n, wall, cpu) in values.items()
                    },
                }
            )
        output = {
            "t": time.time(),
            "ort": ort.__version__,
            "mode": "websocket_server",
            "method": "nested timings include instrumentation overhead; CPU time is caller thread only",
            "results": results,
        }
        if args.output:
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(output, indent=2) + "\n")
            temporary.replace(args.output)
        return output

    async def run():
        async def sample():
            while True:
                await asyncio.sleep(5)
                await asyncio.to_thread(snapshot)

        sampler = asyncio.create_task(sample())
        try:
            await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.server_port, access_log=False)).serve()
        finally:
            sampler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sampler

    AudioStream.__init__ = instrument
    try:
        asyncio.run(run())
    finally:
        AudioStream.__init__ = original_init
        print(json.dumps(snapshot(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--signal", nargs="+", choices=("noise", "quiet", "silence"), default=["noise", "quiet"])
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=("main_burst", "worker_burst", "main_paced", "worker_paced"),
        default=["main_burst", "worker_burst", "main_paced", "worker_paced"],
    )
    parser.add_argument("--qos", choices=("default", "user-initiated", "utility"), default="default")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-port", type=int, help="profile a separate localhost server until Ctrl-C")
    parser.add_argument("--disable-kleidiai", action="store_true", help="diagnostic ORT kernel control")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--prestart-workers", action="store_true", help="warm all executor threads before replay")
    parser.add_argument("--browser-synthetic", action="store_true", help="serve a synthetic MediaStream test page")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.disable_kleidiai:
        import onnxruntime as ort

        import presencekit.audio as audio_module

        def controlled_session(path, threads=1):
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.add_session_config_entry("mlas.disable_kleidiai", "1")
            return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

        audio_module._shared_session = controlled_session
    if args.server_port:
        if args.qos != "default":
            parser.error("server profiling supports default QoS only")
        serve(args)
    else:
        asyncio.run(main(args))

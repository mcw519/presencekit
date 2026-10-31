"""Benchmark optimizations off vs on over a synthetic 60 s scenario.

PRIVACY: frames are synthesized in memory from tests/data/face.jpg and JPEG-encoded/decoded in memory.
Nothing is written to disk.

Scenario (simulated time): 0-15 s person static, 15-30 s person shifting, 30-45 s empty ROI,
45-60 s person back (static). Frames get light sensor noise.

    uv run python scripts/benchmark.py              # optimizations off vs on (60 s scenario)
    uv run python scripts/benchmark.py --latency    # per-stage latency p50/p95/max per config x input
    uv run python scripts/benchmark.py --ws         # end-to-end over a localhost WebSocket (starts and
                                                    # stops a temporary server on 127.0.0.1)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from presence_demo.optimizer import FrameProcessor, recommend_fps
from presence_demo.tracker import Config, PresenceTracker
from presence_demo.vision import VisionPipeline, decode_jpeg, load_pose_detector

FACE = Path(__file__).resolve().parent.parent / "tests" / "data" / "face.jpg"
DURATION = 60.0


class SimClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_frame(t: float, face: np.ndarray, bg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = bg.copy()
    if not (30 <= t < 45):
        dx = 0
        if 15 <= t < 30:  # walking around a bit
            dx = int(40 * np.sin((t - 15) * 0.8))
        x = 200 + dx
        img[60:420, x : x + 240] = face
    noise = rng.normal(0, 2.0, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def run(pipe: VisionPipeline, cfg: Config, fixed_fps: float, label: str) -> dict:
    clk = SimClock()
    proc = FrameProcessor(pipe, clock=clk, pose_loader=load_pose_detector)
    tr = PresenceTracker(cfg, clock=clk)
    face = cv2.resize(cv2.imread(str(FACE)), (240, 360))
    bg = np.full((480, 640, 3), 90, np.uint8)
    bg[:, :, 0] = np.linspace(60, 140, 640, dtype=np.uint8)
    rng = np.random.default_rng(0)
    lat, sig, changed_at, states = [], (), -10.0, []
    cpu, wall0 = 0.0, time.perf_counter()
    while clk.t < DURATION:
        jpg = cv2.imencode(".jpg", make_frame(clk.t, face, bg, rng), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        t0, c0 = time.perf_counter(), time.process_time()
        img, _ = decode_jpeg(jpg)
        stable = tr.state.value not in ("MAYBE_AWAY", "AWAY")
        dets, _ = proc.process(img, cfg, stable)
        snap = tr.update(dets, speaking=(clk.t < 3 or 45 <= clk.t < 47))
        lat.append((time.perf_counter() - t0) * 1000)
        cpu += time.process_time() - c0  # CPU of decode + vision + tracker only (not frame synthesis)
        states.append(snap["state"])
        s = tuple(sorted((x["id"], x["participant"], x["visible"]) for x in snap["tracks"]))
        if s != sig:
            sig, changed_at = s, clk.t
        rec = recommend_fps(snap, cfg, clk.t - changed_at < 3.0)
        clk.t += 1.0 / (rec if rec > 0 else fixed_fps)
    a = np.array(lat)
    return {
        "label": label,
        "frames": len(lat),
        "avg": a.mean(),
        "p95": np.percentile(a, 95),
        "sum_ms": a.sum(),
        "cpu_s": cpu,
        "wall_s": time.perf_counter() - wall0,
        "detect": proc.counts["detect_calls"],
        "embed": proc.counts["embed_computed"],
        "reused": proc.counts["embed_reused"],
        "skipped": proc.counts["skipped"],
        "pose": proc.counts["pose_calls"],
        "away_seen": "AWAY" in states,
    }


OFF = dict(motion_gate=False, embed_throttle=False, adaptive_fps=False, roi_crop=False, max_width=1920)


# ---------------------------------------------------------------- per-stage latency
LAT_CONFIGS = {
    "legacy+rules+rules": dict(tracker_mode="legacy", presence_model="rules", intent_model="rules"),
    "kalman+hmm+score": dict(tracker_mode="kalman", presence_model="hmm", intent_model="score"),
}
STAGES = ("decode", "motion", "detect", "embed", "pose", "appearance", "tracker", "presence", "intent", "total")


def synth(kind: str, w: int, h: int, i: int, face: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """In-memory frame: 'face' (person moving slightly), 'static' (identical every time), 'empty'."""
    img = np.full((h, w, 3), 90, np.uint8)
    img[:, :, 0] = np.linspace(60, 140, w, dtype=np.uint8)
    if kind in ("face", "static"):
        fh = int(h * 0.75)
        fw = int(fh * 240 / 360)
        f = cv2.resize(face, (fw, fh))
        dx = 0 if kind == "static" else int(0.06 * w * np.sin(i * 0.9))
        x, y = (w - fw) // 2 + dx, (h - fh) // 2
        img[y : y + fh, x : x + fw] = f
    if kind != "static":
        img = np.clip(img.astype(np.int16) + rng.normal(0, 2.0, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    return img


def run_latency(pipe, cfg, kind, w, h, frames, face):
    clk = SimClock()
    proc = FrameProcessor(pipe, clock=clk, pose_loader=load_pose_detector)
    tr = PresenceTracker(cfg, clock=clk)
    rng = np.random.default_rng(0)
    rows, skipped = [], 0
    static_jpg = None
    for i in range(frames + 3):
        if kind == "static" and static_jpg is not None:
            jpg = static_jpg
        else:
            im = synth(kind, w, h, i, face, rng)
            jpg = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            del im
            static_jpg = jpg if kind == "static" else None
        t0 = time.perf_counter()
        img, dec = decode_jpeg(jpg)
        dets, tm = proc.process(img, cfg, tr.state.value not in ("MAYBE_AWAY", "AWAY"))
        tr.update(dets, speaking=False)
        total = (time.perf_counter() - t0) * 1000
        del img
        clk.t += 0.5
        if i < 3:
            continue
        skipped += bool(tm.get("skipped"))
        rows.append({"decode": dec, **{k: tm.get(k, 0.0) for k in STAGES[1:6]}, **tr.timings, "total": total})
    return rows, skipped


def pct(rows, st):
    a = np.array([r.get(st, 0.0) for r in rows])
    return f"{np.percentile(a, 50):.1f} / {np.percentile(a, 95):.1f} / {a.max():.1f}"


def backends_table(frames: int) -> None:
    """Embedding backends side by side (every frame computes an embedding: gate + throttle off)."""
    pipe = VisionPipeline()
    face = cv2.imread(str(FACE))
    print(
        f"\n嵌入後端比較 · kalman+hmm+score · 姿勢關 · 移動閘門與 embedding 節流關閉（每張都算 embedding）"
        f" · {frames} 張 · ms p50 / p95 / max"
    )
    print("| 後端 | 實際使用 | 加速器 | 輸入 | embed | total |")
    print("|---|---|---|---|---|---|")
    for be, prec in (("fp32", "int8"), ("int8", "int8"), ("npu", "int8"), ("npu", "fp16")):
        for iname, w, h in (("640×480", 640, 480), ("1280×720", 1280, 720)):
            cfg = Config(
                distance_enabled=True,
                motion_gate=False,
                embed_throttle=False,
                embed_backend=be,
                npu_precision=prec,
                **LAT_CONFIGS["kalman+hmm+score"],
            )
            rows, _ = run_latency(pipe, cfg, "face", w, h, frames, face)
            st = pipe.embed_status
            label = be if be != "npu" else f"npu ({prec})"
            print(
                f"| {label} | {st['model']} | {st['accelerator']} | {iname} "
                f"| {pct(rows, 'embed')} | {pct(rows, 'total')} |"
            )
        if pipe.embed_status["note"]:
            print(f"|  | 註：{pipe.embed_status['note']} | | | | |")


def embed_micro(n: int = 200) -> None:
    """Model-only latency of each embedding model on an aligned 112x112 crop (no fallback logic)."""
    from presence_demo.embed_backends import OpenCVSFace, OrtSFace, TFLiteSFace
    from presence_demo.vision import MODEL_DIR, SFACE, SFACE_INT8, SFACE_TFLITE

    img = cv2.imread(str(FACE))
    crop = cv2.resize(img, (112, 112))
    models = [("fp32 ONNX · ONNX Runtime", lambda: OrtSFace(MODEL_DIR / SFACE[0]))]
    models.append(("fp32 ONNX · OpenCV DNN", lambda: OpenCVSFace(MODEL_DIR / SFACE[0])))
    if (MODEL_DIR / SFACE_INT8[0]).exists():
        models.append(("int8 ONNX · ONNX Runtime", lambda: OrtSFace(MODEL_DIR / SFACE_INT8[0], "int8")))
        models.append(("int8 ONNX · OpenCV DNN", lambda: OpenCVSFace(MODEL_DIR / SFACE_INT8[0], "int8")))
    for prec in ("int8", "fp16"):
        for th in (4, 1):
            p = MODEL_DIR / SFACE_TFLITE[prec]
            if p.exists():
                models.append(
                    (f"{prec} TFLite · LiteRT CPU {th} 執行緒", lambda p=p, th=th: TFLiteSFace(p, threads=th))
                )
    print(f"\n模型本身（112×112 對齊後的臉，{n} 次，ms）")
    print("| 模型 | p50 | p95 | max |")
    print("|---|---|---|---|")
    for name, mk in models:
        m = mk()
        for _ in range(10):
            m.feature_aligned(crop)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            m.feature_aligned(crop)
            ts.append((time.perf_counter() - t0) * 1000)
        a = np.array(ts)
        print(f"| {name} | {np.percentile(a, 50):.2f} | {np.percentile(a, 95):.2f} | {a.max():.2f} |")


def latency(frames: int) -> None:
    pipe = VisionPipeline()
    face = cv2.imread(str(FACE))
    inputs = [
        ("人臉 640×480（移動）", "face", 640, 480),
        ("人臉 1280×720（移動）", "face", 1280, 720),
        ("空畫面 640×480", "empty", 640, 480),
        ("靜止畫面 640×480（移動閘門）", "static", 640, 480),
    ]
    print(
        f"Pipeline: {pipe.name} · {frames} 張/列（前 3 張暖機不計）· 2 fps 模擬時間 · 距離開啟 · "
        "單位 ms，p50 / p95 / max"
    )
    print("| 設定 | 姿勢 | 輸入 | " + " | ".join(STAGES) + " | 跳過率 |")
    print("|" + "---|" * (len(STAGES) + 4))
    for cname, ckw in LAT_CONFIGS.items():
        for pose in (False, True):
            for iname, kind, w, h in inputs:
                cfg = Config(pose_enabled=pose, distance_enabled=True, **ckw)
                rows, skipped = run_latency(pipe, cfg, kind, w, h, frames, face)
                cells = [pct(rows, st) for st in STAGES]
                print(
                    f"| {cname} | {'開' if pose else '關'} | {iname} | "
                    + " | ".join(cells)
                    + f" | {skipped / len(rows):.0%} |"
                )


# ---------------------------------------------------------------- end-to-end WebSocket
def ws_bench(frames: int, port: int) -> None:
    import asyncio
    import base64
    import json
    import subprocess
    import sys

    import websockets

    srv = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "presence_demo.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    face = cv2.imread(str(FACE))

    async def one(ckw: dict, pose: bool, fps: float) -> list[float]:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=None) as ws:
            cfg = {"pose_enabled": pose, "distance_enabled": True, "adaptive_fps": False, **ckw}
            await ws.send(json.dumps({"type": "config", "config": cfg}))
            await ws.send(json.dumps({"type": "start"}))
            rng = np.random.default_rng(0)
            lat = []
            for i in range(frames + 2):
                im = synth("face", 640, 480, i, face, rng)
                b64 = base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]).decode()
                del im
                t0 = time.perf_counter()
                await ws.send(json.dumps({"type": "frame", "data": "data:image/jpeg;base64," + b64, "speech": False}))
                while True:  # skip ticker/stats messages; a processed frame carries 'detections'
                    m = json.loads(await ws.recv())
                    if m.get("type") == "state" and "detections" in m:
                        break
                dt = time.perf_counter() - t0
                if i >= 2:
                    lat.append(dt * 1000)
                await asyncio.sleep(max(0.0, 1.0 / fps - dt))
            await ws.send(json.dumps({"type": "stop"}))
            return lat

    async def main_ws():
        for _ in range(100):  # wait for the server
            try:
                async with websockets.connect(f"ws://127.0.0.1:{port}/ws"):
                    break
            except OSError:
                await asyncio.sleep(0.2)
        print(
            f"WebSocket 127.0.0.1:{port} · 640×480 JPEG q70 · 送出 → 收到該張的 state · {frames} 張/列（前 2 張不計）"
        )
        print("| 設定 | 姿勢 | fps | p50 | p95 | max |")
        print("|---|---|---|---|---|---|")
        for cname, ckw in LAT_CONFIGS.items():
            for pose in (False, True):
                for fps in (1.0, 2.0):
                    a = np.array(await one(ckw, pose, fps))
                    print(
                        f"| {cname} | {'開' if pose else '關'} | {fps:g} | {np.percentile(a, 50):.1f} | "
                        f"{np.percentile(a, 95):.1f} | {a.max():.1f} |"
                    )

    try:
        asyncio.run(main_ws())
    finally:
        srv.terminate()
        srv.wait(timeout=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0, help="fixed fps when adaptive fps is off")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--latency", action="store_true", help="per-stage latency table")
    ap.add_argument("--ws", action="store_true", help="end-to-end latency over a localhost WebSocket")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--port", type=int, default=8799)
    a = ap.parse_args()
    if a.latency or a.ws:
        if a.latency:
            latency(a.frames)
            backends_table(a.frames)
            embed_micro()
        if a.ws:
            ws_bench(a.frames, a.port)
        return
    cv2.setNumThreads(cv2.getNumThreads())
    pipe = VisionPipeline()
    variants = [
        ("全部關閉", Config(**OFF)),
        ("只開 ROI 裁切+縮小", Config(**{**OFF, "roi_crop": True, "max_width": 320})),
        ("只開移動偵測閘門", Config(**{**OFF, "motion_gate": True})),
        ("只開 embedding 節流", Config(**{**OFF, "embed_throttle": True})),
        ("全部開啟（固定 fps）", Config(adaptive_fps=False)),
        ("全部開啟 + 自適應 fps", Config()),
        ("姿勢（其他最佳化關閉）", Config(**{**OFF, "pose_enabled": True})),
        ("全部開啟 + 姿勢", Config(pose_enabled=True)),
        ("全部開啟 + 姿勢 + 距離", Config(pose_enabled=True, distance_enabled=True)),
        ("全部開啟 + 距離（無姿勢）", Config(distance_enabled=True)),
    ]
    run(pipe, Config(**OFF), a.fps, "warmup")
    rows = []
    for label, cfg in variants:
        rs = [run(pipe, cfg, a.fps, label) for _ in range(a.repeat)]
        best = min(rs, key=lambda r: r["cpu_s"])
        rows.append(best)
    print(f"Pipeline: {pipe.name} · OpenCV {cv2.__version__} · {a.repeat} 次取 CPU 最低 · 模擬 {DURATION:.0f} s")
    print(
        "| 設定 | 處理張數 | 每張 avg (ms) | 每張 p95 (ms) | 總 CPU (s) | Δ CPU | detect 次數 "
        "| embed 計算/重用 | 姿勢次數 | 跳過 | AWAY |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    base = rows[0]["cpu_s"]
    for r in rows:
        print(
            f"| {r['label']} | {r['frames']} | {r['avg']:.1f} | {r['p95']:.1f} | {r['cpu_s']:.2f} | "
            f"{(r['cpu_s'] / base - 1) * 100:+.0f}% | {r['detect']} | "
            f"{r['embed']} / {r['reused']} | {r['pose']} | {r['skipped']} | {'是' if r['away_seen'] else '否'} |"
        )


if __name__ == "__main__":
    main()

"""Compare pose scheduling and JPEG framing with a saved working-tree snapshot.

Run twice with the same environment, once with --source pointing at the old
project root and once without it. Uses the repo's test portrait and synthetic
backgrounds in memory, with real YuNet/SFace/MoveNet inference. No media capture,
files containing media, WebSocket or cloud requests are used.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=ROOT)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    sys.path.insert(0, str(args.source.resolve()))

    import cv2
    import numpy as np

    from presencekit.config import Config
    from presencekit.optimizer import FrameProcessor
    from presencekit.vision import VisionPipeline, load_pose_detector

    if args.frames < 2 or args.repeat < 1:
        ap.error("--frames must be >= 2 and --repeat >= 1")
    cv2.setNumThreads(2)  # identical worker budget in both separate processes
    face = cv2.resize(cv2.imread(str(ROOT / "tests" / "data" / "face.jpg")), (240, 360))
    image = np.full((480, 640, 3), 90, np.uint8)
    image[60:420, 200:440] = face
    blank = np.full_like(image, 90)
    pipe = VisionPipeline(allow_download=False)
    pose = load_pose_detector()
    if pose is None:
        raise RuntimeError("MoveNet is required for this benchmark")

    class Clock:
        t = 100.0

        def __call__(self):
            return self.t

    warm = FrameProcessor(pipe, pose_loader=lambda: pose)
    for _ in range(5):
        warm.process(image, Config(pose_enabled=True, motion_gate=False))
    scenarios = [
        ("stable_face_full_5fps", image, 5, False, True),
        ("stable_face_gated_2fps", image, 2, True, True),
        ("unstable_face_5fps", image, 5, False, False),
        ("no_face_5fps", blank, 5, False, True),
    ]
    results = {}
    for name, frame, fps, gated, stable in scenarios:
        runs = []
        for _ in range(args.repeat):
            clock = Clock()
            proc = FrameProcessor(pipe, clock=clock, pose_loader=lambda: pose)
            cfg = Config(pose_enabled=True, motion_gate=gated, tracker_mode="kalman")
            timings, face_counts = [], []
            start_cpu = time.process_time()
            for i in range(args.frames):
                clock.t = 100 + i / fps
                start = time.perf_counter()
                dets, _ = proc.process(frame, cfg, stable)
                timings.append((time.perf_counter() - start) * 1000)
                face_counts.append(sum(d.face_bbox is not None for d in dets))
            runs.append(
                {
                    "cpu_ms": (time.process_time() - start_cpu) * 1000,
                    "frame_p50_ms": float(np.median(timings)),
                    "frame_p95_ms": float(np.percentile(timings, 95)),
                    "pose_calls": proc.counts["pose_calls"],
                    "pose_reused": proc.counts.get("pose_reused", 0),
                    "face_counts": sorted(set(face_counts)),
                    "frames": args.frames,
                    "simulated_fps": fps,
                }
            )
        results[name] = {
            "median_cpu_ms": float(np.median([r["cpu_ms"] for r in runs])),
            "median_frame_p50_ms": float(np.median([r["frame_p50_ms"] for r in runs])),
            "runs": runs,
        }
        print(name, json.dumps(results[name]), flush=True)

    jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    legacy = json.dumps(
        {
            "type": "frame",
            "speech": False,
            "t": 12345.67,
            "data": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
        },
        separators=(",", ":"),
    )
    binary = b"\x04" + struct.pack("<dB", 12345.67, 0) + jpeg
    times = {}
    for kind in ("legacy", "binary"):
        run_times = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(500):
                if kind == "legacy":
                    msg = json.loads(legacy)
                    decoded = base64.b64decode(msg["data"].split(",", 1)[1])
                else:
                    struct.unpack_from("<dB", binary, 1)
                    decoded = binary[10:]
                assert decoded == jpeg
            run_times.append((time.perf_counter() - start) * 1000 / 500)
        times[kind] = float(np.median(run_times))
    output = {
        "opencv": cv2.__version__,
        "cv_threads": cv2.getNumThreads(),
        "pose_scenarios": results,
        "jpeg_framing": {
            "jpeg_bytes": len(jpeg),
            "legacy_wire_bytes": len(legacy.encode()),
            "binary_wire_bytes": len(binary),
            "unpack_ms_per_frame": times,
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["jpeg_framing"]), flush=True)


if __name__ == "__main__":
    main()

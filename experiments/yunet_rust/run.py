"""Compare YuNet post-processing implementations on one local multiplayer video."""

import argparse
import hashlib
import json
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from oracle import HEAD_WIDTHS, STRIDES, _decode, _opencv_nms, _postprocess

from presencekit.config import Config
from presencekit.core.tracker import PresenceTracker
from presencekit.detectors.face import OrtYuNetDetector
from presencekit.embed_backends import OrtSFace
from presencekit.inference import OrtOptions, default_store
from presencekit.sdk import PresenceEngine
from presencekit.vision import VisionPipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIDEO = ROOT / "docs/media/presence-demo-intro-en.mp4"
FPS = 2
ORT = OrtOptions(providers=("cpu",), intra_op_threads=2)
MODELS = ("yunet", "sface", "movenet")
BACKENDS = ("numpy", "opencv", "rust")


class RustPostprocessor:
    def __init__(self):
        sys.path.insert(0, str(HERE / "native/python"))
        try:
            from presencekit_yunet_rs._native import postprocess_buffer
        finally:
            sys.path.pop(0)
        self.native = postprocess_buffer
        self.fallbacks = 0

    def __call__(self, *args):
        result = self.native(*args)
        if result is None:
            self.fallbacks += 1
            return _postprocess(*args)
        return np.frombuffer(result, dtype=np.float32).reshape(-1, 15)


class ExperimentalDetector(OrtYuNetDetector):
    def __init__(self, path, postprocess):
        super().__init__(path, options=ORT)
        self.postprocess = postprocess

    def detect(self, img):
        h, w = img.shape[:2]
        ph, pw = -(-h // 32) * 32, -(-w // 32) * 32
        tensor = np.zeros((1, 3, ph, pw), np.float32)
        tensor[0, :, :h, :w] = img.transpose(2, 0, 1)
        out = self.model.run({self._input: tensor}, self._outputs)
        rows = self.postprocess(pw, out, self.score_thr, self.nms_thr, self.top_k)
        return [(row, float(row[14])) for row in rows]


def opencv(*args):
    return _postprocess(*args, nms=_opencv_nms)


def _jpeg_inputs(video):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video {video}")
        frames = []
        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, len(frames) * 1000 / FPS)
            ok, img = cap.read()
            if not ok:
                break
            h, w = img.shape[:2]
            if w > 640:
                img = cv2.resize(img, (640, round(h * 640 / w)), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            assert ok
            frames.append(jpeg.tobytes())
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {video}")
    return frames


def _replay(frames, backend, observer=None, *, trace=False):
    paths = {name: default_store().local(name) for name in MODELS}
    if not all(paths.values()):
        raise RuntimeError(f"missing ONNX models: {[name for name, path in paths.items() if not path]}")
    native = RustPostprocessor() if backend == "rust" else None
    if backend == "numpy":
        detector = OrtYuNetDetector(paths["yunet"], options=ORT)
    else:
        detector = ExperimentalDetector(paths["yunet"], native if native is not None else opencv)
    embedder = OrtSFace(paths["sface"], options=ORT)
    pipe = VisionPipeline(detector, embedder, allow_download=False, options=ORT)
    cfg = Config(pose_enabled=True, tracker_mode="kalman", presence_model="hmm", motion_gate=False, roi_crop=False)
    now = 0.0
    engine = PresenceEngine(cfg, pipeline=pipe, ort=ORT, allow_download=False, clock=lambda: now)
    for i, jpeg in enumerate(frames[::10][:10]):
        now = i / FPS
        engine.process_jpeg(jpeg, speaking=False, now=now)
    engine.reset()
    models = engine.info()["models"]
    assert {m["model"] for m in models} == {p.name for p in paths.values()}
    assert all(m["providers"] == ["cpu"] and m["threads"] == 2 and m["intra_op_allow_spinning"] is None for m in models)
    tracker = engine.core.tracker
    assert type(tracker) is PresenceTracker
    pairs = []

    if trace:
        associate = tracker._associate_kalman

        def traced(dets, tracks, now):
            predicted, matches = associate(dets, tracks, now)
            pairs[:] = [(i, tracks[j].id) for i, j, _ in matches]
            return predicted, matches

        tracker._associate_kalman = traced

    elapsed = []
    snapshots = []
    start_fallbacks = native.fallbacks if native is not None else 0
    start_cpu = time.process_time()
    for i, jpeg in enumerate(frames):
        now = i / FPS
        with observer(detector) if observer else nullcontext():
            start = time.perf_counter_ns()
            step = engine.process_jpeg(jpeg, speaking=i % 20 < 8, now=now)
            elapsed.append(time.perf_counter_ns() - start)
        assert step is not None
        if trace:
            snap = step.snapshot
            tracks = [(t["id"], t["visible_kind"], t["participant"]) for t in snap["tracks"]]
            log = [(event["t"], event["event"]) for event in snap["log"]]
            snapshots.append((snap["session"], snap["state"], tracks, list(pairs), log))
    cpu = time.process_time() - start_cpu
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats = {
        "frames": len(frames),
        "p95_ms": float(np.percentile(elapsed, 95)) / 1e6,
        "cpu_seconds": cpu,
        "peak_rss_mib": rss / (1 << 20 if sys.platform == "darwin" else 1024),
        "fallbacks": native.fallbacks - start_fallbacks if native is not None else 0,
        "detect_calls": engine.frames.counts["detect_calls"],
        "embedding_calls": engine.frames.counts["embed_computed"],
        "pose_calls": engine.frames.counts["pose_calls"],
    }
    return stats, snapshots


def _compare(expected, actual):
    # Row-by-row comparison checks ordering and landmarks as well as the box.
    np.testing.assert_array_equal(actual[:, 14], expected[:, 14])
    np.testing.assert_allclose(actual[:, :14], expected[:, :14], rtol=0, atol=1e-3)
    return float(np.abs(actual - expected).max(initial=0))


def _heads(candidates):
    pw = 64
    heads = [
        np.zeros((1, (pw // STRIDES[i % len(STRIDES)]) ** 2, lane), np.float32) for i, lane in enumerate(HEAD_WIDTHS)
    ]
    for stride_index, idx, x, y, w, h, score, marker in candidates:
        step = STRIDES[stride_index]
        row, col = divmod(idx, pw // step)
        heads[stride_index][0, idx, 0] = score * score
        heads[stride_index + 3][0, idx, 0] = 1.0
        heads[stride_index + 6][0, idx] = (
            (x + w / 2) / step - col,
            (y + h / 2) / step - row,
            np.log(w / step),
            np.log(h / step),
        )
        heads[stride_index + 9][0, idx] = [marker / step - col, 0] * 5
    return pw, heads


def _boundary_cases():
    low = float(np.nextafter(np.float32(0.5), np.float32(0)))
    high = float(np.nextafter(np.float32(0.5), np.float32(1)))
    box1 = (0, 0, 0.25, 0.25, 12.625, 12.625)
    box2 = (0, 1, 4.25, 0.25, 12.625, 12.625)  # truncated IoU exactly 0.5
    box3 = (1, 0, 30.25, 0.25, 12.625, 12.625)
    boundary_box = (0, 0, -0.25, -0.25, 12.0, 12.0)
    return [
        ("empty", [], 0.5, 0, [], False),
        ("score_below", [(*box1, low, 1)], 0.5, 0, [], False),
        ("score_equal_single", [(*box1, 0.5, 1)], 0.5, 0, [1], False),
        ("score_above", [(*box1, high, 1)], 0.5, 0, [1], False),
        ("score_equal_in_nms", [(*box1, 0.5, 1), (*box3, high, 3)], 0.5, 0, [3], False),
        ("iou_below_threshold", [(*box1, 0.9, 1), (*box2, 0.8, 2)], high, 0, [1, 2], False),
        ("iou_equal_threshold", [(*box1, 0.9, 1), (*box2, 0.8, 2)], 0.5, 0, [1, 2], False),
        ("iou_above_threshold", [(*box1, 0.9, 1), (*box2, 0.8, 2)], low, 0, [1], False),
        ("stable_tie_across_strides", [(*box3, 0.9, 3), (*box1, 0.9, 1)], 0.5, 1, [1], False),
        ("top_k_precedes_suppression", [(*box1, 0.9, 1), (*box2, 0.9, 2), (*box3, 0.8, 3)], 0.3, 2, [1], False),
        ("score_clips_before_sqrt", [(*box1, 2.0, 1)], 0.5, 0, [1], False),
        ("integer_truncation_fallback", [(*boundary_box, 0.9, 1), (*box3, 0.8, 3)], 0.5, 0, [1, 3], True),
    ]


def _write_case(stream, args):
    pw, heads, score, nms, top_k = args
    encoded = [head.reshape(-1).tolist() for head in heads]
    stream.write(json.dumps({"fn": "postprocess", "args": [pw, encoded, score, nms, top_k]}) + "\n")


def verify(frames, case_file):
    rust = RustPostprocessor()
    max_error = 0.0
    all_faces = Counter()
    candidate_counts = []
    captured_batch = []
    cases = _boundary_cases()
    with case_file.open("w", encoding="utf-8") as stream:
        for label, candidates, nms, top_k, markers, fallback in cases:
            args = (*_heads(candidates), 0.5, nms, top_k)
            expected = _postprocess(*args)
            assert expected[:, 4].tolist() == markers, label
            before = rust.fallbacks
            _compare(expected, opencv(*args))
            max_error = max(max_error, _compare(expected, rust(*args)))
            assert rust.fallbacks - before == fallback, label
            _write_case(stream, args)

        fallback_after_boundaries = rust.fallbacks

        @contextmanager
        def observe(detector):
            run = detector.model.run
            detect = detector.detect
            captured = None

            def capture(feeds, outputs):
                nonlocal captured
                result = run(feeds, outputs)
                pw = next(iter(feeds.values())).shape[-1]
                captured = (pw, result, detector.score_thr, detector.nms_thr, detector.top_k)
                return result

            def compared(img):
                nonlocal max_error
                result = detect(img)
                expected = np.asarray([row for row, _ in result], np.float32).reshape(-1, 15)
                assert captured is not None
                np.testing.assert_array_equal(expected, _postprocess(*captured))
                max_error = max(max_error, _compare(expected, rust(*captured)), _compare(expected, opencv(*captured)))
                if len(candidate_counts) % 10 == 0:
                    _write_case(stream, captured)
                all_faces[len(expected)] += 1
                candidate_counts.append(len(_decode(*captured[:3])))
                captured_batch.append(captured)
                return result

            with patch.object(detector.model, "run", capture), patch.object(detector, "detect", compared):
                yield

        baseline, snapshots = _replay(frames, "numpy", observe, trace=True)
    assert baseline["detect_calls"] == len(candidate_counts) == len(frames)
    for mode in BACKENDS[1:]:
        stats, actual = _replay(frames, mode, trace=True)
        assert actual == snapshots, f"{mode} differed in tracking IDs, pairings or states"
        assert _work(stats) == _work(baseline)
    print(
        json.dumps(
            {
                "boundary_cases": len(cases),
                "onnx_outputs": len(candidate_counts),
                "replay_cases": len(cases) + (len(candidate_counts) + 9) // 10,
                "faces_per_frame": dict(sorted(all_faces.items())),
                "candidates_max": max(candidate_counts),
                "max_abs_error": max_error,
                "rust_fallbacks": rust.fallbacks - fallback_after_boundaries,
                "tracker_frames": len(snapshots),
                "postprocess_median_p95_ms": _bench_postprocess(captured_batch),
            },
            indent=2,
        )
    )


def _work(stats):
    return tuple(stats[key] for key in ("frames", "detect_calls", "embedding_calls", "pose_calls"))


def _bench_postprocess(batch):
    methods = {"numpy": _postprocess, "opencv": opencv, "rust": RustPostprocessor()}
    runs = {mode: [] for mode in BACKENDS}
    for repeat in range(7):
        offset = repeat % len(BACKENDS)
        for mode in BACKENDS[offset:] + BACKENDS[:offset]:
            process = methods[mode]
            elapsed = []
            for args in batch * 10:
                start = time.perf_counter_ns()
                process(*args)
                elapsed.append(time.perf_counter_ns() - start)
            runs[mode].append(float(np.percentile(elapsed, 95)) / 1e6)
    return {mode: statistics.median(values) for mode, values in runs.items()}


def bench(video, frames, repeats):
    # All workers receive the exact same JPEG bytes. The source video decoder
    # remains outside their measured process and peak RSS.
    payload = b"".join(len(jpeg).to_bytes(4, "little") + jpeg for jpeg in frames)
    modes = {mode: [] for mode in BACKENDS}
    for i in range(repeats):
        offset = i % len(BACKENDS)
        for mode in BACKENDS[offset:] + BACKENDS[:offset]:
            cmd = [sys.executable, str(__file__), "worker", mode]
            modes[mode].append(json.loads(subprocess.check_output(cmd, cwd=ROOT, input=payload)))
    expected = _work(modes["numpy"][0])
    assert all(_work(row) == expected for runs in modes.values() for row in runs)
    medians = {
        mode: {key: statistics.median(row[key] for row in rows) for key in rows[0]} for mode, rows in modes.items()
    }
    with video.open("rb") as source:
        sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    print(json.dumps({"video_sha256": sha256, "runs": modes, "medians": medians}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=VIDEO)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify").add_argument("cases", type=Path)
    commands.add_parser("bench").add_argument("--repeats", type=int, default=7)
    commands.add_parser("worker").add_argument("backend", choices=BACKENDS)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.command == "bench":
        if args.repeats < 1:
            parser.error("--repeats must be positive")
        bench(args.video, _jpeg_inputs(args.video), args.repeats)
    elif args.command == "verify":
        verify(_jpeg_inputs(args.video), args.cases)
    else:
        source = sys.stdin.buffer
        frames = [source.read(int.from_bytes(header, "little")) for header in iter(lambda: source.read(4), b"")]
        print(json.dumps(_replay(frames, args.backend)[0]))

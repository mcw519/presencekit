"""Verify the tracker and benchmark the optional PyO3 association kernel."""

import argparse
import copy
import itertools
import json
import resource
import runpy
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from oracle import _GATE_KEYS, _pack_observation, _pack_state, _unpack_state

from presencekit.config import Config
from presencekit.core.tracker import PresenceTracker
from presencekit.core.tracks import Detection, Track
from presencekit.inference import OrtOptions
from presencekit.kalman import KalmanBox
from presencekit.sdk import PresenceEngine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE_PYTHON = HERE / "native/python"
VISION_OPTIONS = OrtOptions(providers=("cpu",), intra_op_threads=2)
OPTIONS = {"pose_enabled": True, "tracker_mode": "kalman", "presence_model": "hmm", "intent_model": "score"}


class NativeFilter:
    """Keep the native output row as the filter state for the next update."""

    def __init__(self, row):
        self.row = row

    @property
    def velocity(self):
        return self.row[4], self.row[5]


def _arguments(cfg, tracks, detections, now, pack=_pack_observation):
    states = []
    for track in tracks:
        filt = track.kf
        if isinstance(filt, NativeFilter):
            states.append(filt.row[:25])
        else:
            states.append(_pack_state(filt if filt is not None else KalmanBox(track.bbox, track.last_seen)))
    return (
        now,
        [getattr(cfg, key) for key in _GATE_KEYS],
        states,
        [pack(track) for track in tracks],
        [pack(det) for det in detections],
    )


def _buffer_observation(obs):
    return (
        obs.bbox,
        obs.body.bbox if obs.body is not None else None,
        None if obs.embedding is None else np.asarray(obs.embedding, dtype=np.float32).tobytes(),
        None if obs.appearance is None else np.asarray(obs.appearance, dtype=np.float64).tobytes(),
    )


class BoxedRustTracker(PresenceTracker):
    binding = "associate"
    pack = staticmethod(_pack_observation)

    def __init__(self, config, clock):
        super().__init__(config, clock)
        sys.path.insert(0, str(NATIVE_PYTHON))
        try:
            from presencekit_kalman_rs import _native
        finally:
            sys.path.pop(0)
        self.associate = getattr(_native, self.binding)
        self.reference_frames = 0

    def _associate_kalman(self, dets, tracks, now):
        rows = self.associate(*_arguments(self.cfg, tracks, dets, now, self.pack))
        if rows is None:
            self.reference_frames += 1
            for track in tracks:
                if isinstance(track.kf, NativeFilter):
                    track.kf = _unpack_state(track.kf.row)
            return super()._associate_kalman(dets, tracks, now)
        predicted = []
        for track, row in zip(tracks, rows[: len(tracks)], strict=True):
            track.kf = NativeFilter(row)
            predicted.append(tuple(row[25:]))
        matches = [(int(det), int(track), score) for det, track, score in rows[len(tracks) :]]
        return predicted, matches


class RustTracker(BoxedRustTracker):
    binding = "associate_buffer"
    pack = staticmethod(_buffer_observation)


BACKENDS = {"python": PresenceTracker, "rust-boxed": BoxedRustTracker, "rust": RustTracker}


def _trace(tracker, pairs, write_case=None):
    call = tracker._associate_kalman
    frames = itertools.count()

    def traced(dets, tracks, now):
        if write_case is not None and next(frames) % 8 == 0:
            write_case(json.dumps({"fn": "associate", "args": _arguments(tracker.cfg, tracks, dets, now)}) + "\n")
        predicted, matches = call(dets, tracks, now)
        pairs[:] = [(det, tracks[track].id) for det, track, _ in matches]
        return predicted, matches

    tracker._associate_kalman = traced


def _signature(snap, pairs):
    tracks = [(t["id"], t["visible_kind"], t["participant"]) for t in snap["tracks"]]
    return snap["session"], snap["state"], tracks, list(pairs), snap["log"]


def _write_edges(stream):
    cfg = Config(**OPTIONS)
    gates = [getattr(cfg, key) for key in _GATE_KEYS]
    box = [0.25, 0.15, 0.5, 0.65]
    state = _pack_state(KalmanBox(box, 0))
    row = [box, [], [1.0, 0.0], []]
    opposite = [box, [], [-1.0, 0.0], []]
    examples = [
        [0, gates, [], [], []],
        [0, gates, [state, state], [row, row], [row, row]],
        [0, gates, [state, state], [row, row], [row, row, row]],
        [0, gates, [state], [row], [opposite]],
        [0, gates, [state], [row], []],
        [0, gates[:-1], [], [], []],
        [0, gates, [state], [row], [[box, [], []]]],
    ]
    for args in examples:
        stream.write(json.dumps({"fn": "associate", "args": args}) + "\n")


def _synthetic_inputs(source, scenario, seed):
    rng = np.random.default_rng(seed)
    for person in scenario.persons:
        source["make_identity"](rng, person)
    inputs = []
    t = 0.0
    while t < scenario.duration:
        dets, _ = source["detections"](scenario, t, rng)
        inputs.append((dets, scenario.speech(t), t))
        t += scenario.dt(t)
    return inputs


def _verify_boundaries():
    cfg = Config(**OPTIONS)
    anchor = (0.12, 0.15, 0.32, 0.50)
    far = (0.67, 0.15, 0.87, 0.50)
    unit = np.array([1, 0], dtype=np.float32)
    checked = 0

    def compare(dets, tracks, expected):
        nonlocal checked
        python = PresenceTracker(cfg, lambda: 0.0)
        rust = RustTracker(cfg, lambda: 0.0)
        for track in tracks:
            # Exercise reconstruction of existing native filters before the fallback.
            track.kf = NativeFilter(_pack_state(KalmanBox(track.bbox, 0.0)))
        baseline_tracks = copy.deepcopy(tracks)
        for track in baseline_tracks:
            track.kf = _unpack_state(track.kf.row)
        python_predicted, python_pairs = python._associate_kalman(dets, baseline_tracks, 0.1)
        rust_predicted, rust_pairs = rust._associate_kalman(dets, tracks, 0.1)
        assert [(i, j) for i, j, _ in python_pairs] == expected
        assert rust_pairs == python_pairs and rust.reference_frames == 1, (python_pairs, rust_pairs)
        assert rust_predicted == python_predicted
        assert [_pack_state(track.kf) for track in tracks] == [_pack_state(track.kf) for track in baseline_tracks]
        checked += 1

    for threshold, bbox in ((cfg.cos_weak, anchor), (cfg.match_cos, far)):
        for greater in (False, True):
            score = np.nextafter(np.float32(threshold), np.float32(float("inf") if greater else -float("inf")))
            embedding = np.array([score, np.sqrt(1 - float(score) ** 2)], dtype=np.float32)
            compare(
                [Detection(bbox, 1.0, embedding)],
                [Track(1, anchor, unit, 1.0, 0.0, 0.0, 0.0)],
                [(0, 0)] if greater else [],
            )
    compare(
        [Detection(anchor, 1.0, unit) for _ in range(3)],
        [Track(i, anchor, unit, 1.0, 0.0, 0.0, 0.0) for i in (1, 2)],
        [(0, 0), (1, 1)],
    )
    return checked


def verify(cases_path):
    boundary_cases = _verify_boundaries()
    source = runpy.run_path(str(ROOT / "scripts/eval_tracking.py"))
    frames = pairs_count = transitions = fallback_frames = 0
    with cases_path.open("w", encoding="utf-8") as stream:
        _write_edges(stream)
        for scenario in source["scenarios"]():
            for seed in range(10):
                python = PresenceTracker(Config(**OPTIONS, **scenario.cfg), lambda: 0.0)
                rust = RustTracker(Config(**OPTIONS, **scenario.cfg), lambda: 0.0)
                python_pairs, rust_pairs = [], []
                _trace(python, python_pairs, stream.write if seed < 2 else None)
                _trace(rust, rust_pairs)
                for dets, speaking, t in _synthetic_inputs(source, scenario, seed):
                    expected = python.update(copy.deepcopy(dets), speaking=speaking, now=t)
                    actual = rust.update(dets, speaking=speaking, now=t)
                    assert _signature(actual, rust_pairs) == _signature(expected, python_pairs), (
                        scenario.name,
                        seed,
                        t,
                        _signature(expected, python_pairs),
                        _signature(actual, rust_pairs),
                    )
                    frames += 1
                    pairs_count += len(rust_pairs)
                    transitions += sum(e.get("event") == "transition" and e["t"] == t for e in actual["log"])
                    dt = scenario.dt(t)
                    for instant in np.arange(t + 0.5, t + dt - 1e-9, 0.5):
                        for tracker in (python, rust):
                            tracker.speaking = scenario.speech(instant)
                        assert _signature(rust.tick(instant), []) == _signature(python.tick(instant), [])
                fallback_frames += rust.reference_frames
    expected = _vision("python", trace=True)
    actual = _vision("rust", trace=True)
    assert actual == expected, "fixed ONNX replay differs"
    print(
        json.dumps(
            {
                "tracker_frames": frames,
                "matches": pairs_count,
                "frame_transitions": int(transitions),
                "boundary_cases": boundary_cases,
                "synthetic_reference_frames": fallback_frames,
                "onnx_frames": len(actual),
                "onnx_matches": sum(len(s[3]) for s in actual),
                "onnx_max_tracks": max(len(s[2]) for s in actual),
            },
            indent=2,
        )
    )


def _jpeg_inputs():
    portrait = cv2.imread(str(ROOT / "tests/data/face.jpg"))
    face = cv2.resize(portrait, (128, 198))
    frames = []
    # Real ONNX input for timings. The synthetic verification also supplies distinct identities.
    for people in (1, 2, 3):
        for frame in range(60):
            image = np.full((480, 640, 3), 85, np.uint8)
            for i in range(people if frame < 35 or frame >= 52 else people - 1):
                x = 256 if people == 1 else (80, 256, 432)[i]
                x += int(16 * np.sin(frame * 0.6 + i)) if 14 <= frame < 30 else 0
                image[125:323, x : x + 128] = face
            encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
            assert encoded
            frames.append(jpeg.tobytes())
    return frames


def _vision(backend, trace=False):
    cv2.setNumThreads(2)
    frames = _jpeg_inputs()
    cfg = Config(**{**OPTIONS, "intent_model": "rules"}, leave_distance=5.0)
    now = [0.0]

    def current_time():
        return now[0]

    engine = PresenceEngine(cfg, ort=VISION_OPTIONS, allow_download=False, clock=current_time)
    tracker_type = BACKENDS[backend]
    for i in range(0, len(frames), 20):
        now[0] = i / 2
        engine.process_jpeg(frames[i], speaking=True, now=now[0])
    models = engine.info()["models"]
    assert {m["model"] for m in models} == {
        "yunet_s_dynamic.onnx",
        "face_recognition_sface_2021dec.onnx",
        "movenet_multipose_lightning.onnx",
    }
    assert all(m["providers"] == ["cpu"] and m["threads"] == 2 for m in models)
    assert all(m["intra_op_allow_spinning"] is None for m in models)
    now[0] = 0.0
    engine.reset()
    engine.core.tracker = tracker_type(cfg, current_time)
    pairs, signatures = [], []
    if trace:
        _trace(engine.core.tracker, pairs)
    elapsed = []
    cpu_start = time.process_time()
    for i, jpeg in enumerate(frames):
        now[0] = i / 2
        start = time.perf_counter_ns()
        step = engine.process_jpeg(jpeg, speaking=i % 60 < 6 or i % 60 >= 52, now=now[0])
        elapsed.append(time.perf_counter_ns() - start)
        if trace:
            signatures.append(_signature(step.snapshot, pairs))
    cpu = time.process_time() - cpu_start
    if trace:
        return signatures
    return _measurements(elapsed, cpu, getattr(engine.core.tracker, "reference_frames", 0))


def _measurements(elapsed, cpu, reference_frames):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "frames": len(elapsed),
        "p95_ms": float(np.percentile(elapsed, 95)) / 1e6,
        "cpu_seconds": cpu,
        "reference_frames": reference_frames,
        "peak_rss_mib": rss / (1 << 20 if sys.platform == "darwin" else 1024),
    }


def _tracking(backend):
    source = runpy.run_path(str(ROOT / "scripts/eval_tracking.py"))
    tracker_type = BACKENDS[backend]
    elapsed = []
    cpu, reference_frames = 0.0, 0
    for scenario in source["scenarios"]():
        for seed in range(10):
            inputs = _synthetic_inputs(source, scenario, seed)
            tracker = tracker_type(Config(**OPTIONS, **scenario.cfg), lambda: 0.0)
            start_cpu = time.process_time()
            for dets, speaking, now in inputs:
                start = time.perf_counter_ns()
                tracker.update(dets, speaking, now)
                elapsed.append(time.perf_counter_ns() - start)
            cpu += time.process_time() - start_cpu
            reference_frames += getattr(tracker, "reference_frames", 0)
    return _measurements(elapsed, cpu, reference_frames)


def bench(repeats, workload):
    runs = {name: [] for name in BACKENDS}
    for i in range(repeats):
        order = list(BACKENDS)
        offset = i % len(order)
        for backend in order[offset:] + order[:offset]:
            command = [sys.executable, str(__file__), "worker", backend, workload]
            row = json.loads(subprocess.check_output(command, cwd=ROOT))
            runs[backend].append(row)
    summary = {
        backend: {key: statistics.median(r[key] for r in rows) for key in rows[0]} for backend, rows in runs.items()
    }
    print(json.dumps({"runs": runs, "medians": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify").add_argument("cases", type=Path)
    benchmark = commands.add_parser("bench")
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--workload", choices=("vision", "tracking"), default="vision")
    worker = commands.add_parser("worker")
    worker.add_argument("backend", choices=BACKENDS)
    worker.add_argument("workload", choices=("vision", "tracking"))
    args = parser.parse_args()
    if args.command == "verify":
        verify(args.cases)
    elif args.command == "bench":
        if args.repeats < 1:
            parser.error("--repeats must be at least 1")
        bench(args.repeats, args.workload)
    else:
        measure = _vision if args.workload == "vision" else _tracking
        print(json.dumps(measure(args.backend)))

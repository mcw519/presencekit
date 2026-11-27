"""Offline PureSound feedback A/B on a PresenceKit recording bundle."""

import argparse
import csv
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from presencekit.audio import HOP, SR, CleanVAD, Isolator  # noqa: E402


class FeedbackSession:
    """Return pass B's spectrum and pass A's complete temporal state."""

    def __init__(self, session):
        self.session = session

    def run(self, names, inputs):
        first = self.session.run(names, inputs)
        trial = inputs.copy()
        for name in ("h_0", "c_0", "h_1", "c_1"):
            trial[name] = first[names.index("next_" + name)]
        first[0] = self.session.run([names[0]], trial)[0]
        return first


class SpectrumCascade:
    """A second instance of the same graph consumes the first instance's spectrum."""

    def __init__(self, proc):
        self.session = proc.session
        self.state = {k: v.copy() for k, v in proc.state.items()}

    def run(self, names, inputs):
        first = self.session.run(names, inputs)
        second = self.session.run(names, {"noisy_frame": first[0], **self.state})
        self.state = {k: second[names.index("next_" + k)] for k in self.state}
        first[0] = second[0]
        return first


def read(path):
    with wave.open(str(path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2)
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


def save(path, y):
    with wave.open(str(path), "wb") as w:
        w.setparams((1, 2, SR, 0, "NONE", "not compressed"))
        w.writeframes(np.rint(y * 32768).clip(-32768, 32767).astype("<i2").tobytes())


def replay(x, cfg, mode):
    iso = Isolator(onset_guard=cfg["iso_onset_guard"], dry_blend=cfg["iso_dry_blend"])
    proc = iso.proc
    if mode == "spectrum":
        if proc.onset_guard is not None:
            raise ValueError("Spectrum replay currently supports recordings with onset guard disabled")
        proc.session = SpectrumCascade(proc)
        proc.dry_delay *= 2  # account for both graphs in dry alignment, flush and trimming
    elif mode == "hc":
        proc.session = FeedbackSession(proc.session)
    blocks, times = [], []
    for start in range(0, len(x), 2 * HOP):
        chunk = x[start : start + 2 * HOP]
        t = time.perf_counter_ns()
        out = proc.process_samples(chunk)
        elapsed = (time.perf_counter_ns() - t) / 1e6
        if start >= SR and chunk.size == 2 * HOP:
            times.append(elapsed)
        blocks.append(out)
    state_before_flush = proc.state.copy()
    blocks.append(proc.flush())
    y = np.concatenate(blocks)[proc.dry_delay :]
    assert y.size == x.size and np.isfinite(y).all()
    return y, times, state_before_flush


def trace(y, cfg, delay=0):
    vad = CleanVAD(margin_db=cfg["clean_vad_margin_db"], min_db=cfg["clean_vad_min_db"], hang=cfg["clean_vad_hang"])
    for _ in range(delay):
        vad.step(np.zeros(HOP, np.float32))
    return np.array([vad.step(y[i : i + HOP]) for i in range(0, y.size, HOP)])


def db(x):
    return 10 * np.log10(max(1e-12, np.mean(x.astype(float) ** 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--mode", choices=("hc", "spectrum"), default="hc")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads(next(args.bundle.glob("*.json")).read_text())
    cfg = meta["settings"]
    x = read(next(args.bundle.glob("*_raw.wav")))
    ref = read(next(args.bundle.glob("*_vi.wav")))
    assert x.size == ref.size == meta["samples"]
    waves, timings, states = {}, {"A": [], "B": []}, {}
    for key in "ABBA":
        y, times, state = replay(x, cfg, args.mode if key == "B" else "single")
        if key in waves:
            assert np.array_equal(waves[key], y)
        waves[key], states[key] = y, state
        timings[key].extend(times)
    assert all(np.array_equal(v, states["B"][k]) for k, v in states["A"].items())
    manifest = json.loads((ROOT / "models/dpcrn_curriculum_v2.json").read_text())
    delay = manifest["streaming_delay_frames"]
    traces = {k: trace(v, cfg, delay * (2 if k == "B" and args.mode == "spectrum" else 1)) for k, v in waves.items()}
    raw = trace(x, cfg)
    for k, v in waves.items():
        save(args.output / f"{k}.wav", v)
    with (args.output / "hops.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["start_s", "raw_dbfs", "A_dbfs", "B_dbfs", "raw_vad", "A_vad", "B_vad"])
        for n in range(len(raw)):
            sl = slice(n * HOP, (n + 1) * HOP)
            writer.writerow(
                [
                    n / 100,
                    *(round(db(v[sl]), 3) for v in (x, waves["A"], waves["B"])),
                    int(raw[n]),
                    int(traces["A"][n]),
                    int(traces["B"][n]),
                ]
            )
    summary = {}
    for k, times in timings.items():
        v = traces[k]
        summary[k] = dict(
            dbfs=db(waves[k]),
            on_s=float(v.sum() / 100),
            onsets_s=(np.where(np.diff(v.astype(int), prepend=0) == 1)[0] / 100).tolist(),
            chunk_ms=dict(zip(["p50", "p95", "p99"], np.percentile(times, [50, 95, 99]), strict=True)),
            measured_chunks=len(times),
            over_20ms=int(np.sum(np.array(times) > 20)),
        )
    summary["A_only_s"] = float((traces["A"] & ~traces["B"]).sum() / 100)
    summary["B_only_s"] = float((traces["B"] & ~traces["A"]).sum() / 100)
    summary["recorded_vi_correlation_with_A"] = np.corrcoef(ref, waves["A"])[0, 1]
    summary["mode"] = args.mode
    summary["identical_first_pass_states_before_flush"] = True
    result = json.dumps(summary, indent=2)
    (args.output / "summary.json").write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()

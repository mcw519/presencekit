"""Replay PresenceKit's downstream mix on sample-aligned recorded raw and VI tracks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import wave
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from presencekit.audio import HOP, SR, CleanVAD, CrossCheckAdjuster, VisionView  # noqa: E402
from presencekit.config import Config  # noqa: E402

CHUNK = 2 * HOP
RMS_FLOOR = 1e-12


def read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path)) as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (SR, 1, 2):
            raise ValueError(f"{path}: expected 16 kHz mono PCM16")
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768


def write_pcm(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as f:
        f.setparams((1, 2, SR, 0, "NONE", "not compressed"))
        f.writeframes(np.rint(x * 32768).clip(-32768, 32767).astype("<i2").tobytes())


def make_vad(cfg: Config) -> CleanVAD:
    return CleanVAD(margin_db=cfg.clean_vad_margin_db, min_db=cfg.clean_vad_min_db, hang=cfg.clean_vad_hang)


def vad_trace(x: np.ndarray, cfg: Config) -> np.ndarray:
    vad = make_vad(cfg)
    return np.array([vad.step(x[i : i + HOP]) for i in range(0, len(x) - HOP + 1, HOP)])


def recorded_view(block: dict, cfg: Config) -> VisionView:
    # The recording has sampled decisions and selected target, not vision snapshots.
    # In this capture a target <= near_range is sufficient to reconstruct a near view.
    decision = block["decision"]
    distance = block.get("distance")
    target = block.get("target_track")
    usable = decision != "acoustic_only"
    person = target is not None or decision in {"agree", "false_reject", "false_accept"}
    all_far = decision == "false_accept" or (
        cfg.distance_enabled and distance is not None and distance > cfg.near_range
    )
    near = (
        person
        and not all_far
        and (
            (cfg.distance_enabled and distance is not None and distance <= cfg.near_range)
            or decision == "false_reject"
            or not cfg.distance_enabled
        )
    )
    return VisionView(usable, person, near, all_far, target, distance, block.get("why", ""))


def replay(raw, vi, metadata, cfg, lead_samples):
    base_mix = cfg.iso_dry_blend
    if base_mix <= 0:
        raise ValueError("Recording must have a positive isolation share")
    clock = [b["t"] for b in metadata]
    if not clock or clock != sorted(clock):
        raise ValueError("Decision records must be nonempty and ordered by time")
    views = [recorded_view(b, cfg) for b in metadata]
    raw_vad, clean_vad = vad_trace(raw, cfg), vad_trace(vi, cfg)
    mix, gain = base_mix, 1.0
    adjuster = CrossCheckAdjuster()
    recovered = np.empty_like(vi)
    isolated = np.empty_like(vi)
    records = []

    for start in range(0, len(raw), CHUNK):
        end = min(start + CHUNK, len(raw))
        raw_end = min(len(raw), end + lead_samples) // HOP
        clean_end = start // HOP
        raw_speech = bool(raw_vad[raw_end - 1]) if raw_end else False
        clean_speech = bool(clean_vad[clean_end - 1]) if clean_end else False
        idx = bisect_right(clock, start / SR) - 1
        view = views[max(0, idx)]
        adj = adjuster.decide(view, clean_speech, raw_speech, cfg, base_mix)
        size = end - start
        mix += (adj.mix - mix) * -math.expm1(-size / SR / 0.05)
        share = np.clip(mix / base_mix, 0.0, 1.0)
        tau = cfg.xcheck_attack if adj.gain > gain else cfg.xcheck_release
        ramp = adj.gain + (gain - adj.gain) * np.exp(-np.arange(1, size + 1) / (max(tau, 0.001) * SR))
        gain = ramp[-1]
        isolated[start:end] = vi[start:end] * ramp
        recovered[start:end] = (share * vi[start:end] + (1 - share) * raw[start:end]) * ramp
        records.append(
            {
                "start_s": start / SR,
                "decision": adj.label,
                "raw_vad": raw_speech,
                "clean_vi_vad": clean_speech,
                "speech": adj.speech,
                "enhanced_share": share,
                "gain": gain,
            }
        )
    return {"mix_on": recovered, "mix_off": isolated}, records, raw_vad


def dbfs(x: np.ndarray) -> float:
    return 10 * math.log10(max(RMS_FLOOR, float(np.mean(x.astype(np.float64) ** 2))))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def measure(raw, out, vi, outputs, raw_vad, spans, cfg):
    signals = {"raw": raw, "recorded_out": out, "vi": vi, **outputs}
    vads = {k: vad_trace(s, cfg) for k, s in signals.items()}
    groups = defaultdict(list)
    span_rows = []
    onset_rows = []
    starts = np.arange(len(raw_vad)) * HOP
    # Backdate detector transitions by its consecutive active-hop requirement.
    rises = np.flatnonzero(raw_vad & ~np.r_[False, raw_vad[:-1]])
    min_hops = math.ceil(make_vad(cfg).min_on * SR / HOP)
    onset_samples = (rises - min_hops + 1) * HOP

    def row(tag, start, end, mask):
        active = np.zeros(len(starts), dtype=bool)
        for a, b in mask:
            active |= (starts >= a) & (starts + HOP <= b)
        result = {"tag": tag, "start_s": start, "end_s": end, "seconds": sum(b - a for a, b in mask) / SR}
        for key, signal in signals.items():
            result[f"{key}_dbfs"] = dbfs(np.concatenate([signal[a:b] for a, b in mask]))
            result[f"{key}_waveform_vad_s"] = float(np.count_nonzero(vads[key] & active) * HOP / SR)
        result["mix_off_minus_on_db"] = result["mix_off_dbfs"] - result["mix_on_dbfs"]
        return result

    for span in spans:
        start, end, tag = span["start_s"], span["end_s"], span["tag"]
        first, last = round(start * SR), round(end * SR)
        if not 0 <= first < last <= len(raw):
            raise ValueError(f"Span out of bounds: {span}")
        groups[tag].append((first, last))
        span_rows.append(row(tag, start, end, [(first, last)]))
        if tag == "near / keep":
            for onset in onset_samples[(onset_samples >= first) & (onset_samples + SR // 10 <= last)]:
                onset_rows.append(row(tag, onset / SR, onset / SR + 0.1, [(onset, onset + SR // 10)]))
    summaries = [row(tag, None, None, mask) for tag, mask in groups.items()]
    if onset_rows:
        windows = [(round(r["start_s"] * SR), round(r["end_s"] * SR)) for r in onset_rows]
        summaries.append(row("near / keep first 100 ms", None, None, windows))
    return span_rows, onset_rows, summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("spans", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--raw-lead-ms", type=int, default=60, help="Approximate live arrival lead over aligned VI output")
    args = p.parse_args()
    if args.raw_lead_ms < 0:
        p.error("--raw-lead-ms must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    tag_data = json.loads(args.spans.read_text())
    raw_path = args.bundle / tag_data["file"]
    base_name = raw_path.name.removesuffix("_raw.wav")
    meta = json.loads((args.bundle / f"{base_name}.json").read_text())
    cfg = Config()
    for key, value in meta["settings"].items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    raw = read_pcm(raw_path)
    vi = read_pcm(args.bundle / f"{base_name}_vi.wav")
    out = read_pcm(args.bundle / f"{base_name}_out.wav")
    if (
        tag_data["sample_rate"] != SR
        or len(raw) != len(vi)
        or len(raw) != len(out)
        or len(raw) != round(tag_data["duration_s"] * SR)
    ):
        raise ValueError("Recording tracks and span metadata must have matching sample rate and length")
    outputs, records, raw_vad = replay(raw, vi, meta["blocks"], cfg, args.raw_lead_ms * SR // 1000)
    span_rows, onsets, summaries = measure(raw, out, vi, outputs, raw_vad, tag_data["spans"], cfg)
    for key, signal in outputs.items():
        write_pcm(args.output / f"{key}.wav", signal)
    write_csv(args.output / "spans.csv", span_rows + summaries)
    write_csv(args.output / "near_onsets.csv", onsets)
    write_csv(args.output / "blocks.csv", records)
    result = {
        "raw_lead_ms": args.raw_lead_ms,
        "approximate_vision_replay": True,
        "groups": summaries,
        "near_onsets": len(onsets),
    }
    text = json.dumps(result, indent=2)
    (args.output / "summary.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

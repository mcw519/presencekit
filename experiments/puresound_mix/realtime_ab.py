"""Paced WebSocket A/B of recovery modes against saved span labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fusion_gate import acoustic_onsets
from realtime import Config, TestClient, app, prepare_video, run
from replay import CHUNK, SR, dbfs, read_pcm, vad_trace, write_csv, write_pcm

from presencekit.audio import AudioStream
from presencekit.web import session


def replay_run(client, frames, pcm, raw, cfg, enabled, duration, *, fps, suppress_mouth=False, video_gap=None):
    output = np.full(raw.size, np.nan, np.float32)
    vi = output.copy()
    trace, server_packets, socket_packets = [], [], []
    ledger = []

    class MeasuredStream(AudioStream):
        def process(self, pcm16, snap, now, vis_age=0.0, **kwargs):
            if suppress_mouth:
                # Keep the same visual processing and target selection. Only bypass mouth evidence at the mix gate.
                kwargs["mouth_age"] = math.inf
            decision_vi_vad = self.vad.speech
            out, speech = super().process(pcm16, snap, now, vis_age, **kwargs)
            count = len(out) // 2
            end = self._iso_origin + self.iso.next_output_sample
            start = end - count
            recovery = self._recovery_meta or {}
            row = {
                "input_end_s": self._input_samples / SR,
                "output_start_s": start / SR,
                "output_end_s": end / SR,
                "output_samples": count,
                "decision": self.adj.label,
                "reason": self.adj.why,
                "vision_age_ms": round(vis_age * 1000, 2) if math.isfinite(vis_age) else None,
                "speech": speech,
                "decision_vi_vad": decision_vi_vad,
                "vi_vad": self.vad.speech,
                "mix": self.mix,
                "resets": self.iso.resets,
                "target_track": self.adj.target_track,
                "mouth_score": recovery.get("mouth_score"),
                "mouth_age_ms": recovery.get("mouth_age_ms"),
                "mouth": recovery.get("mouth", False),
                "acoustic": recovery.get("acoustic", False),
                "allowed": recovery.get("allowed", False),
            }
            trace.append(row)
            if count:
                server_packets.append(out)
                # A model reset can leave a hole or emit warm-up before its own input starts.
                left, right = max(start, self._iso_origin), min(end, raw.size)
                if right > left:
                    aligned = slice(left - start, right - start)
                    dest = slice(left, right)
                    if np.isfinite(output[dest]).any() or not np.array_equal(self.iso.last_ref[aligned], raw[dest]):
                        raise ValueError("Output no longer matches the raw sample timeline")
                    output[dest] = np.frombuffer(out, "<i2")[aligned].astype(np.float32) / 32768
                    vi[dest] = self.iso.last_vi[aligned]
                    ledger.append((left, right, row))
            return out, speech

    with patch.object(session, "AudioStream", MeasuredStream):
        metrics = run(
            client,
            frames,
            pcm,
            fps,
            enabled,
            cfg,
            fixed_capture=True,
            output_packets=socket_packets,
            video_seconds=duration,
            video_gap=video_gap,
        )
    if metrics["audio_dropped"] or server_packets != socket_packets or len(trace) != len(pcm):
        raise ValueError("Packet loss or incomplete audio replay")
    if suppress_mouth and any(row["mouth"] for row in trace):
        raise ValueError("The acoustic-only run used mouth evidence")
    return output, vi, trace, ledger, metrics


def trigger_counts(ledger, start, end):
    counts = {}
    for name in ("mouth", "mouth_only", "acoustic", "both"):
        previous_end = -1
        blocks = episodes = 0
        for left, right, row in ledger:
            if right <= start or left >= end:
                continue
            flag = {
                "mouth": row["mouth"],
                "mouth_only": row["mouth"] and not row["acoustic"],
                "acoustic": row["acoustic"],
                "both": row["mouth"] and row["acoustic"],
            }[name]
            if flag:
                episodes += int(left != previous_end)
                blocks += 1
                previous_end = right
            else:
                previous_end = -1
        counts[name] = {"blocks": blocks, "episodes": episodes}
    return counts


def expiry_probe(client, video, raw, cfg, output, fps):
    start, stop = 27.5, 33.65
    gap = (30.5 - start, 33.0 - start)
    frames = prepare_video(video, start, stop - start)
    audio = raw[round(start * SR) : round(stop * SR)]
    padded = np.pad(audio, (0, 5 * CHUNK))
    pcm = [(padded[i : i + CHUNK] * 32768).astype("<i2").tobytes() for i in range(0, padded.size, CHUNK)]
    final, vi, trace, _, metrics = replay_run(
        client, frames, pcm, audio, cfg, True, stop - start, fps=fps, video_gap=gap
    )
    for row in trace:
        for field in ("input_end_s", "output_start_s", "output_end_s"):
            row[field] = round(row[field] + start, 6)
    stale = [row for row in trace if row["reason"] == "vision stale" and row["vision_age_ms"] is not None]
    if not stale:
        raise ValueError("The vision-expiry probe did not reach the stale fallback")
    first, last = min(stale, key=lambda row: row["input_end_s"]), max(stale, key=lambda row: row["input_end_s"])
    weak = [row for row in trace if row["output_start_s"] < 32.67 and row["output_end_s"] > 32.57]
    sl = slice(round((32.57 - start) * SR), round((32.67 - start) * SR))
    if not np.isfinite(final[sl]).all() or not weak:
        raise ValueError("The weak onset is missing from the expiry probe")
    write_csv(output / "vision_expiry_blocks.csv", trace)
    return {
        "gap_s": [30.5, 33.0],
        "no_frame_blocks": sum(row["vision_age_ms"] is None for row in trace),
        "stale_blocks": len(stale),
        "first_stale_input_end_s": first["input_end_s"],
        "last_stale_input_end_s": last["input_end_s"],
        "decisions": dict(Counter(row["decision"] for row in stale)),
        "recovery_blocks": sum(row["allowed"] for row in stale),
        "speech_blocks": sum(row["speech"] for row in stale),
        "weak_onset": {
            "decisions": dict(Counter(row["decision"] for row in weak)),
            "vision_age_ms": [min(row["vision_age_ms"] for row in weak), max(row["vision_age_ms"] for row in weak)],
            "out_dbfs": dbfs(final[sl]),
            "vi_dbfs": dbfs(vi[sl]),
        },
        "run": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("spans", type=Path)
    parser.add_argument("output", type=Path)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--audio-only-ab", action="store_true", help="Compare fusion with the mouth cue bypassed")
    comparison.add_argument("--legacy-vs-audio", action="store_true", help="Compare the two shipped recovery modes")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--probe-vision-expiry", action="store_true", help="Replay a short excerpt with a camera gap")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    labels = json.loads(args.spans.read_text())
    raw_path = args.bundle / labels["file"]
    stem = raw_path.name.removesuffix("_raw.wav")
    raw = read_pcm(raw_path)
    duration = labels["duration_s"]
    if len(raw) != round(duration * SR) or labels["sample_rate"] != SR:
        raise ValueError("Raw length and format must match labels")
    settings = json.loads((args.bundle / f"{stem}.json").read_text())["settings"]
    defaults = Config()
    cfg = {key: value for key, value in settings.items() if hasattr(defaults, key)}
    cfg.update(asr_provider="none", reid_enabled=False, adaptive_fps=False)
    frames = prepare_video(args.bundle / f"{stem}.mp4", 0, duration)
    # Keep a paced 100 ms zero tail so the final labelled raw samples can be emitted.
    padded = np.pad(raw, (0, 5 * CHUNK))
    pcm = [(padded[i : i + CHUNK] * 32768).astype("<i2").tobytes() for i in range(0, padded.size, CHUNK)]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    cases = (
        (("fusion", True, False), ("audio_only", True, True))
        if args.audio_only_ab
        else (("off", False, False), ("on", True, False))
    )
    if args.legacy_vs_audio:
        cases = (("legacy", False, False), ("audio", True, False))
    probe = None
    with TestClient(app) as client:
        for name, enabled, suppress_mouth in cases:
            print(f"Starting {name} at {args.fps:g} fps", flush=True)
            mode = name if args.legacy_vs_audio else ("audiovisual" if enabled else "legacy")
            run_cfg = {**cfg, "xcheck_recovery_mode": mode}
            result = replay_run(
                client, frames, pcm, raw, run_cfg, enabled, duration, fps=args.fps, suppress_mouth=suppress_mouth
            )
            results.append(result)
            output, vi, trace, _, metrics = result
            metrics["mode"] = name
            metrics["no_frame_blocks"] = sum(row["vision_age_ms"] is None for row in trace)
            metrics["past_vision_expiry_blocks"] = sum(
                row["vision_age_ms"] is not None and row["reason"] == "vision stale" for row in trace
            )
            write_pcm(args.output / f"{name}_out.wav", np.nan_to_num(output))
            write_pcm(args.output / f"{name}_vi.wav", np.nan_to_num(vi))
            write_csv(args.output / f"{name}_blocks.csv", trace)
            print(json.dumps(metrics), flush=True)
        if args.probe_vision_expiry:
            print("Probing vision expiry with audio recovery", flush=True)
            probe = expiry_probe(
                client,
                args.bundle / f"{stem}.mp4",
                raw,
                {**cfg, "xcheck_recovery_mode": "audio"},
                args.output,
                args.fps,
            )
    a, b = results
    masks = [np.isfinite(result[0]) for result in results]
    common = masks[0] & masks[1]
    if args.audio_only_ab or args.legacy_vs_audio:
        detector_cfg = Config()
        detector_cfg.update(cfg)
        events = acoustic_onsets(raw, vad_trace(raw, detector_cfg), detector_cfg)
        near = [span for span in labels["spans"] if span["tag"] == "near / keep"]
        onsets = [
            {"tag": "near onset first 100 ms", "start_s": e["start_s"], "end_s": round(e["start_s"] + 0.1, 3)}
            for e in events
            if any(span["start_s"] <= e["start_s"] <= span["end_s"] - 0.1 for span in near)
        ]
    else:
        onsets = [{"tag": "weak onset first 100 ms", "start_s": 32.57, "end_s": 32.67}]
    windows = [*labels["spans"], *onsets]
    rows = []
    first, second = [case[0] for case in cases]
    for w in windows:
        lo, hi = (round(w[key] * SR) for key in ("start_s", "end_s"))
        keep = common[lo:hi]
        if not keep.any():
            raise ValueError(f"No comparable audio in {w}")
        metrics = {
            "tag": w["tag"],
            "start_s": w["start_s"],
            "end_s": w["end_s"],
            "shared_samples": int(keep.sum()),
            "window_samples": hi - lo,
        }
        for name, result in zip((first, second), results, strict=True):
            metrics[f"{name}_out_dbfs"] = dbfs(result[0][lo:hi][keep])
            metrics[f"{name}_vi_dbfs"] = dbfs(result[1][lo:hi][keep])
        metrics[f"{second}_minus_{first}_out_db"] = metrics[f"{second}_out_dbfs"] - metrics[f"{first}_out_dbfs"]
        for mode, result in zip((first, second), results, strict=True):
            for name, counts in trigger_counts(result[3], lo, hi).items():
                for field, val in counts.items():
                    metrics[f"{mode}_{name}_{field}"] = val
        rows.append(metrics)
    write_csv(args.output / "spans.csv", rows)
    summary = {
        "duration_s": duration,
        "frame_rate": args.fps,
        "runs": [a[4], b[4]],
        "aligned_samples": [int(mask.sum()) for mask in masks],
        "common_samples": int(common.sum()),
        f"{first}_triggers": trigger_counts(a[3], 0, raw.size),
        f"{second}_triggers": trigger_counts(b[3], 0, raw.size),
        "vision_expiry": probe,
        "spans": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Replay mouth-assisted recovery with a short acoustic onset fallback."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from mouth_gate import gated_mix, motion_by_block
from replay import CHUNK, HOP, SR, Config, make_vad, measure, read_pcm, replay, write_csv, write_pcm

from presencekit.acoustic_recovery import HISTORY_S, HOLD_S, RISE_DB, TILT_DB, onset_features


def acoustic_onsets(raw, raw_vad, cfg):
    """Compute features once the raw VAD confirms the onset."""
    active_hops = math.ceil(make_vad(cfg).min_on * SR / HOP)
    history = round(HISTORY_S * SR)
    events = []
    for last_hop in np.flatnonzero(raw_vad & ~np.r_[False, raw_vad[:-1]]):
        start = (int(last_hop) - active_hops + 1) * HOP
        ready = start + active_hops * HOP
        if start < history:
            continue
        rise, tilt = onset_features(raw[start:ready], raw[start - history : start], SR)
        events.append(
            {"start_sample": start, "ready_sample": ready, "start_s": start / SR, "rise_db": rise, "tilt_db": tilt}
        )
    return events


def rescue_blocks(events, blocks, lead, rise_db=RISE_DB, tilt_db=TILT_DB):
    allowed = np.zeros(len(blocks), dtype=bool)
    for event in events:
        if event["rise_db"] < rise_db or event["tilt_db"] > tilt_db:
            continue
        start = event["start_sample"]
        ready = event["ready_sample"]
        for i in range(start // CHUNK, min(len(blocks), math.ceil((start + HOLD_S * SR) / CHUNK))):
            if (i + 1) * CHUNK + lead >= ready:
                allowed[i] = blocks[i]["decision"] == "false_reject"
    return allowed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("spans", type=Path)
    p.add_argument("mouth_run", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()

    labels = json.loads(args.spans.read_text())
    stem = labels["file"].removesuffix("_raw.wav")
    raw, vi, out = (read_pcm(args.bundle / f"{stem}_{name}.wav") for name in ("raw", "vi", "out"))
    if not len(raw) == len(vi) == len(out) == round(labels["duration_s"] * SR) or labels["sample_rate"] != SR:
        raise ValueError("Audio tracks must match the span metadata")
    metadata = json.loads((args.bundle / f"{stem}.json").read_text())
    cfg = Config()
    for key, value in metadata["settings"].items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    threshold = json.loads((args.mouth_run / "summary.json").read_text())["motion_threshold"]
    with (args.mouth_run / "frames.csv").open() as f:
        frames = [
            {
                "video_t_s": float(row["video_t_s"]),
                "mouth_motion": float(row["mouth_motion"]) if row["mouth_motion"] else None,
            }
            for row in csv.DictReader(f)
        ]
    lead = round(0.06 * SR)
    outputs, blocks, raw_vad = replay(raw, vi, metadata["blocks"], cfg, lead)
    motion = motion_by_block(frames, blocks)
    events = acoustic_onsets(raw, raw_vad, cfg)
    rescue = rescue_blocks(events, blocks, lead)

    outputs["mix_mouth"], _ = gated_mix(raw, vi, blocks, motion, cfg, threshold)
    outputs["mix_audio"], _ = gated_mix(raw, vi, blocks, [None] * len(blocks), cfg, threshold, acoustic_recovery=rescue)
    outputs["mix_fusion"], trace = gated_mix(raw, vi, blocks, motion, cfg, threshold, acoustic_recovery=rescue)
    span_rows, onsets, groups = measure(raw, out, vi, outputs, raw_vad, labels["spans"], cfg)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, audio in outputs.items():
        write_pcm(args.output / f"{name}.wav", audio)
    for row, audio in zip(trace, rescue, strict=True):
        row["allows_recovery"] = row["mouth_allows_recovery"] or bool(audio)
        row["audio_allows_recovery"] = bool(audio)
    write_csv(args.output / "blocks.csv", trace)
    write_csv(args.output / "spans.csv", span_rows + groups)
    write_csv(args.output / "onsets.csv", onsets)
    write_csv(args.output / "audio_events.csv", events)

    sensitivity = []
    for lead_ms in (40, 60, 80):
        lead_samples = round(lead_ms * SR / 1000)
        base, test_blocks, _ = replay(raw, vi, metadata["blocks"], cfg, lead_samples)
        for offset_ms in (-80, 0, 80):
            visual = motion_by_block(frames, test_blocks, offset_ms / 1000)
            audio = rescue_blocks(events, test_blocks, lead_samples)
            base["mix_mouth"], _ = gated_mix(raw, vi, test_blocks, visual, cfg, threshold)
            base["mix_fusion"], _ = gated_mix(raw, vi, test_blocks, visual, cfg, threshold, acoustic_recovery=audio)
            spans, attacks, _ = measure(raw, out, vi, base, raw_vad, labels["spans"], cfg)
            far = [s for s in spans if s["tag"] == "far / suppress"]
            weak = next(a for a in attacks if abs(a["start_s"] - 32.57) < 0.001)
            sensitivity.append(
                {
                    "raw_lead_ms": lead_ms,
                    "video_offset_ms": offset_ms,
                    "far1_dbfs": far[0]["mix_fusion_dbfs"],
                    "far2_dbfs": far[1]["mix_fusion_dbfs"],
                    "far1_change_from_mouth_db": far[0]["mix_fusion_dbfs"] - far[0]["mix_mouth_dbfs"],
                    "far2_change_from_mouth_db": far[1]["mix_fusion_dbfs"] - far[1]["mix_mouth_dbfs"],
                    "weak_onset_dbfs": weak["mix_fusion_dbfs"],
                }
            )
    write_csv(args.output / "sensitivity.csv", sensitivity)
    summary = {
        "mouth_threshold": threshold,
        "rise_db": RISE_DB,
        "tilt_db": TILT_DB,
        "hold_ms": round(HOLD_S * 1000),
        "acoustic_events": sum(e["rise_db"] >= RISE_DB and e["tilt_db"] <= TILT_DB for e in events),
        "audio_recovery_blocks": int(rescue.sum()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()

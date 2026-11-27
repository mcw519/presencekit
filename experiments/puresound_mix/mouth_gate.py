"""Compare a YuNet/OpenCV mouth-motion gate with the recorded VI replay baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from replay import CHUNK, SR, Config, measure, read_pcm, replay, write_csv, write_pcm

from presencekit.detectors.face import OrtYuNetDetector
from presencekit.inference import OrtOptions
from presencekit.mouth_motion import FRESH_S, align_face, mouth_motion


def extract(video, model):
    detector = OrtYuNetDetector(model, options=OrtOptions(providers=("cpu",), intra_op_threads=1))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video}")
    previous = previous_row = None
    last_t = -1
    records = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if t <= last_t:
                raise ValueError("Video timestamps must increase")
            faces = detector.detect(frame)
            row = max((f for f, _ in faces), key=lambda f: f[2] * f[3], default=None)
            aligned = align_face(frame, row) if row is not None else None
            score = None
            if aligned is not None and previous is not None and t - last_t <= FRESH_S:
                center = row[:2] + row[2:4] / 2
                old_center = previous_row[:2] + previous_row[2:4] / 2
                scale_ratio = row[2] / previous_row[2]
                if np.linalg.norm(center - old_center) <= previous_row[2] * 0.25 and 0.75 <= scale_ratio <= 1.33:
                    score = mouth_motion(previous, aligned)
            records.append({"video_t_s": t, "faces": len(faces), "mouth_motion": score})
            previous, previous_row, last_t = aligned, row, t
    finally:
        capture.release()
    if not records:
        raise ValueError("No frames decoded")
    return records


def motion_by_block(frames, blocks, offset_s=0):
    times = np.array([x["video_t_s"] + offset_s for x in frames])
    scores = np.array([x["mouth_motion"] if x["mouth_motion"] is not None else np.nan for x in frames])
    values = []
    for b in blocks:
        t = b["start_s"]
        end = np.searchsorted(times, t, side="right")
        recent = scores[max(0, end - 3) : end]
        if end and t - times[end - 1] <= FRESH_S and len(recent) == 3 and np.all(np.isfinite(recent)):
            values.append(float(np.median(recent)))
        else:
            values.append(None)
    return values


def gated_mix(raw, vi, blocks, motion, cfg, threshold, *, acoustic_recovery=None):
    # This recording has 100% VI at base mix and a unity post-mix gain.
    if cfg.iso_dry_blend != 1 or any(b["gain"] != 1 for b in blocks):
        raise ValueError("Mouth replay requires base mix and gain of 1")
    output = np.empty_like(vi)
    mix = 1.0
    trace = []
    if acoustic_recovery is None:
        acoustic_recovery = (False for _ in blocks)
    for i, (b, score, acoustic) in enumerate(zip(blocks, motion, acoustic_recovery, strict=True)):
        mouth = score is not None and score > threshold
        allow = b["decision"] == "false_reject" and (mouth or acoustic)
        share = cfg.xcheck_recover_mix if allow else 1
        sl = slice(i * CHUNK, (i + 1) * CHUNK)
        mix += (share - mix) * -math.expm1(-len(vi[sl]) / SR / 0.05)
        output[sl] = mix * vi[sl] + (1 - mix) * raw[sl]
        trace.append(
            {
                "start_s": b["start_s"],
                "decision": b["decision"],
                "mouth_motion": score,
                "mouth_allows_recovery": b["decision"] == "false_reject" and mouth,
                "enhanced_share": mix,
            }
        )
    return output, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("spans", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/yunet_s_dynamic.onnx"))
    parser.add_argument("--calibration-end", type=float, default=10.441)
    args = parser.parse_args()
    labels = json.loads(args.spans.read_text())
    raw_name = labels["file"]
    stem = raw_name.removesuffix("_raw.wav")
    raw, vi, out = (read_pcm(args.bundle / f"{stem}_{name}.wav") for name in ("raw", "vi", "out"))
    if not len(raw) == len(vi) == len(out) == round(labels["duration_s"] * SR) or labels["sample_rate"] != SR:
        raise ValueError("Audio tracks must match the span metadata")
    metadata = json.loads((args.bundle / f"{stem}.json").read_text())
    cfg = Config()
    for key, value in metadata["settings"].items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    outputs, blocks, raw_vad = replay(raw, vi, metadata["blocks"], cfg, round(SR * 0.06))
    frames = extract(args.bundle / f"{stem}.mp4", args.model)
    nominal = motion_by_block(frames, blocks)
    calibration = [
        s for s, b in zip(nominal, blocks, strict=True) if s is not None and b["start_s"] < args.calibration_end
    ]
    if not calibration:
        raise ValueError("No valid mouth motion for calibration")
    # Keep the main rule independent of the late weak onset and the far spans.
    threshold = float(np.percentile(calibration, 75))
    outputs["mix_mouth"], trace = gated_mix(raw, vi, blocks, nominal, cfg, threshold)
    span_rows, onsets, groups = measure(raw, out, vi, outputs, raw_vad, labels["spans"], cfg)
    sensitivity = []
    for offset in (-0.08, 0, 0.08):
        motion = motion_by_block(frames, blocks, offset)
        for percentile in (0, 50, 75, 90):
            cutoff = float(np.percentile(calibration, percentile))
            test, _ = gated_mix(raw, vi, blocks, motion, cfg, cutoff)
            sensitivity_spans, sensitivity_onsets, _ = measure(
                raw, out, vi, {**outputs, "mix_mouth": test}, raw_vad, labels["spans"], cfg
            )
            far = [r for r in sensitivity_spans if r["tag"] == "far / suppress"]
            weak = next(r for r in sensitivity_onsets if abs(r["start_s"] - 32.57) < 0.001)
            sensitivity.append(
                {
                    "video_offset_ms": round(offset * 1000),
                    "calibration_percentile": percentile,
                    "threshold": cutoff,
                    "far1_dbfs": far[0]["mix_mouth_dbfs"],
                    "far2_dbfs": far[1]["mix_mouth_dbfs"],
                    "weak_onset_dbfs": weak["mix_mouth_dbfs"],
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, signal in outputs.items():
        write_pcm(args.output / f"{name}.wav", signal)
    for name, rows in (
        ("frames", frames),
        ("blocks", trace),
        ("spans", span_rows + groups),
        ("onsets", onsets),
        ("sensitivity", sensitivity),
    ):
        write_csv(args.output / f"{name}.csv", rows)
    summary = {
        "frames": len(frames),
        "valid_frame_scores": sum(f["mouth_motion"] is not None for f in frames),
        "motion_threshold": threshold,
        "calibration_end_s": args.calibration_end,
        "valid_motion_blocks": sum(s is not None for s in nominal),
        "recovery_blocks": sum(r["mouth_allows_recovery"] for r in trace),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

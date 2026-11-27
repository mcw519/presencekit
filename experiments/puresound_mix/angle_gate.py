"""Exploratory replay with face frontalness gating only dry recovery."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from replay import CHUNK, SR, read_pcm, write_pcm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("replay_dir", type=Path)
    p.add_argument("samples", type=Path)
    p.add_argument("--threshold", type=float, default=0.90)
    args = p.parse_args()
    with args.samples.open() as f:
        faces = list(csv.DictReader(f))
    with (args.replay_dir / "blocks.csv").open() as f:
        blocks = list(csv.DictReader(f))
    raw = read_pcm(next(args.bundle.glob("*_raw.wav")))
    vi = read_pcm(next(args.bundle.glob("*_vi.wav")))
    times = np.array([float(row["video_t_s"]) for row in faces])
    out = np.empty_like(vi)
    mix = 1.0
    # This recording uses base VI share 1, recover share 0.3 and unity gain.
    for block, start in zip(blocks, range(0, len(raw), CHUNK), strict=True):
        assert float(block["gain"]) == 1
        t = start / SR
        idx = np.searchsorted(times, t, side="right") - 1
        frontal = faces[idx]["near_largest_frontal"] if idx >= 0 and t - times[idx] <= 0.30 else ""
        recover = block["decision"] == "false_reject" and frontal and float(frontal) >= args.threshold
        target = 0.3 if recover else 1.0
        sl = slice(start, start + CHUNK)
        mix += (target - mix) * -math.expm1(-vi[sl].size / SR / 0.05)
        out[sl] = mix * vi[sl] + (1 - mix) * raw[sl]
    write_pcm(args.replay_dir / "mix_angle.wav", out)


if __name__ == "__main__":
    main()

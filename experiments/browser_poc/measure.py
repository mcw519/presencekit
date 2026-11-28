"""Compare sample-aligned Python #9 and browser results."""

import csv
import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2] / "artifacts/browser-poc"
RATE = 16000


def db(x):
    return float(10 * np.log10(max(1e-12, np.mean(np.square(x, dtype=np.float64)))))


def measure(directory):
    name = directory.name
    ref = np.load(HERE / f"{name}.npz")
    web = {
        k: np.fromfile(HERE / f"{name}-web-{k}.bin", dtype)
        for k, dtype in [("out", "<f4"), ("vi", "<f4"), ("speech", "u1"), ("allowed", "u1")]
    }
    trace = json.loads((HERE / f"{name}-web.json").read_text())
    print(
        name,
        "frames:",
        len(trace["videoMs"]),
        "blocks:",
        len(trace["audioMs"]),
        "vision p95:",
        round(np.percentile(trace["videoMs"], 95), 2),
        "audio p95:",
        round(np.percentile(trace["audioMs"], 95), 2),
        "audio dispatch-to-output p95:",
        round(np.percentile(trace["latencyMs"], 95), 2),
        "backlog:",
        trace["maxAudioBacklog"],
    )
    windows = json.loads((directory / "fixture.json").read_text())["spans"]
    for span in windows:
        a, b = span["start_s"], span["end_s"]
        region = slice(round(a * RATE), round(b * RATE))
        shared = np.isfinite(ref["out"][region]) & np.isfinite(web["out"][region])
        p = {k: ref[k][region][shared] for k in web}
        w = {k: web[k][region][shared] for k in web}
        flags = (
            sum(
                max(0, min(round(b * RATE), t["right"]) - max(round(a * RATE), t["left"]))
                for t in trace["trace"]
                if t["view"].get("anyFar")
            )
            / RATE
        )
        row = dict(
            recording=name,
            tag=span["tag"],
            start=a,
            end=b,
            paired_s=shared.sum() / RATE,
            py_out_dbfs=db(p["out"]),
            web_out_dbfs=db(w["out"]),
            py_vi_dbfs=db(p["vi"]),
            web_vi_dbfs=db(w["vi"]),
            vi_diff_dbfs=db(w["vi"] - p["vi"]),
            py_speech_s=p["speech"].sum() / RATE,
            web_speech_s=w["speech"].sum() / RATE,
            py_recovered_s=p["allowed"].sum() / RATE,
            web_recovered_s=w["allowed"].sum() / RATE,
            web_any_far_s=flags,
            out_different_samples=np.count_nonzero(p["out"] != w["out"]),
            out_max_difference_lsb=round(np.max(np.abs(p["out"] - w["out"])) * 32768),
            speech_difference_s=np.count_nonzero(p["speech"] != w["speech"]) / RATE,
        )
        if span["tag"] in ("far / suppress", "weak onset") or (name.startswith("presence-2026-11-27") and a == 11.665):
            if not shared.all():
                raise ValueError(f"Missing audio in {name} at {a}s")
            for mode, signal in (("python", p["out"]), ("web", w["out"])):
                with wave.open(str(HERE / f"{name}-{a:g}-{mode}.wav"), "wb") as handle:
                    handle.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
                    handle.writeframes(np.round(signal * 32768).astype("<i2").tobytes())
        print(span["tag"], a, b, *(round(row[k], 4) for k in list(row)[4:]))
        yield row


if __name__ == "__main__":
    rows = [row for p in sorted((HERE / "fixtures").glob("*")) for row in measure(p)]
    with (HERE / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

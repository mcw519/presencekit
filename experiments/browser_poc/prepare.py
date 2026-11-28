"""Stage static assets and create private, 2 fps replay fixtures and Python references."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from presencekit.inference.models import CATALOG

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "aba2962876c8ec8abec7971378d670677e0ec859"
STATIC = ROOT / "artifacts/browser-poc/site"
sys.path.insert(0, str(ROOT / "experiments/puresound_mix"))
from realtime_ab import (  # noqa: E402
    CHUNK,
    SR,
    Config,
    TestClient,
    app,
    prepare_video,
    read_pcm,
    replay_run,
    write_pcm,
)


def stage(upstream):
    assets = ROOT / "presencekit/static"
    shutil.copytree(assets / "vendor/puresound", STATIC / "design", dirs_exist_ok=True)
    shutil.copyfile(assets / "app.css", STATIC / "design/presence.css")
    for name in ("LICENSE", "NOTICE"):
        shutil.copyfile(ROOT / name, STATIC / name)
    sdk = upstream / "sdk/web"
    subprocess.run(
        [
            "git",
            "-C",
            str(upstream),
            "diff",
            "--quiet",
            PIN,
            "--",
            "sdk/web/src",
            "sdk/web/package-lock.json",
            "sdk/web/tsconfig.json",
        ],
        check=True,
    )
    for name in ("runtime.js", "audio.js"):
        target = STATIC / "vendor" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sdk / "dist" / name, target)
    ort = sdk / "node_modules/onnxruntime-web"
    if json.loads((ort / "package.json").read_text())["version"] != "1.24.3":
        raise ValueError("Expected PureSound's pinned ONNX Runtime Web 1.24.3")
    for name in ("ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"):
        shutil.copyfile(ort / "dist" / name, STATIC / "vendor" / name)
    for src, dst in [
        ("LICENSE", "PURESOUND-LICENSE"),
        ("NOTICE", "PURESOUND-NOTICE"),
        ("licenses/LICENSE", "ORT-LICENSE"),
        ("licenses/ThirdPartyNotices.txt", "ORT-THIRD-PARTY"),
    ]:
        shutil.copyfile(sdk / src, STATIC / "vendor" / dst)
    (STATIC / "models").mkdir(exist_ok=True)
    shutil.copyfile(HERE / "licenses/YUNET-LICENSE", STATIC / "models/YUNET-LICENSE")
    for key in ("puresound", "puresound-manifest", "yunet"):
        spec = CATALOG[key]
        source = ROOT / "models" / spec.filename
        if hashlib.sha256(source.read_bytes()).hexdigest() != spec.sha256:
            raise ValueError(f"Unexpected model hash: {source}")
        shutil.copyfile(source, STATIC / "models" / spec.filename)
    for path in HERE.glob("*"):
        if path.suffix in (".html", ".mjs", ".css") and path.name != "verify.mjs":
            shutil.copyfile(path, STATIC / path.name)


def fixture(bundle, reference):
    label_path = bundle / (bundle.name + "_raw.spans.json")
    if not label_path.exists():
        label_path = bundle.parent / label_path.name
    labels = json.loads(label_path.read_text())
    raw_file = bundle / labels["file"]
    raw = read_pcm(raw_file)
    duration = raw.size / SR
    assert duration == labels["duration_s"] and labels["sample_rate"] == SR
    cfg = {
        k: v
        for k, v in json.loads((bundle / (bundle.name + ".json")).read_text())["settings"].items()
        if hasattr(Config(), k)
    }
    cfg.update(
        adaptive_fps=False,
        asr_provider="none",
        reid_enabled=False,
        xcheck_recovery_mode="audio",
        distance_enabled=True,
        zone_max=1.3,
        near_range=1.3,
        iso_dry_blend=1,
        iso_onset_guard=False,
    )
    frames = prepare_video(bundle / (bundle.name + ".mp4"), 0, duration)
    directory = STATIC.parent / "fixtures" / bundle.name
    directory.mkdir(parents=True, exist_ok=True)
    indices = np.searchsorted([t for t, _ in frames], np.arange(0, duration, 0.5), side="right") - 1
    videos = []
    for tick, index in enumerate(indices):
        if index < 0:
            continue
        name = f"{tick:03}.jpg"
        (directory / name).write_bytes(frames[index][1])
        videos.append({"time": tick / 2, "file": name})
    shutil.copyfile(raw_file, directory / "raw.wav")
    windows = [s for s in labels["spans"] if s["tag"] != "double-talk"]
    if bundle.name.startswith("presence-2026-11-24"):
        windows.append({"tag": "weak onset", "start_s": 32.57, "end_s": 32.67})
    (directory / "fixture.json").write_text(json.dumps({"frames": videos, "settings": cfg, "spans": windows}))

    if reference:
        padded = np.pad(raw, (0, 5 * CHUNK))
        pcm = [(padded[i : i + CHUNK] * 32768).astype("<i2").tobytes() for i in range(0, padded.size, CHUNK)]
        with TestClient(app) as client:
            out, vi, _, ledger, timing = replay_run(client, frames, pcm, raw, cfg, True, duration, fps=2)
        speech, allowed = np.zeros((2, raw.size), dtype=bool)
        for a, b, row in ledger:
            speech[a:b] = row["speech"]
            allowed[a:b] = row["allowed"]
        ref = STATIC.parent / bundle.name
        np.savez_compressed(ref.with_suffix(".npz"), out=out, vi=vi, speech=speech, allowed=allowed)
        write_pcm(ref.with_suffix(".wav"), np.nan_to_num(out))
        ref.with_suffix(".json").write_text(json.dumps(timing, indent=2))
        print(bundle.name, "Python #9", timing["video_packets"], timing["audio_dropped"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puresound", type=Path, default=ROOT.parent / "PureSound")
    parser.add_argument("--reference", action="store_true", help="Run paced Python #9 replay")
    parser.add_argument("bundle", type=Path, nargs="*")
    args = parser.parse_args()
    if args.reference:
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "-c",
                "core.fsmonitor=false",
                "diff",
                "--quiet",
                "7ee50a2",
                "--",
                "presencekit",
                "experiments/puresound_mix",
            ],
            check=True,
        )
    stage(args.puresound)
    for bundle in args.bundle:
        fixture(bundle, args.reference)

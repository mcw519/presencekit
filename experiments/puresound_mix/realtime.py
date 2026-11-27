"""Pace a recording through the real WebSocket runtime without a browser or network."""

from __future__ import annotations

import argparse
import json
import struct
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from replay import CHUNK, SR, Config

from presencekit.server import app


def prepare_video(path, start, seconds):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 - start
            if t >= seconds:
                break
            if t >= 0:
                encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not encoded:
                    raise ValueError("Could not encode frame")
                frames.append((t, jpeg.tobytes()))
    finally:
        capture.release()
    if not frames:
        raise ValueError("No video frames")
    return frames


def run(
    client,
    frames,
    pcm,
    fps,
    enabled,
    cfg,
    *,
    fixed_capture=False,
    output_packets=None,
    video_seconds=None,
    video_gap=None,
):
    audio, stats, errors = [], [], []
    audio_events = [(i * CHUNK / SR, 1, chunk) for i, chunk in enumerate(pcm, start=1)]
    video_events = []
    next_frame = 0
    if fixed_capture:
        times = np.array([t for t, _ in frames])
        duration = video_seconds or len(pcm) * CHUNK / SR
        for tick in np.arange(0, duration, 1 / fps):
            index = np.searchsorted(times, tick, side="right") - 1
            if index >= 0:
                video_events.append((tick, 4, frames[index][1]))
    else:
        for t, jpeg in frames:
            if t + 0.01 >= next_frame:
                video_events.append((t, 4, jpeg))
                next_frame += 1 / fps
    events = sorted(audio_events + video_events)
    if video_gap is not None:
        first, last = video_gap
        events = [event for event in events if event[1] == 1 or not first <= event[0] < last]
        video_events = [event for event in video_events if not first <= event[0] < last]

    with client.websocket_connect("/ws") as conn:
        settings = dict(cfg)
        if "xcheck_recovery_mode" not in settings:
            settings["xcheck_mouth_recovery"] = enabled
        conn.send_json({"type": "config", "config": settings})
        conn.send_json({"type": "start"})
        ready = threading.Event()

        def read():
            while True:
                msg = conn.receive()
                if msg["type"] == "websocket.close":
                    return
                if output_packets is not None and msg.get("bytes", b"").startswith(b"\x02"):
                    output_packets.append(msg["bytes"][9:])
                if msg.get("text"):
                    event = json.loads(msg["text"])
                    kind = event["type"]
                    if kind == "pong":
                        return
                    if kind == "state" and event.get("running"):
                        ready.set()
                    if kind == "audio" and event.get("available"):
                        audio.append(event)
                    if kind == "stats":
                        stats.append(event["conn"])
                    if kind == "error":
                        errors.append(event)

        reader = threading.Thread(target=read)
        reader.start()
        if not ready.wait(10):
            raise RuntimeError("Runtime did not start")
        begin = time.perf_counter()
        for t, tag, data in events:
            remaining = begin + t - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            packet = struct.pack("<Bd", tag, t * 1000)
            conn.send_bytes(packet + (b"\0" if tag == 4 else b"") + data)
        time.sleep(0.3)
        conn.send_json({"type": "ping", "t": 0})
        reader.join(timeout=3)
        if reader.is_alive():
            raise RuntimeError("No pong from runtime")
        conn.send_json({"type": "stop"})
    if not audio or not stats or errors:
        raise RuntimeError(f"Replay did not complete: {errors}")
    recent = audio[len(audio) // 4 :]
    recovery = [a["recovery"] for a in recent if "recovery" in a]

    def percentile(values, at):
        values = [v for v in values if v is not None]
        return round(float(np.percentile(values, at)), 3) if values else None

    return {
        "target_fps": fps,
        "enabled": enabled,
        "video_packets": len(video_events),
        "frames_processed_last_stats": stats[-1]["frames"]["processed"],
        "processed_fps_last_5s": stats[-1]["fps"],
        "vision_p95_ms": stats[-1]["latency_ms"]["total"]["p95"],
        "mouth_processing_p95_ms": stats[-1]["latency_ms"].get("mouth", {}).get("p95"),
        "audio_processing_p95_ms": audio[-1]["proc_p95_ms"],
        "audio_queue_p95_ms": audio[-1]["queue_wait_ms_p95"],
        "audio_dropped": audio[-1]["dropped"],
        "vision_capture_gap_p95_ms": percentile([a["vision_timing"]["capture_gap_ms"] for a in recent], 95),
        "mouth_frame_interval_median_ms": percentile([r["frame_interval_ms"] for r in recovery], 50),
        "mouth_score_available_samples": sum(r["mouth_score"] is not None for r in recovery),
        "recovery_samples": len(recovery),
        "mouth_allowed_samples": sum(r["mouth"] for r in recovery),
        "acoustic_allowed_samples": sum(r["acoustic"] for r in recovery),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=24)
    parser.add_argument("--seconds", type=float, default=9)
    args = parser.parse_args()

    video = next(args.bundle.glob("*.mp4"))
    raw = next(args.bundle.glob("*_raw.wav"))
    metadata = json.loads(next(args.bundle.glob("*.json")).read_text())["settings"]
    defaults = Config()
    cfg = {key: value for key, value in metadata.items() if hasattr(defaults, key)}
    cfg.update(asr_provider="none", reid_enabled=False, adaptive_fps=False)
    frames = prepare_video(video, args.start, args.seconds)
    with wave.open(str(raw), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SR, 1, 2):
            raise ValueError("Expected 16 kHz mono PCM16")
        wav.setpos(round(args.start * SR))
        data = wav.readframes(round(args.seconds * SR))
    pcm = [data[i : i + CHUNK * 2] for i in range(0, len(data), CHUNK * 2)]
    results = []
    with TestClient(app) as client:
        for fps, enabled in ((0.5, True), (2, True), (15, True), (24, False), (24, True)):
            result = run(client, frames, pcm, fps, enabled, cfg)
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"start_s": args.start, "duration_s": args.seconds, "runs": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

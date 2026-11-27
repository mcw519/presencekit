"""SDK facade: PresenceEngine drives the same layers as the web runtime, with no web framework and no threads."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from presencekit import audio
from presencekit.config import Config, State
from presencekit.core.tracks import Detection
from presencekit.sdk import PresenceEngine
from presencekit.vision import VisionPipeline

FACE = Path(__file__).parent / "data" / "face.jpg"


def frame(face: bool = True) -> np.ndarray:
    fr = np.full((480, 640, 3), 120, np.uint8)
    if face:
        f = cv2.imread(str(FACE))
        f = cv2.resize(f, (int(f.shape[1] * 200 / f.shape[0]), 200))
        y, x = 140, 320 - f.shape[1] // 2
        fr[y : y + 200, x : x + f.shape[1]] = f
    return fr


@pytest.fixture(scope="module")
def pipe():
    return VisionPipeline(allow_download=False)  # ONNX Runtime models when present, else Haar + histogram


def engine(pipe, **kw):
    t = [0.0]
    return PresenceEngine(pipeline=pipe, pose=False, clock=lambda: t[0], **kw), t


def test_import_loads_no_web_framework_and_starts_no_thread():
    code = (
        "import sys, threading; import presencekit.sdk; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('fastapi', 'starlette', 'uvicorn', 'psutil') "
        "or m.startswith(('presencekit.web', 'presencekit.server', 'presencekit.runtime'))); "
        "print(repr((bad, threading.active_count())))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "([], 1)"


def test_a_visible_speaker_opens_a_session_and_absence_ends_it(pipe):
    eng, t = engine(pipe)
    seen = []
    for i in range(20):  # 10 s in view, speaking
        t[0] = i * 0.5
        step = eng.process_frame(frame(), speaking=True)
        seen += [e.extra.get("to") for e in step.events if e.kind == "transition"]
    assert "ACTIVE" in seen and eng.state == State.ACTIVE
    assert step.snapshot["tracks"] and eng.timings["detect"] >= 0
    for i in range(1, 121):  # 60 s: empty frames, silence, ticks in between
        t[0] = 10 + i * 0.5
        eng.process_frame(frame(face=False), speaking=False) if i % 2 else eng.tick()
        if eng.state == State.ENDED:
            break
    assert eng.state == State.ENDED


def test_set_speaking_reset_and_jpeg_input(pipe):
    eng, t = engine(pipe)
    ok, jpg = cv2.imencode(".jpg", frame())
    assert ok and eng.process_jpeg(jpg.tobytes()) is not None
    assert eng.process_jpeg(b"not a jpeg") is None
    assert eng.process_jpeg(b"") is None
    assert eng.set_speaking(True).snapshot["speaking"] is True
    eng.reset()
    assert eng.snapshot()["tracks"] == [] and eng.snapshot()["speaking"] is False


def test_configure_validates_like_the_web_settings(pipe):
    eng, _ = engine(pipe, config=Config())
    assert eng.configure({"max_width": 1e999}) == ["invalid value for max_width"]  # no OverflowError
    assert eng.configure({"fps_high": 99}) and eng.config.fps_high < 99
    assert "no visible person" in eng.calibrate_distance(1.5)[0]


def test_custom_pose_detector_and_info(pipe):
    class Pose:
        name = "fake pose"
        calls = 0

        def detect(self, img):
            Pose.calls += 1
            return []

    eng, _ = engine(pipe)
    eng.frames.pose_loader = lambda: Pose()
    eng.configure({"pose_enabled": True})
    eng.process_frame(frame())
    info = eng.info()
    assert Pose.calls == 1 and info["pose"] == "fake pose" and info["pipeline"] == pipe.name
    assert info["audio"] is None and isinstance(info["models"], list)


def test_process_audio_needs_the_puresound_pipeline(pipe):
    eng, _ = engine(pipe)
    with pytest.raises(RuntimeError, match="audio_pipeline"):
        eng.process_audio(bytes(640))
    if not audio.puresound_available()[0]:
        pytest.skip("PureSound SDK or model not installed")
    eng.configure({"audio_pipeline": "puresound"})
    total = 0
    for _ in range(10):
        out, step = eng.process_audio(bytes(640))  # 20 ms of silence
        total += len(out)
        assert step is None  # silence never changes the speech decision
    assert total > 0 and eng.info()["audio"] == "puresound"


def test_audio_only_recovery_keeps_a_quiet_near_session_and_ends_after_departure(pipe, monkeypatch):
    class SilentVI:
        base_mix = 1.0
        resets = 0
        next_output_sample = 0

        def __init__(self, **kwargs):
            pass

        def reset(self):
            self.resets += 1
            self.next_output_sample = 0

        def process(self, raw, mix):
            self.next_output_sample += raw.size
            self.last_vi = np.zeros_like(raw)
            return (1 - mix) * raw

    monkeypatch.setattr(audio, "puresound_available", lambda: (True, "ok"))
    monkeypatch.setattr(audio, "Isolator", SilentVI)
    cfg = Config(
        audio_pipeline="puresound",
        xcheck_recovery_mode="audio",
        iso_dry_blend=1.0,
        distance_enabled=True,
        near_range=1.3,
        presence_model="hmm",
        tracker_mode="kalman",
        intent_model="score",
    )
    eng, t = engine(pipe, config=cfg)
    near_present = True
    near_box = (0.33, 0.26, 0.55, 0.55)
    far_box = (0.66, 0.33, 0.70, 0.39)

    def detections(*args, **kwargs):
        far = Detection(far_box, 0.95, frontal=1.0)
        near = Detection(near_box, 0.95, frontal=1.0)
        return ([near, far] if near_present else [far]), {}

    monkeypatch.setattr(eng.frames, "process", detections)
    image = np.zeros((16, 16, 3), np.uint8)
    silence = bytes(640)
    samples = np.arange(320) / audio.SR
    near_onset = (4000 * np.sin(2 * np.pi * 240 * samples)).astype("<i2").tobytes()
    far_tone = (4000 * np.sin(2 * np.pi * 2700 * samples)).astype("<i2").tobytes()

    def far_input(i):
        # Syllable-like pauses lower the adaptive floor while staying within the raw VAD hangover.
        return far_tone if i % 15 < 10 else silence

    def feed(i, pcm):
        t[0] = i * 0.02
        if i % 25 == 0:
            eng.process_frame(image)
        eng.process_audio(pcm)

    # Start with a qualifying near utterance, then leave time for the participant to join.
    for i in range(250):
        feed(i, near_onset if 25 <= i < 45 else silence)
    snap = eng.snapshot()
    assert snap["state"] == State.IDLE and not snap["speaking"]
    assert len([track for track in snap["tracks"] if track["participant"]]) == 1
    session = snap["session"]
    last_speech = eng.core.tracker.last_speech

    # Two minutes with a silent near participant and an ongoing far-side raw VAD signal.
    for i in range(250, 6250):
        feed(i, far_input(i))
        assert eng.state == State.IDLE
    assert eng._audio.raw_vad.speech and eng._audio.adj.label == "false_reject"
    assert not eng.snapshot()["speaking"] and eng.core.tracker.last_speech == last_speech
    assert eng.snapshot()["session"] == session

    # Continue the same raw VAD signal while the near face leaves the camera's view.
    near_present = False
    ended_at = None
    for i in range(6250, 7500):
        feed(i, far_input(i))
        assert eng._audio.raw_vad.speech and not eng.core.tracker.speaking
        if ended_at is not None:
            assert eng.state == State.ENDED
        elif eng.state == State.ENDED:
            ended_at = t[0]
    snap = eng.snapshot()
    assert ended_at is not None and ended_at - 125 <= 20
    assert snap["session"] == session


def test_switching_to_host_vad_discards_puresound_state(pipe, monkeypatch):
    stream = Mock()
    stream.process.return_value = (b"", True)
    new_stream = Mock()
    new_stream.process.return_value = (b"", False)
    factory = Mock(side_effect=[stream, new_stream])
    monkeypatch.setattr(audio, "puresound_available", lambda: (True, "ok"))
    monkeypatch.setattr(audio, "AudioStream", factory)
    eng, _ = engine(pipe, config=Config(audio_pipeline="puresound"))
    image = frame()

    eng.process_audio(bytes(640))
    assert eng.snapshot()["speaking"] is True
    assert eng.configure({"audio_pipeline": "invalid"})
    assert eng.process_frame(image, speaking=False).snapshot["speaking"] is True

    assert eng.configure({"audio_pipeline": "energy_vad"}) == []
    assert eng.info()["audio"] is None
    assert eng.process_frame(image).snapshot["speaking"] is False
    assert eng.process_frame(image, speaking=True).snapshot["speaking"] is True
    assert eng.process_frame(image, speaking=False).snapshot["speaking"] is False

    eng.configure({"audio_pipeline": "puresound"})
    eng.process_audio(bytes(640))
    assert factory.call_count == 2
    assert eng.process_frame(image, speaking=True).snapshot["speaking"] is False

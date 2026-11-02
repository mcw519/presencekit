"""SDK facade: PresenceEngine drives the same layers as the web runtime, with no web framework and no threads."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from presencekit import audio
from presencekit.config import Config, State
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

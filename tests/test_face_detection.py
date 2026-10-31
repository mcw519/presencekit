"""Face detection regressions: small, low-light and partly visible faces (images made or loaded in memory only)."""

import logging
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
from sources import ui_source

from presencekit.optimizer import FrameProcessor
from presencekit.tracker import Config

FACE = Path(__file__).parent / "data" / "face.jpg"


@pytest.fixture(scope="module")
def pipe():
    from presencekit.vision import VisionPipeline

    try:
        return VisionPipeline(allow_download=False)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"models unavailable: {e}")


def frame_with_face(height: int, cx: float) -> np.ndarray:
    face = cv2.imread(str(FACE))
    w = int(face.shape[1] * height / face.shape[0])
    f = cv2.resize(face, (w, height))
    fr = np.full((480, 640, 3), 120, np.uint8)
    x, y = int(cx * 640 - w / 2), 240 - height // 2
    fr[y : y + height, x : x + w] = f
    return fr


@pytest.mark.parametrize("cx", [0.3, 0.5, 0.7])
def test_default_max_width_detects_mid_distance_face(pipe, cx):
    # 100 px face image in a 640x480 frame: missed with the old max_width=320, found with the default
    fr = frame_with_face(100, cx)
    assert Config().max_width >= 640
    assert FrameProcessor(pipe).process(fr, Config(), True)[0]
    assert not FrameProcessor(pipe).process(fr, Config(max_width=320), True)[0]


def test_roi_uses_raw_camera_coordinates(pipe):
    # frames are sent unmirrored; a face on the raw right side is only found by a right-side ROI
    fr = frame_with_face(200, 0.75)
    right = Config(face_roi=(0.5, 0.0, 1.0, 1.0), crop_margin=0.0)
    left = Config(face_roi=(0.0, 0.0, 0.5, 1.0), crop_margin=0.0)
    d = FrameProcessor(pipe).process(fr, right, True)[0]
    assert d and d[0].bbox[0] > 0.5
    assert not FrameProcessor(pipe).process(fr, left, True)[0]


def test_ui_sends_unmirrored_frames_and_raw_roi():
    html = ui_source()
    assert "og.drawImage(v,0,0,off.width,off.height)" in html  # mirroring is CSS only, never in the sent frame
    assert "mx=x=>MIR()?1-x:x" in html and "C(mx((e.clientX-r.left)/r.width))" in html  # pointer -> raw x
    assert "mbox=b=>MIR()?[1-b[2],b[1],1-b[0],b[3]" in html  # raw ROI -> mirrored drawing


def test_motion_gate_never_reuses_no_face(pipe):
    clock = type("C", (), {"t": 0.0, "__call__": lambda s: s.t})()
    fp = FrameProcessor(pipe, clock=clock)
    cfg = Config(motion_gate=True, force_detect_seconds=10)
    empty = np.full((480, 640, 3), 120, np.uint8)
    assert fp.process(empty, cfg, True)[0] == []
    clock.t = 0.5
    _, tm = fp.process(empty, cfg, True)  # unchanged + no face -> must detect again, not reuse "nobody"
    assert not tm["skipped"]
    assert fp.counts["detect_calls"] == 2 and fp.counts["detect_hits"] == 0


def test_shared_detector_is_serialized_across_connections(pipe):
    # two connections share one YuNet; unsynchronized use corrupted memory (crash / wrong results)
    from presencekit.server import _vision_job

    a = cv2.imencode(".jpg", frame_with_face(200, 0.5))[1].tobytes()
    b = cv2.imencode(".jpg", cv2.resize(frame_with_face(200, 0.5), (400, 300)))[1].tobytes()
    cfg = Config(motion_gate=False, roi_crop=False)
    bad = []

    def run(data):
        fp = FrameProcessor(pipe)
        for _ in range(60):
            dets, _ = _vision_job(fp, data, cfg, True)
            if len(dets) != 1:
                bad.append(len(dets))

    ts = [threading.Thread(target=run, args=(d,)) for d in (a, b, a, b)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert bad == []


def test_ws_handshake_log_redacts_token():
    from presencekit.server import _RedactToken

    rec = logging.LogRecord(
        "uvicorn.error", 20, "", 0, '%s - "WebSocket %s" [accepted]', ("1.2.3.4:0", "/ws?token=SECRET123"), None
    )
    _RedactToken().filter(rec)
    assert "SECRET123" not in rec.getMessage() and "token=***" in rec.getMessage()

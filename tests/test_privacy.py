"""Privacy guard: processing frames over the WebSocket must not write any file or log image data."""

import base64
import logging
import os
import struct
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from presencekit import server

FACE = Path(__file__).parent / "data" / "face.jpg"


def _snapshot(*dirs: Path) -> dict[str, tuple[int, float]]:
    out = {}
    for d in dirs:
        for root, dnames, files in os.walk(d):
            dnames[:] = [
                x for x in dnames if x not in {".venv", ".git", "__pycache__", ".pytest_cache", "logs"}
            ]  # logs/: the live dev server
            for f in files:
                p = os.path.join(root, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                out[p] = (st.st_size, st.st_mtime)
    return out


def _frame() -> str:
    img = cv2.resize(cv2.imread(str(FACE)), (480, 480))
    f = np.zeros((480, 640, 3), np.uint8)
    f[:, 80:560] = img
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", f)[1].tobytes()).decode()


@pytest.mark.parametrize("binary", [False, True])
def test_ws_frames_write_no_files_and_log_no_image_data(monkeypatch, caplog, binary):
    server.pipeline()  # load/download models before taking the snapshot
    data = _frame()
    dirs = [Path.cwd(), Path(tempfile.gettempdir())]
    before = _snapshot(*dirs)

    def forbidden(*a, **k):
        raise AssertionError("cv2.imwrite must never be called")

    monkeypatch.setattr(cv2, "imwrite", forbidden)
    caplog.set_level(logging.DEBUG)
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "start"})
        for _ in range(3):
            if binary:
                ws.send_bytes(struct.pack("<BdB", 4, 1000.0, 0) + base64.b64decode(data.split(",")[1]))
            else:
                ws.send_json({"type": "frame", "speech": False, "data": data})
            while True:
                m = ws.receive_json()
                if m.get("type") == "state" and "detections" in m:
                    break
        assert m["detections"], "face should be detected"
        ws.send_json({"type": "stop"})

    after = _snapshot(*dirs)
    cwd, tmp = str(dirs[0]), str(dirs[1])
    img_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".npy")
    # project dir: nothing may change; shared temp dir (other processes may use it): no new image-like files
    new_or_changed = {
        p
        for p in after
        if before.get(p) != after[p] and (p.startswith(cwd) or (p.startswith(tmp) and p.lower().endswith(img_ext)))
    }
    assert not new_or_changed, f"files written during frame processing: {sorted(new_or_changed)[:10]}"
    payload = data.split(",")[1]
    assert payload[:64] not in caplog.text and "data:image" not in caplog.text

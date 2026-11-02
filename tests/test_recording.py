"""Optional client-side recording: aligned raw / vi / out streams; the server stores nothing."""

import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sources import server_source, ui_source

from presencekit import audio
from presencekit.audio import AudioStream
from presencekit.tracker import Config

OK, WHY = audio.puresound_available()
needs_ps = pytest.mark.skipif(not OK, reason=WHY)


def sig(sec=2.0, seed=1):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(int(16000 * sec)) * 0.1).astype(np.float32)
    return (np.clip(x, -1, 1) * 32767).astype("<i2")


@needs_ps
def test_tracks_are_sample_aligned_and_equal_length():
    a = AudioStream(Config(audio_pipeline="puresound", xcheck_enabled=False))
    x = sig()
    outs, vis, raws = [], [], []
    for i in range(0, x.size, 320):
        o, _ = a.process(x[i : i + 320].tobytes(), None, i / 16000)
        vi, rw = a.recording_tracks()
        assert len(o) == len(vi) == len(rw)
        outs.append(o), vis.append(vi), raws.append(rw)
    raw = np.frombuffer(b"".join(raws), "<i2")
    d = a.iso.delay
    # raw track == mic input delayed by exactly the PureSound streaming delay (zeros before)
    assert np.all(raw[:d] == 0)
    n = raw.size - d
    assert np.max(np.abs(raw[d:].astype(int) - x[:n].astype(int))) <= 1
    out = np.frombuffer(b"".join(outs), "<i2")
    vi = np.frombuffer(b"".join(vis), "<i2")
    assert out.size == vi.size == raw.size
    # no cross-check and default mix 0.9 -> out is the SDK output itself
    assert np.max(np.abs(out.astype(int) - vi.astype(int))) <= 1


@needs_ps
def test_ws_record_streams_three_tracks_and_writes_no_files(tmp_path, monkeypatch):
    # Watch a private temp dir (not the shared system /tmp, where unrelated processes write files and made
    # this test flaky) and the repo minus logs/.
    priv = tmp_path / "tmp"
    priv.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(priv))
    monkeypatch.setenv("TMPDIR", str(priv))
    from fastapi.testclient import TestClient

    from presencekit import server

    def snap(d):
        out = {}
        for root, dn, files in os.walk(d):
            dn[:] = [x for x in dn if x not in {".venv", ".git", "__pycache__", ".pytest_cache", "logs", ".ruff_cache"}]
            for f in files:
                p = os.path.join(root, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                out[p] = (st.st_size, st.st_mtime)
        return out

    server.pipeline()
    audio.Isolator()
    dirs = [Path.cwd(), Path(tempfile.gettempdir())]
    before = [snap(d) for d in dirs]
    x = sig(1.0)
    got = []
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "config": {"audio_pipeline": "puresound"}})
        ws.send_json({"type": "start"})
        ws.send_json({"type": "record", "on": True})
        for i in range(0, x.size, 320):
            ws.send_bytes(b"\x01" + struct.pack("<d", float(i)) + x[i : i + 320].tobytes())
        while len(got) < 10:
            m = ws.receive()
            b = m.get("bytes")
            if b and b[0] == 3:
                t, n = struct.unpack("<dI", b[1:13])
                assert len(b) == 13 + 6 * n  # out + vi + raw, n samples each
                got.append(n)
        ws.send_json({"type": "stop"})
    after = [snap(d) for d in dirs]
    ext = (".wav", ".pcm", ".raw", ".zip", ".json", ".npy", ".jpg", ".png")
    changed = [p for b, a in zip(before, after, strict=True) for p in a if b.get(p) != a[p] and p.lower().endswith(ext)]
    assert not changed, changed[:5]


def test_recording_is_client_side_opt_in():
    html = ui_source()
    assert "let rec=null" in html and "REC_MAX_S=600" in html  # off by default, 10 min cap
    assert "_raw.wav" in html and "_vi.wav" in html and "_out.wav" in html
    src = server_source()
    assert "wave" not in src and "open(" not in src  # the server has no file writer for audio

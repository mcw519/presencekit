"""Audio: PureSound glue (smoke, skipped without the SDK/model), clean VAD, reset on group change,
entry-onset rule, and audio privacy (no files written)."""

import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from presencekit import audio
from presencekit.audio import SR, AudioStream, CleanVAD, CrossCheckAdjuster
from presencekit.tracker import Config, Detection, PresenceTracker

OK, WHY = audio.puresound_available()
needs_ps = pytest.mark.skipif(not OK, reason=WHY)


def speechlike(sec=4.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * sec)) / SR
    ph = 2 * np.pi * np.cumsum(140 + 20 * np.sin(2 * np.pi * 0.5 * t)) / SR
    on = (np.sin(2 * np.pi * 0.7 * t) > 0).astype(float)
    return (sum(np.sin(k * ph) / k for k in range(1, 15)) * on * 0.1 + 0.01 * rng.normal(size=t.size)).astype(
        np.float32
    )


# ---------------------------------------------------------------- PureSound smoke (optional extra)
@needs_ps
def test_adapter_default_mix_is_sdk_output_and_chunk_invariant():
    from puresound_streaming import PureSoundStreamingRuntime

    x = speechlike()
    ref = PureSoundStreamingRuntime(audio.MODELS_DIR / f"{audio.MODEL_NAME}.onnx", provider="cpu").process_samples(x)
    iso = audio.Isolator()
    assert iso.base_mix == pytest.approx(0.9) and iso.delay == 480 and iso.vad_heads == []
    a = np.concatenate([iso.process(x[i : i + 320], iso.base_mix) for i in range(0, x.size, 320)])
    iso.reset()
    b = np.concatenate([iso.process(x[i : i + 517], iso.base_mix) for i in range(0, x.size, 517)])
    n = min(a.size, b.size, ref.size)
    assert n > x.size - 1000 and np.isfinite(a).all()
    assert np.abs(a[:n] - ref[:n]).max() == 0.0  # glue adds nothing at the default mix
    assert np.abs(a[:n] - b[:n]).max() == 0.0  # chunking invariance


@needs_ps
def test_lower_mix_adds_aligned_dry():
    x = speechlike(2.0)
    iso = audio.Isolator()
    full = np.concatenate([iso.process(x[i : i + 320], 0.9) for i in range(0, x.size, 320)])
    iso.reset()
    half = np.concatenate([iso.process(x[i : i + 320], 0.45) for i in range(0, x.size, 320)])
    n = min(full.size, half.size)
    dry = np.concatenate([np.zeros(480, np.float32), x])[:n]  # input aligned by the 30 ms look-ahead
    assert np.abs(half[:n] - np.clip(0.5 * full[:n] + 0.5 * dry, -1, 1)).max() < 1e-5


# ---------------------------------------------------------------- glue logic (no SDK needed)
class FakeIso:
    base_mix, delay, vad_heads = 0.9, 480, []

    class config:
        win_length = 512

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def process(self, x, mix):
        return x * mix


def snap(session, parts, log=()):
    return {
        "session": session,
        "tracks": [{"id": i, "participant": True, "visible": True, "distance": None} for i in parts],
        "log": list(log),
    }


def test_reset_on_new_group_and_participant_change():
    s = AudioStream(Config(), isolator_factory=FakeIso)
    assert not s.on_vision(snap(1, [1]), 0)
    assert not s.on_vision(snap(1, [1]), 1)
    assert s.on_vision(snap(1, [2], [{"event": "NEW_GROUP", "t": 2.0}]), 2) and s.iso.resets == 1
    assert not s.on_vision(snap(1, [2], [{"event": "NEW_GROUP", "t": 2.0}]), 3)  # same event: once
    assert s.on_vision(snap(2, [3]), 4) and s.iso.resets == 2  # new session
    s.cfg.iso_reset_on_change = False
    assert not s.on_vision(snap(3, [4]), 5)


def vsnap(*people, path="present"):
    """people: distances (None = unknown)."""
    return {
        "session": 1,
        "path": path,
        "log": [],
        "tracks": [{"id": i + 1, "participant": True, "visible": True, "distance": d} for i, d in enumerate(people)],
    }


def xc(snap, clean, raw, dist=True, age=0.1, calibrated=True, now=10.0, ad=None, **cfgkw):
    c = Config()
    c.update({"distance_enabled": dist, **cfgkw})
    ad = ad or CrossCheckAdjuster()
    return ad.decide(ad.view(snap, age, c, now, calibrated), clean, raw, c, 0.9)


@pytest.mark.parametrize(
    "people,clean,raw,label,speech,attenuated",
    [
        ((), True, True, "leak", False, True),  # 1) clean speech, nobody in view
        ((0.6,), False, True, "false_reject", True, False),  # 2) near talker, model quiet, raw speech
        ((1.8, 2.2), True, True, "false_accept", False, True),  # 3) clean speech, everyone beyond near_range
        ((0.7,), True, True, "agree", True, False),  # 4) clean speech + near person
        ((0.7, 2.0), True, True, "agree", True, False),  # near one wins over a far one
        ((), False, False, "silence", False, False),
        ((1.8,), False, True, "silence", False, False),  # far person + raw speech: not recovered
        ((0.6,), False, False, "silence", False, False),
    ],
)
def test_crosscheck_cases(people, clean, raw, label, speech, attenuated):
    a = xc(vsnap(*people), clean, raw)
    assert (a.label, a.speech) == (label, speech)
    assert (a.gain < 0.1) == attenuated
    assert a.mix == (0.3 if label == "false_reject" else 0.9)


@pytest.mark.parametrize(
    "kw",
    [
        dict(snap=None),  # no vision
        dict(snap=vsnap(0.6), age=5.0),  # stale frame
        dict(snap=vsnap(0.6, path="uncertain")),  # low confidence
        dict(snap=vsnap(), xcheck_enabled=False),
    ],
)
def test_uncertain_vision_trusts_acoustics(kw):
    snap = kw.pop("snap")
    for clean in (True, False):
        a = xc(snap, clean, True, **kw)
        assert a.label == "acoustic_only" and a.speech == clean and a.gain == 1.0 and a.mix == 0.9


def test_distance_disabled_or_uncalibrated_skips_false_accept():
    assert xc(vsnap(2.5), True, True, dist=False).label == "agree"  # presence in ROI only
    assert xc(vsnap(None), False, True, dist=False).label == "false_reject"
    assert xc(vsnap(2.5), True, True, calibrated=False, xcheck_require_calib=True).label == "agree"
    assert xc(vsnap(2.5), True, True, calibrated=False).label == "false_accept"  # calibration not required


def test_vision_hold_bridges_frame_gaps():
    ad = CrossCheckAdjuster()
    assert xc(vsnap(0.6), True, True, now=10.0, ad=ad).label == "agree"
    assert xc(vsnap(), True, True, now=10.5, ad=ad).label == "agree"  # within xcheck_hold 0.75 s
    assert xc(vsnap(), True, True, now=11.0, ad=ad).label == "leak"


class LoudIso(FakeIso):
    def process(self, x, mix):
        return np.full_like(x, 0.2)


def test_stream_attenuates_leak_and_counts():
    c = Config()
    s = AudioStream(c, isolator_factory=LoudIso)
    pcm = np.zeros(320, "<i2").tobytes()
    for k in range(30):  # clean VAD turns on (model output loud), nobody in view -> leak
        out, speech = s.process(pcm, vsnap(), 10 + k * 0.02, 0.1)
        assert not speech
    y = np.frombuffer(out, "<i2") / 32768
    assert s.adj.label == "leak" and s.counts["leak"] == 1 and np.abs(y).max() < 0.2 * 0.02
    out, speech = s.process(pcm, vsnap(0.5), 11.0, 0.1)
    assert speech and s.adj.label == "agree"


def test_decision_durations_use_audio_samples_and_preserve_entry_counts():
    class SelectedAdjuster(CrossCheckAdjuster):
        label = "acoustic_only"

        def decide(self, view, clean, raw, cfg, base_mix):
            return audio.Adjust(base_mix, label=self.label)

    adjuster = SelectedAdjuster()
    s = AudioStream(Config(), isolator_factory=FakeIso, adjuster=adjuster)
    # Irregular chunks and wall-clock gaps must not bias duration toward frequent decisions,
    # include time without input audio, or exclude the initial acoustic_only label.
    chunks = [("acoustic_only", 17), ("silence", 160), ("silence", 320), ("false_reject", 73)]
    chunks += [(label, 53 + i) for i, label in enumerate(audio.LABELS)]
    expected = dict.fromkeys(audio.LABELS, 0)
    expected_counts = dict.fromkeys(audio.LABELS, 0)
    previous = "acoustic_only"
    for i, (label, size) in enumerate(chunks):
        adjuster.label = label
        s.process(np.zeros(size, "<i2").tobytes(), None, now=i * 1000)
        expected[label] += size
        if previous != label:
            expected_counts[label] += 1
        previous = label
    meta = s.meta()
    total = sum(expected.values())
    assert meta["counts"] == expected_counts
    assert meta["decision_duration_s"] == pytest.approx({k: n / SR for k, n in expected.items()})
    assert meta["decision_ratio"] == pytest.approx({k: n / total for k, n in expected.items()})
    assert meta["classified_audio_s"] == pytest.approx(total / SR)
    assert sum(meta["decision_duration_s"].values()) == pytest.approx(meta["classified_audio_s"])
    assert sum(meta["decision_ratio"].values()) == pytest.approx(1)
    # Callers may hold or mutate a metadata snapshot without changing stream statistics.
    meta["decision_duration_s"]["silence"] = -1
    assert s.meta()["decision_duration_s"]["silence"] == pytest.approx(expected["silence"] / SR)


def test_decision_duration_zero_audio_and_model_reset():
    s = AudioStream(Config(), isolator_factory=FakeIso)
    meta = s.meta()
    assert meta["classified_audio_s"] == 0
    assert meta["decision_duration_s"] == dict.fromkeys(audio.LABELS, 0)
    assert meta["decision_ratio"] == dict.fromkeys(audio.LABELS, 0)
    s.process(b"", None, now=0)
    assert s.meta()["classified_audio_s"] == 0
    assert sum(s.meta()["decision_ratio"].values()) == 0
    s.process(np.zeros(320, "<i2").tobytes(), None, now=1)
    before = s.meta()
    assert before["decision_duration_s"]["acoustic_only"] == 0.02
    assert before["decision_ratio"]["acoustic_only"] == 1
    s.on_vision(snap(1, [1]), now=2)
    assert s.on_vision(snap(2, [2]), now=3)
    after = s.meta()
    for field in ("counts", "decision_duration_s", "decision_ratio", "classified_audio_s"):
        assert after[field] == before[field]
    assert after["resets"] == 1


def test_clean_vad_onset_and_hangover():
    v = CleanVAD()
    quiet = np.full(160, 1e-4, np.float32)
    loud = (0.1 * np.sin(np.arange(160) * 0.3)).astype(np.float32)
    for _ in range(100):
        v.step(quiet)
    assert not v.speech
    states = [v.step(loud) for _ in range(20)]
    assert states.index(True) == 5 and v.onsets == 1  # 60 ms min-run
    states = [v.step(quiet) for _ in range(40)]
    assert states.index(False) == 29  # 0.3 s hangover
    far = (0.001 * np.sin(np.arange(160) * 0.3)).astype(np.float32)  # -63 dBFS: below clean_vad_min_db
    assert not any(v.step(far) for _ in range(30))


# ---------------------------------------------------------------- entry rule (tracker)
E = np.random.default_rng(1).normal(size=128)


def det():
    return Detection((0.4, 0.3, 0.55, 0.55), 0.9, E.copy(), frontal=1.0)


def mk(priority):
    t = [0.0]
    c = Config()
    c.update({"priority_mode": priority, "audio_pipeline": "puresound"})
    tr = PresenceTracker(c, clock=lambda: t[0])
    tr.speech_source = "clean"
    return tr, t


@pytest.mark.parametrize("priority", ["vision", "balanced", "audio"])
def test_entry_needs_clean_onset_plus_visible_person(priority):
    tr, t = mk(priority)
    for i in range(10):  # person in view, silent: no session (vision alone is not an entry)
        t[0] = i * 0.5
        tr.update([det()], now=t[0])
    assert tr.state.value == "ENDED"
    tr.set_speech(True, now=5.0)
    assert tr.state.value == "ACTIVE" and tr.session == 1
    assert "entry" in [e for e in tr.log if e["event"] == "transition"][-1]["reason"]


@pytest.mark.parametrize("priority", ["vision", "balanced", "audio"])
def test_speech_without_person_does_not_open(priority):  # TV / far talker in an empty room
    tr, t = mk(priority)
    for i in range(6):
        t[0] = i * 0.5
        tr.update([], now=t[0])
    tr.set_speech(True, now=3.0)
    tr.update([], now=3.5)
    assert tr.state.value == "ENDED"


def test_far_person_plus_speech_does_not_open():
    tr, t = mk("balanced")
    tr.cfg.update({"distance_enabled": True})
    small = Detection((0.47, 0.4, 0.49, 0.43), 0.9, E.copy(), frontal=1.0)  # tiny face -> far
    for i in range(6):
        t[0] = i * 0.5
        tr.update([small], now=t[0])
    tr.set_speech(True, now=3.0)
    assert tr.state.value == "ENDED"


def test_browser_vad_behaviour_unchanged():
    tr = PresenceTracker(Config(), clock=lambda: 0.0)
    tr.update([det()], now=0.0)
    assert tr.state.value == "ACTIVE"  # energy_vad: person in ROI starts a session as before


# ---------------------------------------------------------------- privacy over the WebSocket
@needs_ps
def test_ws_audio_round_trip_writes_no_files():
    from fastapi.testclient import TestClient

    from presencekit import server

    def snapshot(*dirs):
        out = {}
        for d in dirs:
            for root, dn, files in os.walk(d):
                dn[:] = [x for x in dn if x not in {".venv", ".git", "__pycache__", ".pytest_cache", "logs"}]
                for f in files:
                    p = os.path.join(root, f)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    out[p] = (st.st_size, st.st_mtime)
        return out

    server.pipeline()
    dirs = [Path.cwd(), Path(tempfile.gettempdir())]
    audio.Isolator()  # warm the shared session before the snapshot
    before = snapshot(*dirs)
    x = (speechlike(1.0) * 32767).astype("<i2")
    got = 0
    metas = []
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "config": {"audio_pipeline": "puresound"}})
        ws.send_json({"type": "start"})
        for i in range(0, x.size, 320):
            ws.send_bytes(b"\x01" + struct.pack("<d", float(i)) + x[i : i + 320].tobytes())
        # the server keeps <= 500 ms of backlog (drops the oldest while the model warms up), so expect >= 0.5 s back
        while got < 8000 - 1200:
            m = ws.receive()
            if m.get("bytes"):
                assert m["bytes"][0] == 2
                got += (len(m["bytes"]) - 9) // 2
            elif m.get("text") and '"audio"' in m["text"]:
                metas.append(m["text"])
        ws.send_json({"type": "stop"})
    assert metas and '"available":true' in metas[-1]
    after = snapshot(*dirs)
    cwd, tmp = str(dirs[0]), str(dirs[1])
    ext = (".wav", ".pcm", ".raw", ".flac", ".mp3", ".ogg", ".npy", ".jpg", ".png")
    changed = {
        p
        for p in after
        if before.get(p) != after[p] and (p.startswith(cwd) or (p.startswith(tmp) and p.lower().endswith(ext)))
    }
    assert not changed, sorted(changed)[:10]


def test_audio_block_metrics_include_batch_wait_and_exclude_it_from_processing(monkeypatch):
    import time

    from fastapi.testclient import TestClient

    from presencekit.server import app
    from presencekit.web import session

    samples = []
    original_record = session.AudioStats.record

    def record(self, done, elapsed, duration, wait):
        samples.append((elapsed, duration, wait))
        original_record(self, done, elapsed, duration, wait)

    class FakeStream:
        def __init__(self, cfg):
            self.cfg = cfg
            time.sleep(0.05)  # allow later packets to collect into a batch

        def on_vision(self, snap, now):
            pass

        def process(self, pcm, snap, now, vis_age):
            time.sleep(0.01)
            return pcm, False

        def meta(self):
            return {"rtf": 0.5, "counts": {}}

    monkeypatch.setattr(session, "AudioStream", FakeStream)
    monkeypatch.setattr(session, "puresound_available", lambda: (True, ""))
    monkeypatch.setattr(session.AudioStats, "record", record)
    meta = None
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "config": {"audio_pipeline": "puresound"}})
        for n in range(5):
            ws.send_bytes(b"\x01" + struct.pack("<d", float(n)) + bytes(640))
        got = 0
        while got < 5:
            message = ws.receive()
            if message.get("bytes"):
                got += 1
            elif message.get("text"):
                data = json.loads(message["text"])
                if data.get("type") == "audio":
                    meta = data
    assert len(samples) == 5
    assert all(duration == 0.02 for _, duration, _ in samples)
    assert all(wait > elapsed for elapsed, _, wait in samples)  # queued initialization is not processing
    assert samples[-1][2] > samples[-2][2] + 0.005  # earlier block inside the same batch
    assert meta is not None and meta["rtf_recent"] > 0
    assert meta["proc_p95_ms"] > 0 and meta["queue_wait_ms_avg"] > meta["proc_p95_ms"]

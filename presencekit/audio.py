"""Server-side audio: PureSound streaming voice isolation glue, clean VAD, vision cross-check.

Only thin glue lives here. The isolation runtime is PureSound's own portable SDK
(``puresound_streaming``, installed with ``uv sync --extra puresound``); it is imported lazily so the
demo still runs without it (``audio_pipeline = energy_vad``).

PRIVACY: audio is processed in memory only; nothing here opens a file except the model + manifest.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .inference import CATALOG, OrtOptions, default_root, load_model

SR = 16000
HOP = 160  # 10 ms
MODEL_NAME = "dpcrn_curriculum_v2"
PURESOUND_SUBDIR = "egs/voice_isolate/pretrained_ckpt/streaming"
# sha256 of the files at PureSound commit 121bf62 (= model_zoo/catalog.yaml); the catalogue is the source
PURESOUND_FILES = {CATALOG[k].filename: CATALOG[k].sha256 for k in ("puresound", "puresound-manifest")}
MODELS_DIR = default_root()  # honours PRESENCE_MODEL_DIR, like the vision models


def puresound_available() -> tuple[bool, str]:
    """(ok, reason) – SDK importable and model files present."""
    try:
        import puresound_streaming  # noqa: F401
    except ImportError:
        return False, "puresound SDK not installed (uv sync --extra puresound)"
    missing = [n for n in PURESOUND_FILES if not (MODELS_DIR / n).exists()]
    if missing:
        return False, "model missing in models/ (run scripts/fetch_puresound_model.py)"
    return True, "ok"


def _shared_session(onnx_path: Path, threads: int = 1):
    """One ORT session per process, shared through :func:`presencekit.inference.load_model` (``run`` is
    thread-safe; per-stream state lives in the processor). Always the CPU provider: 10 ms frames are too small to
    gain from an accelerator."""
    return load_model(onnx_path, OrtOptions(providers=("cpu",), intra_op_threads=threads, inter_op_threads=1)).session


class Isolator:
    """PureSound SDK ``StftFrameOrtProcessor`` + a vision-controlled extra dry mix.

    The SDK applies the manifest's ``dry_blend`` (0.9 enhanced / 0.1 input, latency-aligned) and the
    optional onset guard exactly as upstream. The SDK has no runtime mix setter, so a lower enhanced
    share ``mix`` < 0.9 is produced here as ``a*sdk_out + (1-a)*dry`` with ``a = mix/0.9`` and the dry
    input aligned by the same ``streaming_delay_frames*hop``. At ``mix = 0.9`` ``a = 1`` and the output
    is the SDK's, sample for sample.
    """

    def __init__(self, model_dir: Path = MODELS_DIR, onset_guard: bool = False, guard_params: dict | None = None):
        from puresound_streaming import StftFrameOrtProcessor, StreamingRuntimeConfig

        onnx = model_dir / f"{MODEL_NAME}.onnx"
        self.manifest = json.loads((model_dir / f"{MODEL_NAME}.json").read_text(encoding="utf-8"))
        man = dict(self.manifest)
        if onset_guard:
            man["onset_guard"] = dict(guard_params or {"t_arm_s": 1.0})
        self.config = StreamingRuntimeConfig.from_manifest(man)
        self.proc = StftFrameOrtProcessor(self.config, _shared_session(onnx))
        self.base_mix = float(self.config.dry_blend)  # 0.9
        self.delay = self.config.streaming_delay_frames * self.config.hop_length  # 480 samples = 30 ms
        self.vad_heads = list(self.config.extra_output_names)  # v2: none -> energy VAD on the output
        self.reset()

    def reset(self) -> None:
        """New stream: clears model state + buffers and re-arms the onset guard (SDK ``reset``)."""
        self.proc.reset()
        self._dry = np.zeros(0, np.float32)
        self._dry_start = 0  # absolute input index of _dry[0]
        self._emitted = 0
        self.last_vi = self.last_ref = np.zeros(0, np.float32)
        self.resets = getattr(self, "resets", -1) + 1

    def process(self, x: np.ndarray, mix: float) -> np.ndarray:
        x = np.asarray(x, np.float32).reshape(-1)
        self._dry = np.concatenate([self._dry, x])
        y = self.proc.process_samples(x)
        n = y.size
        start = self._emitted - self.delay  # input index carried by y[0]
        self._emitted += n
        a = min(1.0, max(0.0, mix / self.base_mix))
        # input aligned to y (same samples, PureSound delay compensated); also kept for optional client recording
        ref = np.zeros(n, np.float32)
        lo, hi = max(start, self._dry_start), min(start + n, self._dry_start + self._dry.size)
        if hi > lo:
            ref[lo - start : hi - start] = self._dry[lo - self._dry_start : hi - self._dry_start]
        self.last_vi, self.last_ref = y, ref  # in memory only
        if n and a < 1.0:
            y = np.clip(a * y + (1.0 - a) * ref, -1.0, 1.0).astype(np.float32)
        keep = max(0, self._emitted - self.delay - self._dry_start)
        if keep:
            self._dry = self._dry[keep:]
            self._dry_start += keep
        return y


@dataclass
class CleanVAD:
    """Energy VAD on the enhanced signal (10 ms hops): adaptive floor + margin, absolute minimum level,
    min-run for onset, hangover for offset. Used because the released v2 export has no VAD head."""

    margin_db: float = 10.0
    min_db: float = -50.0
    min_on: float = 0.06
    hang: float = 0.3
    floor_db: float = -70.0
    speech: bool = False
    level_db: float = -100.0
    _run: int = 0
    _quiet: int = 0
    onsets: int = 0

    def step(self, hop: np.ndarray) -> bool:
        e = float(10 * math.log10(float(np.mean(hop.astype(np.float64) ** 2)) + 1e-10))
        self.level_db = e
        # floor: falls fast, rises slowly (≈3 dB/s) so speech does not lift it
        self.floor_db = e if e < self.floor_db else min(e, self.floor_db + 0.03)
        active = e > max(self.min_db, self.floor_db + self.margin_db)
        if active:
            self._run += 1
            self._quiet = 0
            if not self.speech and self._run * HOP / SR >= self.min_on:
                self.speech = True
                self.onsets += 1
        else:
            self._run = 0
            self._quiet += 1
            if self.speech and self._quiet * HOP / SR >= self.hang:
                self.speech = False
        return self.speech


LABELS = ("agree", "leak", "false_reject", "false_accept", "silence", "acoustic_only")


@dataclass
class Adjust:
    """Per-block decision: enhanced share ``mix`` (<= manifest dry_blend), target output ``gain`` (linear),
    corrected ``speech`` for the state machine, and the decision ``label`` (one of :data:`LABELS`)."""

    mix: float
    gain: float = 1.0
    speech: bool = False
    label: str = "acoustic_only"
    target_track: int | None = None
    distance: float | None = None
    why: str = ""


@dataclass
class VisionView:
    """What vision says about the zone in front of the robot, held for ``xcheck_hold`` s (vision runs at
    0.5-2 fps while audio blocks are 20 ms, so a short hold aligns the two streams)."""

    usable: bool  # False -> acoustic_only (no recent frame / uncertain / not running)
    person: bool  # anyone visible in the ROI (held)
    near: bool  # someone within near_range (held); = person when distance is not usable
    all_far: bool  # distance usable and every visible person beyond near_range
    target: int | None = None
    distance: float | None = None
    why: str = ""


class CrossCheckAdjuster:
    """Vision cross-checks PureSound's filtered result per block (docs/audio.md, decision table).

    ====================  ======================  ===============================================
    clean VAD / raw VAD    vision                  decision
    ====================  ======================  ===============================================
    clean speech           nobody in ROI           leak          -> attenuate, not speech
    clean speech           everyone > near_range   false_accept  -> attenuate, not speech
    clean speech           person <= near_range    agree         -> pass, speech
    no clean, raw speech   person <= near_range    false_reject  -> raise passthrough, speech
    otherwise              any                     silence       -> pass, not speech
    any                    uncertain / stale       acoustic_only -> PureSound output, clean VAD
    ====================  ======================  ===============================================
    """

    name = "crosscheck"

    def __init__(self):
        self._person_t = -1e9
        self._near_t = -1e9
        self._far_t = -1e9

    def view(self, snap: dict | None, vis_age: float, cfg, now: float, calibrated: bool) -> VisionView:
        if snap is None or vis_age > cfg.xcheck_stale or snap.get("path") == "uncertain":
            why = "no vision" if snap is None else ("vision stale" if vis_age > cfg.xcheck_stale else "uncertain")
            return VisionView(False, False, False, False, why=why)
        dist_ok = cfg.distance_enabled and (calibrated or not cfg.xcheck_require_calib)
        vis = [t for t in snap.get("tracks", []) if t["visible"]]
        withd = [t for t in vis if t.get("distance") is not None]
        near = [t for t in vis if not dist_ok or t.get("distance") is None or t["distance"] <= cfg.near_range]
        all_far = dist_ok and bool(vis) and len(withd) == len(vis) and not near
        if vis:
            self._person_t = now
        if near:
            self._near_t = now
        if all_far:
            self._far_t = now
        h = cfg.xcheck_hold
        person = now - self._person_t <= h
        is_near = now - self._near_t <= h
        # far only counts when nobody was near within the hold (asymmetric: protect onsets of a near talker)
        far = (now - self._far_t <= h) and not is_near
        pick = min(near or vis, key=lambda t: t.get("distance") or 0.0) if vis else None
        return VisionView(
            True,
            person,
            is_near,
            far,
            pick["id"] if pick else None,
            pick.get("distance") if pick else None,
            "distance" if dist_ok else "presence only",
        )

    def decide(self, v: VisionView, clean: bool, raw: bool, cfg, base_mix: float) -> Adjust:
        floor = 0.0 if cfg.xcheck_floor_db <= -60 else 10 ** (cfg.xcheck_floor_db / 20)
        if not cfg.xcheck_enabled or not v.usable:
            return Adjust(base_mix, 1.0, clean, "acoustic_only", v.target, v.distance, v.why or "disabled")
        if clean and not v.person:
            return Adjust(base_mix, floor, False, "leak", None, None, "clean speech, nobody in view")
        if clean and v.all_far:
            return Adjust(base_mix, floor, False, "false_accept", v.target, v.distance, "speaker beyond near_range")
        if clean:
            return Adjust(base_mix, 1.0, True, "agree", v.target, v.distance, "clean speech + near person")
        if raw and v.near:
            mix = min(base_mix, cfg.xcheck_recover_mix)
            return Adjust(mix, 1.0, True, "false_reject", v.target, v.distance, "near person talking, model quiet")
        return Adjust(base_mix, 1.0, False, "silence", v.target, v.distance, "")


class AudioStream:
    """Per-connection audio: PCM16 in -> PureSound isolation -> vision cross-check -> clean PCM16 out,
    corrected speech flag for the state machine, and per-block metadata."""

    def __init__(self, cfg, isolator_factory=None, adjuster=None):
        self.cfg = cfg
        self.iso = (isolator_factory or (lambda: Isolator(onset_guard=cfg.iso_onset_guard)))()
        self._guard_cfg = cfg.iso_onset_guard
        self.vad = CleanVAD()  # on the enhanced signal
        self.raw_vad = CleanVAD()  # on the raw input (for false_reject)
        self.adjuster = adjuster or CrossCheckAdjuster()
        self.adj = Adjust(self.iso.base_mix)
        self.gain = 1.0
        self.mix = self.iso.base_mix
        self.counts = {k: 0 for k in LABELS}  # number of entries into each label
        # Input sample counts, not wall time or label entries. Include quiet / vision-unavailable
        # audio so each duration and ratio has the same, explicit denominator.
        self._classified_samples = {k: 0 for k in LABELS}
        self.calibrated = False
        self._pending = np.zeros(0, np.float32)
        self._raw_pending = np.zeros(0, np.float32)
        self.compute_s = 0.0
        self.audio_s = 0.0
        self.raw_db = -100.0
        self.clean_db = -100.0
        self._group_sig = None
        self._ng_t = -1.0

    def on_vision(self, snap: dict | None, now: float) -> bool:
        """Reset the stream on NEW_GROUP / session change / participant set change. Returns True if reset."""
        if snap is None:
            return False
        parts = frozenset(t["id"] for t in snap.get("tracks", []) if t["participant"])
        sig = (snap.get("session"), parts)
        ng = [e["t"] for e in snap.get("log", []) if e.get("event") == "NEW_GROUP" and e.get("t", 0) > self._ng_t]
        if ng:
            self._ng_t = max(ng)
        old, self._group_sig = self._group_sig, sig
        changed = old is not None and bool(parts) and (old[0] != sig[0] or (bool(old[1]) and old[1] != parts))
        if self.cfg.iso_reset_on_change and (ng or changed):
            self.iso.reset()
            return True
        return False

    @staticmethod
    def _db(x) -> float:
        return float(10 * math.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-10)) if x.size else -100.0

    def _sync_params(self):
        c = self.cfg
        for v in (self.vad, self.raw_vad):
            v.margin_db, v.min_db, v.hang = c.clean_vad_margin_db, c.clean_vad_min_db, c.clean_vad_hang
        if c.iso_onset_guard != self._guard_cfg:  # guard toggled -> rebuild processor
            self._guard_cfg = c.iso_onset_guard
            self.iso = Isolator(onset_guard=self._guard_cfg)

    def process(self, pcm16: bytes, snap: dict | None, now: float, vis_age: float = 0.0) -> tuple[bytes, bool]:
        """One chunk. The decision uses the VADs up to the previous block (<= one chunk, 20 ms, of lag)."""
        self._sync_params()
        c = self.cfg
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        t0 = time.perf_counter()
        self.raw_db = self._db(x)
        self._raw_pending = np.concatenate([self._raw_pending, x])
        while self._raw_pending.size >= HOP:
            self.raw_vad.step(self._raw_pending[:HOP])
            self._raw_pending = self._raw_pending[HOP:]
        view = self.adjuster.view(snap, vis_age, c, now, self.calibrated)
        adj = self.adjuster.decide(view, self.vad.speech, self.raw_vad.speech, c, self.iso.base_mix)
        if adj.label != self.adj.label:
            self.counts[adj.label] += 1
        self.adj = adj
        dt = x.size / SR
        self.mix += (adj.mix - self.mix) * (1 - math.exp(-dt / 0.05))  # 50 ms mix smoothing
        y = self.iso.process(x, self.mix)
        self._pending = np.concatenate([self._pending, y])
        while self._pending.size >= HOP:
            self.vad.step(self._pending[:HOP])  # clean VAD on the PureSound output (before correction)
            self._pending = self._pending[HOP:]
        tau = c.xcheck_attack if adj.gain > self.gain else c.xcheck_release
        if y.size:
            a = 1 - math.exp(-1 / (max(tau, 1e-3) * SR))
            g = adj.gain + (self.gain - adj.gain) * (1 - a) ** np.arange(1, y.size + 1)
            self.gain = float(g[-1])
            y = y * g.astype(np.float32)
        self.clean_db = self._db(y)
        self.compute_s += time.perf_counter() - t0
        self.audio_s += x.size / SR
        self._classified_samples[adj.label] += x.size
        return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes(), adj.speech

    def recording_tracks(self) -> tuple[bytes, bytes]:
        """(vi, raw) PCM16 of the last block, sample-aligned with the returned output (PureSound delay
        compensated). Only sent to the browser when the user records there; never stored on the server."""

        def q(a):
            return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()

        return q(self.iso.last_vi), q(self.iso.last_ref)

    def meta(self) -> dict:
        a = self.adj
        classified_samples = sum(self._classified_samples.values())
        return {
            "decision": a.label,
            "why": a.why,
            "counts": dict(self.counts),
            # Lifetime of this AudioStream (model resets do not clear counts or durations).
            # Classifications describe decisions, not ground-truth recognition accuracy.
            "decision_duration_s": {k: n / SR for k, n in self._classified_samples.items()},
            "decision_ratio": {
                k: n / classified_samples if classified_samples else 0.0 for k, n in self._classified_samples.items()
            },
            "classified_audio_s": classified_samples / SR,
            "speech": a.speech,
            "clean_vad": self.vad.speech,
            "raw_vad": self.raw_vad.speech,
            "gain": round(self.gain, 3),
            "mix": round(self.mix, 3),
            "target_track": a.target_track,
            "distance": a.distance,
            "clean_db": round(self.clean_db, 1),
            "raw_db": round(self.raw_db, 1),
            "rtf": round(self.compute_s / self.audio_s, 3) if self.audio_s else None,
            "resets": self.iso.resets,
            "onsets": self.vad.onsets,
            "vad_source": "model_head" if self.iso.vad_heads else "energy_on_enhanced",
            "algo_latency_ms": round((self.iso.delay + self.iso.config.win_length) / SR * 1000, 1),
        }

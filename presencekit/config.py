"""Tracker configuration: :class:`Config` (every tunable parameter with its default), ``RANGES`` (valid range and
type per field, used to clamp values coming from the browser) and the session :class:`State` enum.

The full reference with explanations is generated into docs/configuration.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    MAYBE_AWAY = "MAYBE_AWAY"
    AWAY = "AWAY"
    ENDED = "ENDED"


@dataclass
class Config:
    """Tracker parameters. All coordinates are normalized; times in seconds. See docs/configuration.md."""

    face_roi: tuple = (0.2, 0.05, 0.8, 0.95)  # face ROI (formerly `roi`), normalized
    min_conf: float = 0.4  # detections below are "low confidence" (0.4: tuned for a low robot camera, v0.11.1)
    iou_match: float = 0.3
    cos_match: float = 0.40  # cosine >= this => same person (SFace ~0.363)
    cos_weak: float = 0.20  # with good IoU, this weaker cosine is accepted
    # --- embedding backend ---
    embed_backend: str = (
        "fp32"  # fp32 (SFace, ONNX Runtime) | int8 (official opencv_zoo int8 ONNX) | npu (TFLite + delegate)
    )
    # external TFLite delegate .so (NNAPI / NeuroPilot); empty -> CPU. SECURITY: loading a shared library
    # runs native code, so this is set only server-side (env PRESENCE_NPU_DELEGATE), never over WebSocket.
    npu_delegate_path: str = field(default_factory=lambda: os.environ.get("PRESENCE_NPU_DELEGATE", ""))
    npu_precision: str = "int8"  # npu backend model: int8 (full-integer) | fp16
    cos_match_int8: float = 0.40  # cos_match used with quantized embeddings (int8 / npu int8)
    embed_ema: float = 0.15  # template update rate
    join_seconds: float = 2.0  # continuous time in ROI to become participant
    grace_seconds: float = 15.0  # keep missing tracks this long
    absent_frames: int = 3  # consecutive empty frames before "empty"
    present_frames: int = 2  # consecutive confident frames before "present"
    idle_after: float = 3.0  # silence before ACTIVE -> IDLE
    base_timeout: float = 20.0  # silence timeout when vision is unknown
    present_timeout: float = 15.0  # silence timeout while participants are visible (balanced mode)
    empty_timeout: float = 5.0  # shortened timeout when ROI is empty
    maybe_away_frac: float = 0.5  # fraction of timeout to enter MAYBE_AWAY
    end_after_away: float = 5.0  # AWAY -> ENDED
    new_group_seconds: float = 3.0  # old group missing + new group confirmed this long
    priority_mode: str = "vision"  # vision | balanced | audio
    # --- performance / power (see presencekit/optimizer.py) ---
    motion_gate: bool = True  # skip detection when the (downscaled) frame barely changed
    motion_threshold: float = 0.015  # mean abs gray diff (0..1) below which a frame is "static"
    force_detect_seconds: float = 2.0  # full detection at least this often even if static
    embed_throttle: bool = True  # reuse a face embedding when the face barely moved
    embed_interval: float = 3.0  # recompute each face embedding at least this often
    embed_move_thresh: float = 0.15  # center shift / size change (relative to box) forcing recompute
    adaptive_fps: bool = True  # server recommends client fps from state
    fps_low: float = 0.5  # recommended fps when stable
    fps_high: float = 2.0  # recommended fps when things change / MAYBE_AWAY
    roi_crop: bool = True  # crop ROI (+margin) and downscale before detection
    crop_margin: float = 0.1  # margin around ROI, fraction of frame
    max_width: int = 640  # max width fed to the detector (320 missed faces < ~80 px tall)
    # --- pose / body (optional) ---
    pose_enabled: bool = False  # run the pose detector (MoveNet MultiPose Lightning)
    pose_interval: float = 2.0  # seconds between stable face-matched poses; 0 disables pose throttle
    pose_conf: float = 0.4  # min pose/person confidence
    body_min_conf: float = 0.4  # confidence needed for a body-only (no face) detection to count as confident
    pose_roi: tuple = (0.0, 0.0, 1.0, 1.0)  # body ROI (center of body box), normalized
    roi_rule: str = "either"  # either | face | body : which ROI makes a person count
    turned_away_early: bool = True  # turned away + moving away -> early MAYBE_AWAY
    turned_away_seconds: float = 2.0  # how long the cue must persist
    recede_ratio: float = 0.1  # relative size shrink over ~2 s that counts as "moving away"
    # --- distance (optional) ---
    distance_enabled: bool = False
    camera_hfov: float = 60.0  # horizontal field of view, degrees
    face_width_m: float = 0.15  # average face width
    shoulder_width_m: float = 0.38  # average shoulder width
    distance_calib: float = 1.0  # calibration factor (see calibrate)
    distance_ema: float = 0.3  # weight of the newest estimate
    zone_min: float = 0.4  # conversation zone, meters
    zone_max: float = 2.0
    leave_distance: float = 2.5  # beyond this a visible person counts as left
    # --- tracking (Layer 4) ---
    tracker_mode: str = "legacy"  # legacy (greedy IoU+cosine) | kalman (Kalman + Hungarian + appearance)
    coast_seconds: float = 2.0  # kalman: unmatched tracks coast on the prediction this long
    coast_present_conf: float = 0.4  # kalman: a coasting track counts as present while vis_conf >= this
    gate_mahalanobis: float = 9.21  # kalman: chi-square gate on the predicted center (2 dof, 99%)
    appearance_min_sim: float = 0.6  # kalman: torso-histogram similarity below this => different person (no face)
    appearance_ema: float = 0.2  # kalman: appearance template update rate
    # --- presence model (Layer 2) ---
    presence_model: str = "rules"  # rules (counters + timeouts) | hmm (per-track Bayes filter)
    leave_halflife: float = 1.5  # hmm: with nobody seen, odds of "present" halve every this many seconds
    detect_gain: float = 4.0  # hmm: evidence rate per second for a detection (x confidence)
    lowconf_factor: float = 0.15  # hmm: low-confidence frames decay this fraction of the normal rate
    p_active: float = 0.8  # hmm: group P >= this -> ACTIVE/IDLE (from MAYBE_AWAY/AWAY)
    p_maybe: float = 0.6  # hmm: group P < this -> MAYBE_AWAY
    p_away: float = 0.3  # hmm: group P < this for away_hold_seconds -> AWAY
    away_hold_seconds: float = 1.0
    # --- interaction intent (Layer 3) ---
    intent_model: str = "rules"  # rules (join after join_seconds in ROI) | score (per-track intent score)
    intent_tau: float = 1.0  # EMA time constant (s) of the intent score
    join_threshold: float = 0.6  # score: join when intent >= this for join_seconds
    drop_threshold: float = 0.35  # score: participant -> bystander when intent < this for drop_seconds
    drop_seconds: float = 8.0
    intent_weighted_presence: bool = False  # hmm: weight participant P by intent
    w_facing: float = 0.45
    w_distance: float = 0.1
    w_dwell: float = 0.1
    w_approach: float = 0.05
    w_still: float = 0.15
    w_center: float = 0.15
    speech_join: str = "auto"  # auto (vision: off, else single_candidate) | off | single_candidate | all
    present_max_silence: float = 0.0  # vision mode: cap on silence with people present (0 = off)
    # --- audio (v0.9.0, see docs/audio.md) ---
    audio_pipeline: str = "energy_vad"  # energy_vad (browser VAD) | puresound (server isolation + clean VAD)
    iso_onset_guard: bool = False  # PureSound onset guard (input untouched until 1 s of sustained speech)
    iso_reset_on_change: bool = True  # reset isolation state on NEW_GROUP / session / participant change
    onset_needs_vision: bool = True  # clean speech starts/returns only with a visible near candidate
    onset_entry_only: bool = True  # clean VAD active: a session opens ONLY on clean onset + visible near person
    clean_vad_margin_db: float = 10.0  # clean VAD: dB over the adaptive floor of the enhanced signal
    clean_vad_min_db: float = -50.0  # clean VAD: absolute minimum level (dBFS)
    clean_vad_hang: float = 0.3  # clean VAD hangover (s)
    xcheck_enabled: bool = True  # vision cross-checks the clean result (leak / false reject / false accept)
    near_range: float = 1.0  # m: PureSound pickup range; speakers beyond it are "false accept"
    xcheck_hold: float = 0.75  # s: vision presence/near hold (aligns 1-2 fps vision with 20 ms audio)
    xcheck_stale: float = 2.0  # s: no processed frame for this long -> acoustic only
    xcheck_require_calib: bool = False  # only use distance (false_accept) after a distance calibration
    xcheck_floor_db: float = -40.0  # leak / false_accept output level (<= -60 = mute)
    xcheck_recover_mix: float = 0.3  # false_reject: enhanced share (rest = latency-aligned raw input)
    xcheck_attack: float = 0.01  # s: gain ramp up
    xcheck_release: float = 0.05  # s: gain ramp down
    # --- cloud ASR (optional, external service; docs/asr.md) ---
    asr_provider: str = "none"  # none | assemblyai
    asr_source: str = "clean"  # clean (corrected PureSound output) | raw
    asr_feed: str = "speech"  # speech: only corrected-speech blocks (silence otherwise) | session: all audio
    asr_region: str = "global"  # global | us | eu (AssemblyAI data residency endpoints)
    asr_speech_model: str = "universal-3-6-pro"
    asr_language_codes: str = "zh,en"  # Universal-3.x Pro only; empty = native code-switching
    asr_keyterms: str = ""  # comma-separated, max 100
    asr_format_turns: bool = True  # Universal-Streaming only
    asr_eot_confidence: float = 0.4  # Universal-Streaming only
    asr_min_turn_silence: int = 400  # ms
    asr_max_turn_silence: int = 1280  # ms
    # optional cross-session re-ID ("is this a new user?"), docs/reid.md. Off by default; cloud; needs consent.
    reid_enabled: bool = False
    reid_consent: bool = False  # the operator confirmed users are informed (face crops kept in memory + sent out)
    reid_show_crops: bool = True  # send small display thumbnails of the compared faces to this browser only
    reid_provider: str = "jev"
    reid_model: str = "jev-latest"
    reid_threshold: float = 0.9  # >= -> same person -> resume the previous conversation
    reid_uncertain_low: float = 0.4  # [low, threshold): one more frame, else new conversation
    reid_ttl: float = 600.0  # history retention since last normal ENDED (memory only)
    reid_max_refs: int = 3  # ended conversations, up to three member references each
    reid_min_face: float = 0.06  # min face width (fraction of frame) to use as query/reference
    reid_decision_timeout: float = 10.0  # wall-clock cap including crop collection and remote calls
    reid_face_wait: float = 3.0  # after the first participant appears, wait this long for a usable face
    reid_timeout: float = 2.0  # per HTTP call
    reid_cooldown: float = 5.0  # min seconds between decisions (rapid re-opens -> new, no call)
    reid_cap_per_hour: int = 30
    reid_cap_total: int = 200  # per connection
    reid_cost_per_call: float = 0.0  # user-entered unit price, for the cost estimate only

    @property
    def match_cos(self) -> float:
        """Strong-match cosine threshold for the active embedding backend."""
        quant = self.embed_backend == "int8" or (self.embed_backend == "npu" and self.npu_precision == "int8")
        return self.cos_match_int8 if quant else self.cos_match

    @property
    def roi(self) -> tuple:
        """Backward-compatible alias for :attr:`face_roi`."""
        return self.face_roi

    @roi.setter
    def roi(self, v: tuple) -> None:
        self.face_roi = tuple(v)

    def group(self, name: str) -> ConfigGroup:
        """Grouped view (v0.13), e.g. ``cfg.group("reid").reid_threshold`` is ``cfg.reid_threshold``. See GROUPS."""
        return ConfigGroup(self, name)

    def as_groups(self) -> dict[str, dict]:
        """Nested ``{group: {key: value}}`` copy of every field (v0.13). The flat keys remain the wire format."""
        return {g: {k: getattr(self, k) for k in keys} for g, keys in GROUPS.items()}

    def update(self, d: dict) -> list[str]:
        """Validate + clamp; unknown keys / bad values are skipped. Returns warnings."""
        warnings = []
        d = dict(d or {})
        if "roi" in d:  # migrate legacy key
            v = d.pop("roi")
            d.setdefault("face_roi", v)
        for k, v in d.items():
            if k not in RANGES:
                warnings.append(f"unknown key {k}")
                continue
            lo, hi, typ = RANGES[k]
            try:
                if typ == "text":
                    if not isinstance(v, str):
                        raise ValueError
                    setattr(self, k, v[:hi])
                    continue
                if typ is str:
                    if v not in lo:
                        raise ValueError
                    setattr(self, k, v)
                    continue
                if typ is bool:
                    if isinstance(v, str):
                        if v.lower() not in ("true", "false", "1", "0", "on", "off"):
                            raise ValueError
                        v = v.lower() in ("true", "1", "on")
                    setattr(self, k, bool(v))
                    continue
                if typ is tuple:
                    vals = [min(hi, max(lo, float(x))) for x in v]
                    if len(vals) != 4:
                        raise ValueError
                    x0, y0, x1, y1 = vals
                    x0, x1 = sorted((x0, x1))
                    y0, y1 = sorted((y0, y1))
                    if x1 - x0 < 0.05 or y1 - y0 < 0.05:
                        warnings.append(f"{k} too small; ignored")
                        continue
                    new = (x0, y0, x1, y1)
                else:
                    raw = float(v)
                    if raw != raw:  # NaN
                        raise ValueError
                    new = typ(round(min(hi, max(lo, raw))) if typ is int else min(hi, max(lo, raw)))
                    if new != (typ(raw) if typ is float else round(raw)):
                        warnings.append(f"{k} clamped to {new}")
            except (TypeError, ValueError, OverflowError):
                warnings.append(f"invalid value for {k}")
                continue
            setattr(self, k, new)
        if self.zone_min > self.zone_max:
            self.zone_min = self.zone_max
            warnings.append("zone_min lowered to zone_max")
        if self.fps_low > self.fps_high:
            self.fps_low = self.fps_high
            warnings.append("fps_low lowered to fps_high")
        if not (self.p_away <= self.p_maybe <= self.p_active):
            self.p_away = min(self.p_away, self.p_maybe, self.p_active)
            self.p_active = max(self.p_active, self.p_maybe)
            warnings.append("presence thresholds reordered: p_away <= p_maybe <= p_active")
        if self.drop_threshold > self.join_threshold:
            self.drop_threshold = self.join_threshold
            warnings.append("drop_threshold lowered to join_threshold")
        if self.cos_weak > self.cos_match:
            self.cos_weak = self.cos_match
            warnings.append("cos_weak lowered to cos_match")
        return warnings


# (min, max, type) for every Config field
RANGES = {
    "face_roi": (0.0, 1.0, tuple),
    "pose_roi": (0.0, 1.0, tuple),
    "min_conf": (0.0, 1.0, float),
    "iou_match": (0.0, 1.0, float),
    "cos_match": (0.0, 1.0, float),
    "cos_weak": (0.0, 1.0, float),
    "embed_backend": (("fp32", "int8", "npu"), None, str),
    "npu_precision": (("int8", "fp16"), None, str),
    "cos_match_int8": (0.0, 1.0, float),
    "embed_ema": (0.0, 1.0, float),
    "join_seconds": (0.0, 30.0, float),
    "grace_seconds": (0.0, 120.0, float),
    "absent_frames": (1, 20, int),
    "present_frames": (1, 20, int),
    "idle_after": (0.0, 60.0, float),
    "base_timeout": (1.0, 300.0, float),
    "present_timeout": (1.0, 600.0, float),
    "empty_timeout": (1.0, 120.0, float),
    "maybe_away_frac": (0.05, 0.95, float),
    "end_after_away": (0.0, 120.0, float),
    "new_group_seconds": (0.0, 60.0, float),
    "present_max_silence": (0.0, 3600.0, float),
    "priority_mode": (("vision", "balanced", "audio"), None, str),
    "roi_rule": (("either", "face", "body"), None, str),
    "pose_enabled": (False, True, bool),
    "pose_conf": (0.05, 1.0, float),
    "pose_interval": (0.0, 2.0, float),
    "body_min_conf": (0.05, 1.0, float),
    "tracker_mode": (("legacy", "kalman"), None, str),
    "coast_seconds": (0.0, 10.0, float),
    "coast_present_conf": (0.0, 1.0, float),
    "gate_mahalanobis": (1.0, 100.0, float),
    "appearance_min_sim": (0.0, 1.0, float),
    "appearance_ema": (0.0, 1.0, float),
    "presence_model": (("rules", "hmm"), None, str),
    "leave_halflife": (0.1, 30.0, float),
    "detect_gain": (0.1, 20.0, float),
    "lowconf_factor": (0.0, 1.0, float),
    "p_active": (0.05, 0.99, float),
    "p_maybe": (0.02, 0.98, float),
    "p_away": (0.01, 0.95, float),
    "away_hold_seconds": (0.0, 30.0, float),
    "intent_model": (("rules", "score"), None, str),
    "intent_tau": (0.1, 20.0, float),
    "join_threshold": (0.05, 0.99, float),
    "drop_threshold": (0.0, 0.95, float),
    "drop_seconds": (0.0, 120.0, float),
    "intent_weighted_presence": (False, True, bool),
    "w_facing": (0.0, 1.0, float),
    "w_distance": (0.0, 1.0, float),
    "w_dwell": (0.0, 1.0, float),
    "w_approach": (0.0, 1.0, float),
    "w_still": (0.0, 1.0, float),
    "w_center": (0.0, 1.0, float),
    "speech_join": (("auto", "off", "single_candidate", "all"), None, str),
    "turned_away_early": (False, True, bool),
    "turned_away_seconds": (0.0, 30.0, float),
    "recede_ratio": (0.01, 0.9, float),
    "distance_enabled": (False, True, bool),
    "camera_hfov": (20.0, 170.0, float),
    "face_width_m": (0.08, 0.3, float),
    "shoulder_width_m": (0.2, 0.6, float),
    "distance_calib": (0.2, 5.0, float),
    "distance_ema": (0.01, 1.0, float),
    "zone_min": (0.0, 10.0, float),
    "zone_max": (0.1, 20.0, float),
    "leave_distance": (0.2, 30.0, float),
    "motion_gate": (False, True, bool),
    "motion_threshold": (0.0, 0.2, float),
    "force_detect_seconds": (0.5, 30.0, float),
    "embed_throttle": (False, True, bool),
    "embed_interval": (0.0, 30.0, float),
    "embed_move_thresh": (0.0, 1.0, float),
    "adaptive_fps": (False, True, bool),
    "fps_low": (0.2, 5.0, float),
    "fps_high": (0.2, 5.0, float),
    "roi_crop": (False, True, bool),
    "crop_margin": (0.0, 0.5, float),
    "max_width": (96, 1920, int),
    "audio_pipeline": (("energy_vad", "puresound"), None, str),
    "iso_onset_guard": (False, True, bool),
    "iso_reset_on_change": (False, True, bool),
    "onset_needs_vision": (False, True, bool),
    "onset_entry_only": (False, True, bool),
    "clean_vad_margin_db": (3.0, 30.0, float),
    "clean_vad_min_db": (-80.0, -10.0, float),
    "clean_vad_hang": (0.0, 2.0, float),
    "xcheck_enabled": (False, True, bool),
    "near_range": (0.2, 10.0, float),
    "xcheck_hold": (0.0, 5.0, float),
    "xcheck_stale": (0.2, 30.0, float),
    "xcheck_require_calib": (False, True, bool),
    "xcheck_floor_db": (-80.0, 0.0, float),
    "xcheck_recover_mix": (0.0, 0.9, float),
    "xcheck_attack": (0.001, 0.5, float),
    "xcheck_release": (0.005, 2.0, float),
    "asr_provider": (("none", "assemblyai"), None, str),
    "asr_source": (("clean", "raw"), None, str),
    "asr_feed": (("speech", "session"), None, str),
    "asr_region": (("global", "us", "eu"), None, str),
    "asr_speech_model": (
        ("universal-3-6-pro", "universal-3-5-pro", "universal-streaming-english", "universal-streaming-multilingual"),
        None,
        str,
    ),
    "asr_language_codes": (None, 200, "text"),
    "asr_keyterms": (None, 2000, "text"),
    "asr_format_turns": (False, True, bool),
    "asr_eot_confidence": (0.0, 1.0, float),
    "asr_min_turn_silence": (50, 10000, int),
    "asr_max_turn_silence": (100, 10000, int),
    "reid_enabled": (False, True, bool),
    "reid_consent": (False, True, bool),
    "reid_show_crops": (False, True, bool),
    "reid_provider": (("jev",), None, str),
    "reid_model": (("jev-latest", "jev-1.13.0"), None, str),
    "reid_threshold": (0.5, 0.99, float),
    "reid_uncertain_low": (0.0, 0.95, float),
    "reid_ttl": (30.0, 600.0, float),
    "reid_max_refs": (1, 5, int),
    "reid_min_face": (0.02, 0.5, float),
    "reid_decision_timeout": (0.5, 30.0, float),
    "reid_face_wait": (0.5, 15.0, float),
    "reid_timeout": (0.3, 10.0, float),
    "reid_cooldown": (0.0, 120.0, float),
    "reid_cap_per_hour": (1, 600, int),
    "reid_cap_total": (1, 10000, int),
    "reid_cost_per_call": (0.0, 10.0, float),
}


# --- v0.13 grouped view --------------------------------------------------------------------------------------------
# Every Config field belongs to exactly one group. The flat keys above stay the storage, the wire format
# (/api/config, WebSocket ``config`` messages, presets) and the documented reference. Deprecation of flat-key access
# in favor of the grouped view is planned for the next minor version; v0.13 only adds the view.
# Server-process settings (token, host, port, model dir, ...) are environment variables, not Config fields.
GROUPS: dict[str, tuple[str, ...]] = {
    "core": (
        "face_roi",
        "pose_roi",
        "roi_rule",
        "min_conf",
        "body_min_conf",
        "priority_mode",
        "join_seconds",
        "grace_seconds",
        "absent_frames",
        "present_frames",
        "idle_after",
        "base_timeout",
        "present_timeout",
        "empty_timeout",
        "maybe_away_frac",
        "end_after_away",
        "new_group_seconds",
        "present_max_silence",
        "speech_join",
        "onset_needs_vision",
        "onset_entry_only",
    ),
    "tracker": (
        "iou_match",
        "cos_match",
        "cos_weak",
        "cos_match_int8",
        "embed_ema",
        "tracker_mode",
        "coast_seconds",
        "coast_present_conf",
        "gate_mahalanobis",
        "appearance_min_sim",
        "appearance_ema",
    ),
    "presence": (
        "presence_model",
        "leave_halflife",
        "detect_gain",
        "lowconf_factor",
        "p_active",
        "p_maybe",
        "p_away",
        "away_hold_seconds",
    ),
    "intent": (
        "intent_model",
        "intent_tau",
        "join_threshold",
        "drop_threshold",
        "drop_seconds",
        "intent_weighted_presence",
        "w_facing",
        "w_distance",
        "w_dwell",
        "w_approach",
        "w_still",
        "w_center",
    ),
    "vision": ("embed_backend", "npu_delegate_path", "npu_precision"),
    "pose": (
        "pose_enabled",
        "pose_interval",
        "pose_conf",
        "turned_away_early",
        "turned_away_seconds",
        "recede_ratio",
        "distance_enabled",
        "camera_hfov",
        "face_width_m",
        "shoulder_width_m",
        "distance_calib",
        "distance_ema",
        "zone_min",
        "zone_max",
        "leave_distance",
    ),
    "perf": (
        "motion_gate",
        "motion_threshold",
        "force_detect_seconds",
        "embed_throttle",
        "embed_interval",
        "embed_move_thresh",
        "adaptive_fps",
        "fps_low",
        "fps_high",
        "roi_crop",
        "crop_margin",
        "max_width",
    ),
    "audio": (
        "audio_pipeline",
        "iso_onset_guard",
        "iso_reset_on_change",
        "clean_vad_margin_db",
        "clean_vad_min_db",
        "clean_vad_hang",
        "xcheck_enabled",
        "near_range",
        "xcheck_hold",
        "xcheck_stale",
        "xcheck_require_calib",
        "xcheck_floor_db",
        "xcheck_recover_mix",
        "xcheck_attack",
        "xcheck_release",
    ),
    "asr": (
        "asr_provider",
        "asr_source",
        "asr_feed",
        "asr_region",
        "asr_speech_model",
        "asr_language_codes",
        "asr_keyterms",
        "asr_format_turns",
        "asr_eot_confidence",
        "asr_min_turn_silence",
        "asr_max_turn_silence",
    ),
    "reid": (
        "reid_enabled",
        "reid_consent",
        "reid_show_crops",
        "reid_provider",
        "reid_model",
        "reid_threshold",
        "reid_uncertain_low",
        "reid_ttl",
        "reid_max_refs",
        "reid_min_face",
        "reid_decision_timeout",
        "reid_face_wait",
        "reid_timeout",
        "reid_cooldown",
        "reid_cap_per_hour",
        "reid_cap_total",
        "reid_cost_per_call",
    ),
}
_GROUP_OF = {k: g for g, keys in GROUPS.items() for k in keys}


def group_of(key: str) -> str | None:
    """Group name of a flat Config key (``roi`` is the legacy alias of ``face_roi``)."""
    return _GROUP_OF.get("face_roi" if key == "roi" else key)


class ConfigGroup:
    """Live view of one group of a :class:`Config`: attribute reads/writes go to the flat fields."""

    __slots__ = ("_cfg", "_name", "_keys")

    def __init__(self, cfg: Config, name: str):
        if name not in GROUPS:
            raise KeyError(f"unknown config group {name}")
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_keys", GROUPS[name])

    def __getattr__(self, k):
        if k in self._keys:
            return getattr(self._cfg, k)
        raise AttributeError(f"config group {self._name} has no key {k}")

    def __setattr__(self, k, v):
        if k not in self._keys:
            raise AttributeError(f"config group {self._name} has no key {k}")
        setattr(self._cfg, k, v)

    def __dir__(self):
        return list(self._keys)

    def __repr__(self):
        return f"ConfigGroup({self._name}, {self.as_dict()})"

    @property
    def name(self) -> str:
        return self._name

    def keys(self) -> tuple[str, ...]:
        return self._keys

    def as_dict(self) -> dict:
        return {k: getattr(self._cfg, k) for k in self._keys}

    def update(self, d: dict) -> list[str]:
        """Validated update restricted to this group (same rules and warnings as :meth:`Config.update`)."""
        d = dict(d or {})
        outside = [k for k in d if k not in self._keys]
        for k in outside:
            d.pop(k)
        return [f"unknown key {k}" for k in outside] + self._cfg.update(d)

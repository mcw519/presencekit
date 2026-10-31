"""Embeddable presence SDK: camera frames (and optionally audio) in, presence state and events out.

:class:`PresenceEngine` composes the layers for one camera/microphone stream with no web framework, no asyncio and
no threads of its own. The host owns capture and timing and calls in::

    from presence_demo.sdk import PresenceEngine

    engine = PresenceEngine()                          # YuNet + SFace on ONNX Runtime (downloaded on first use)
    step = engine.process_frame(bgr, speaking=vad)     # or engine.process_jpeg(jpeg_bytes, speaking=vad)
    for ev in step.events:
        print(ev.kind, ev.reason)                      # e.g. "transition", "speech detected"
    step.snapshot["state"]                             # ACTIVE | IDLE | MAYBE_AWAY | AWAY | ENDED

Speech comes from the host's own VAD (``speaking=`` / :meth:`PresenceEngine.set_speaking`), or, with the
``puresound`` extra and ``audio_pipeline="puresound"``, from :meth:`PresenceEngine.process_audio` (PureSound voice
isolation + clean VAD + vision cross-check, as in the web demo). Call :meth:`PresenceEngine.tick` when no input
arrives so timeouts keep running. Every neural model runs on ONNX Runtime; pass
:class:`~presence_demo.inference.OrtOptions` to pick execution providers and threads.

Importing this module loads no web framework, starts no thread and reads no model.
PRIVACY: frames and audio stay in memory; nothing is written to disk except downloaded model files.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .body import calibrate
from .config import Config, State
from .core import Observation, PresenceCore, Step
from .detectors.base import PoseDetector
from .inference import OrtOptions, loaded_models
from .optimizer import FrameProcessor
from .vision import VisionPipeline, decode_jpeg, load_pose_detector


class PresenceEngine:
    """Presence detection for one stream. Methods are serialized by an internal lock, so capture and audio
    callbacks may call in from different threads. Separate engines share model sessions, not state.

    ``config``: tracker parameters (:class:`~presence_demo.config.Config`; change later with :meth:`configure`).
    ``ort``: how the ONNX models run (default ``OrtOptions.from_env()``). ``pipeline``: a prebuilt
    :class:`~presence_demo.vision.VisionPipeline` (e.g. with a custom detector). ``pose``: ``False`` disables the
    pose model, a :class:`~presence_demo.detectors.PoseDetector` replaces it; by default MoveNet is loaded on the
    first frame with ``config.pose_enabled``. ``allow_download``: fetch missing downloadable models.
    ``clock``: monotonic seconds, used when a call passes no ``now``.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        ort: OrtOptions | None = None,
        pipeline: VisionPipeline | None = None,
        pose: PoseDetector | bool | None = None,
        allow_download: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.options = ort or OrtOptions.from_env()
        self.clock = clock
        self.pipeline = pipeline or VisionPipeline(allow_download=allow_download, options=self.options)
        if pose is False:
            loader = None
        elif pose is None or pose is True:
            loader = lambda: load_pose_detector(allow_download, self.options)  # noqa: E731
        else:
            loader = lambda: pose  # noqa: E731
        self.frames = FrameProcessor(self.pipeline, clock=clock, pose_loader=loader)
        self.core = PresenceCore(config or Config(), clock=clock)
        self.timings: dict = {}  # ms per stage of the last frame (detect, embed, pose, ...)
        self._audio = None
        self._calibrated = False
        self._last_frame: float | None = None
        self._lock = threading.RLock()

    # --- state ------------------------------------------------------------------------------------------------------

    @property
    def config(self) -> Config:
        return self.core.cfg

    @property
    def state(self) -> State:
        return self.core.state

    def snapshot(self, now: float | None = None) -> dict:
        with self._lock:
            return self.core.snapshot(now)

    def configure(self, changes: dict[str, Any]) -> list[str]:
        """Apply settings (flat keys, validated and clamped like the web settings). Returns warnings."""
        with self._lock:
            return self.config.update(changes)

    # --- input ------------------------------------------------------------------------------------------------------

    def process_frame(self, image: np.ndarray, speaking: bool | None = None, now: float | None = None) -> Step:
        """One BGR camera frame (``uint8 HxWx3``, unmirrored). ``speaking`` is the host VAD at this frame (None:
        unchanged; ignored while :meth:`process_audio` drives speech)."""
        with self._lock:
            now = self.clock() if now is None else now
            cfg = self.config
            self.frames.prepare(cfg)
            stable = self.state not in (State.MAYBE_AWAY, State.AWAY)
            dets, self.timings = self.frames.process(image, cfg, stable)
            tr = self.core.tracker
            tr.speech_source = "clean" if self._audio is not None else "browser"
            step = self.core.update(Observation(dets, None if self._audio is not None else speaking), now)
            self._last_frame = now
            return step

    def process_jpeg(self, data: bytes, speaking: bool | None = None, now: float | None = None) -> Step | None:
        """Like :meth:`process_frame` for JPEG bytes (decoded in memory). None when the bytes do not decode."""
        img, _ = decode_jpeg(data)
        return None if img is None else self.process_frame(img, speaking, now)

    def set_speaking(self, speaking: bool, now: float | None = None) -> Step:
        """Host VAD changed (between frames)."""
        with self._lock:
            self.core.tracker.speech_source = "browser"
            return self.core.update(Observation(speaking=bool(speaking)), now)

    def tick(self, now: float | None = None) -> Step:
        """Advance timers without new input (call about every 0.5 s while idle)."""
        with self._lock:
            return self.core.update(Observation(), now)

    def process_audio(self, pcm16: bytes, now: float | None = None) -> tuple[bytes, Step | None]:
        """One chunk of 16 kHz mono PCM16 (20 ms recommended) through PureSound isolation and the clean VAD.

        Returns the cleaned PCM16 and the :class:`Step` when the speech decision changed (else None). Needs the
        ``puresound`` extra, the PureSound model files and ``config.audio_pipeline == "puresound"``."""
        from . import audio

        with self._lock:
            cfg = self.config
            if cfg.audio_pipeline != "puresound":
                raise RuntimeError('process_audio needs config.audio_pipeline == "puresound"')
            if self._audio is None:
                ok, why = audio.puresound_available()
                if not ok:
                    raise RuntimeError(why)
                self._audio = audio.AudioStream(cfg)
            a = self._audio
            a.cfg, a.calibrated = cfg, self._calibrated
            now = self.clock() if now is None else now
            snap = self.core.snapshot(now)
            a.on_vision(snap, now)
            vis_age = math.inf if self._last_frame is None else max(0.0, now - self._last_frame)
            out, speech = a.process(pcm16, snap, now, vis_age)
            tr = self.core.tracker
            tr.speech_source = "clean"
            step = self.core.update(Observation(speaking=speech), now) if speech != tr.speaking else None
            return out, step

    # --- control ----------------------------------------------------------------------------------------------------

    def calibrate_distance(self, meters: float) -> list[str]:
        """Calibrate distance so the closest visible person is ``meters`` away. Returns warnings/result."""
        with self._lock:
            warns = calibrate(self.core.tracker, meters)
            self._calibrated = self._calibrated or any(w.startswith("calibrated") for w in warns)
            return warns

    def reset(self) -> None:
        """Forget tracks and session state (same settings and models)."""
        with self._lock:
            self.core = PresenceCore(self.config, clock=self.clock)
            self.frames.reset()
            if self._audio is not None:
                self._audio.iso.reset()
            self._last_frame = None

    def close(self) -> None:
        """Release the per-stream audio state. Model sessions are shared per process and stay cached."""
        with self._lock:
            self._audio = None

    def info(self) -> dict:
        """Pipeline, pose and ONNX Runtime session details (diagnostics)."""
        pose = self.frames._pose
        return {
            **self.pipeline.info(),
            "pose": pose.name if pose else ("unavailable" if self.frames.pose_available is False else "not loaded"),
            "pose_accelerator": getattr(pose, "accelerator", None) if pose else None,
            "audio": "puresound" if self._audio is not None else None,
            "models": loaded_models(),
        }

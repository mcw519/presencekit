"""Process-wide dependencies of the presence runtime, gathered in one explicit object.

v0.13 keeps ownership where it was: the objects are the existing module singletons
(:data:`presencekit.runtime.continuity.PAGES`, :data:`~presencekit.runtime.resources.VISION_LOCK`,
:data:`~presencekit.runtime.resources.AUDIO_POOL`, the lazily loaded vision pipeline and pose detector, the
in-memory re-ID stores). :class:`AppContext` only *references* them so a :class:`PresenceRuntime` receives its
dependencies explicitly instead of importing them. The web layer builds one per connection from its own module
globals (so a replacement assigned on ``presencekit.web.session`` is honored, exactly as in v0.12).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

from .continuity import PageRegistry


@dataclass(frozen=True)
class AppContext:
    """Explicit dependency bundle for one :class:`~presencekit.runtime.session.PresenceRuntime`."""

    pages: PageRegistry
    vision_lock: threading.Lock
    audio_pool: Executor
    pipeline: Callable[[], Any]  # -> VisionPipeline (blocking; loads once per process)
    pose_detector: Callable[[], Any]  # -> PoseDetector | None (loads once per process)
    pose_cache: dict
    vision_job: Callable[..., Any]  # (proc, jpeg, cfg, stable, conn) -> (detections, timings) | None
    calibrate: Callable[..., list[str]]  # (tracker, meters) -> warnings
    audio_stream: Callable[..., Any]  # Config -> AudioStream
    puresound_available: Callable[[], tuple[bool, str | None]]

    @classmethod
    def default(cls) -> AppContext:
        """The process singletons as currently bound in their owning modules."""
        from .. import audio
        from . import continuity, resources

        return cls(
            pages=continuity.PAGES,
            vision_lock=resources.VISION_LOCK,
            audio_pool=resources.AUDIO_POOL,
            pipeline=resources.pipeline,
            pose_detector=resources.pose_detector,
            pose_cache=resources._pose_cache,
            vision_job=resources._vision_job,
            calibrate=resources.calibrate,
            audio_stream=audio.AudioStream,
            puresound_available=audio.puresound_available,
        )

    @staticmethod
    def reid_key_status() -> dict:
        """Whether a re-ID key is configured (never the key itself)."""
        from ..identity import key_status

        return key_status()

    @staticmethod
    def wipe_references() -> int:
        """Drop every in-memory re-ID reference in the process (server-side wipe)."""
        from ..identity import wipe_all

        return wipe_all()

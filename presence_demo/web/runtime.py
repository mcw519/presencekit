"""v0.12 location of the process-wide shared resources; they now live in :mod:`presence_demo.runtime.resources`.

Read-only re-exports: assign module state (e.g. ``_pipeline``) on ``presence_demo.runtime.resources``.
"""

from ..runtime.resources import (  # noqa: F401  (re-exported)
    _PIPE_LOCK,
    AUDIO_POOL,
    TOKEN,
    VISION_LOCK,
    VISION_LOCK_TIMEOUT,
    VisionBusy,
    _pose_cache,
    _vision_job,
    calibrate,
    pipeline,
    pose_detector,
)

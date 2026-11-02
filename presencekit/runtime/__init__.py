"""Presence runtime: per-connection orchestration without any web framework.

* :class:`PresenceRuntime` one client connection (tracker, vision/audio workers, ASR gate, watchdog, re-ID
  driver, control messages); transport-neutral, it talks to an outbox object
* :class:`AppContext` the process-wide dependencies it receives explicitly
* :mod:`.protocol` binary/JSON wire protocol (``CLIENT_MESSAGE_TYPES``, video framing)
* :mod:`.continuity` page continuity (``PageRegistry``, ``ObservationClock``)
* :mod:`.resources` process singletons (vision pipeline, pose detector, vision lock, audio pool)
* :class:`FrameProcessor` per-connection frame optimizer (defined in :mod:`presencekit.optimizer`)

Layering: may import ``core``, ``detectors``, ``identity`` and the shared modules; never ``fastapi``,
``starlette``, ``uvicorn`` or ``presencekit.web`` / ``presencekit.server``.
"""

from ..optimizer import FrameProcessor
from .alignment import VisionTiming, client_timestamp, decision_timing
from .context import AppContext
from .continuity import PAGES, ObservationClock, PageRegistry, PageState
from .protocol import CLIENT_MESSAGE_TYPES, ClientMessageType, binary_frame, legacy_frame
from .session import PresenceRuntime

__all__ = [
    "CLIENT_MESSAGE_TYPES",
    "PAGES",
    "AppContext",
    "ClientMessageType",
    "FrameProcessor",
    "ObservationClock",
    "PageRegistry",
    "PageState",
    "PresenceRuntime",
    "VisionTiming",
    "binary_frame",
    "client_timestamp",
    "decision_timing",
    "legacy_frame",
]

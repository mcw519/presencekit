"""Optional cross-session re-identification: "is this a new user?" (``reid_enabled``).

When the presence tracker opens a NEW session, the first good face of that session is compared against the
reference face(s) of the previous session(s) by a pluggable cloud backend. Same person -> the new session
*resumes* the previous ``conversation_id``; different / uncertain / any failure -> a new conversation. There
are NO calls inside a session, and none at all when no unexpired reference exists.

PRIVACY (documented exception to "the server never stores images", see docs/reid.md):
* Off by default; needs ``reid_enabled`` AND ``reid_consent``.
* Reference crops (up to three participant faces per conversation, 320x320 JPEG) are memory-only.
  ``reid_max_refs`` caps inactive conversations; ``reid_ttl`` starts at their last normal ENDED.
  The transport may retain this store across same-page reconnects. Expiry, consent withdrawal, reset,
  page release and shutdown wipe references; interrupt() preserves active references for reconnect.
  Never written to disk, never logged, never sent back to the browser.
* With ``reid_show_crops`` (default on) a 112 px thumbnail of the probe and compared reference is sent to this
  connection's browser only for display (``reid_view``); never written to disk, logs or diagnostic events.
* Only face crops (never whole frames) are sent to the backend. The API key is memory-only (env
  ``TYPESAFE_API_KEY`` or set from the UI) and never logged or echoed.

v0.13: the implementation lives in the :mod:`presence_demo.identity` layer; this module re-exports
every v0.12 name (``CrossSessionReid`` is :class:`presence_demo.identity.ContinuationDriver`).
"""

from __future__ import annotations

from .identity.crops import CROP_MAX, THUMB_PX, face_crop, face_quality, thumb
from .identity.decision import Decision, MatchTally, decide
from .identity.driver import ContinuationDriver
from .identity.guard import BREAKER_ERRORS, BREAKER_PAUSE, CallGuard
from .identity.keys import _key, key_status, set_key
from .identity.matcher import JEV_URL, QUESTION, CloudReid, IdentityMatcher, JevClient, JevMatcher, ReidError
from .identity.store import _STORES, Ref, ReferenceStore, wipe_all

CrossSessionReid = ContinuationDriver

__all__ = [
    "BREAKER_ERRORS",
    "BREAKER_PAUSE",
    "CROP_MAX",
    "JEV_URL",
    "QUESTION",
    "THUMB_PX",
    "_STORES",
    "_key",
    "CallGuard",
    "CloudReid",
    "ContinuationDriver",
    "CrossSessionReid",
    "Decision",
    "IdentityMatcher",
    "JevClient",
    "JevMatcher",
    "MatchTally",
    "Ref",
    "ReferenceStore",
    "ReidError",
    "decide",
    "face_crop",
    "face_quality",
    "key_status",
    "set_key",
    "thumb",
    "wipe_all",
]

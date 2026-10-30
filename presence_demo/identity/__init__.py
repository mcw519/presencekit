"""Identity layer (v0.13): cross-session continuation ("is this the same user?").

* :class:`IdentityMatcher` protocol, shipped backend :class:`JevMatcher` (``JevClient``)
* :class:`ReferenceStore` memory-only reference crops, :func:`wipe_all`
* :class:`CallGuard` call budget / cooldown / breaker, pure :func:`decide`
* :class:`ContinuationDriver` per-page driver (v0.12 ``presence_demo.reid.CrossSessionReid``)

Never imports the web framework or the web/runtime layers. ``presence_demo.reid`` re-exports every v0.12 name.
"""

from .crops import CROP_MAX, THUMB_PX, face_crop, face_quality, thumb
from .decision import Decision, MatchTally, decide
from .driver import ContinuationDriver
from .guard import BREAKER_ERRORS, BREAKER_PAUSE, CallGuard
from .keys import key_status, set_key
from .matcher import JEV_URL, QUESTION, CloudReid, IdentityMatcher, JevClient, JevMatcher, ReidError
from .store import Ref, ReferenceStore, wipe_all

__all__ = [
    "BREAKER_ERRORS",
    "BREAKER_PAUSE",
    "CROP_MAX",
    "JEV_URL",
    "QUESTION",
    "THUMB_PX",
    "CallGuard",
    "CloudReid",
    "ContinuationDriver",
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

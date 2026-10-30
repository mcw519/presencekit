"""v0.12 location of page continuity; it now lives in :mod:`presence_demo.runtime.continuity`.

``PAGES`` here is the same registry object; callers that look it up at call time (``server``) honor a
replacement assigned on this module.
"""

from ..runtime.continuity import (  # noqa: F401  (re-exported)
    PAGE_TOKEN,
    PAGE_TTL,
    PAGES,
    ObservationClock,
    PageRegistry,
    PageState,
)

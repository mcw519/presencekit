"""``PresenceCore``: the typed entry point of the core layer.

``PresenceCore.update(Observation, now) -> Step`` wraps a :class:`PresenceTracker` without changing its behavior:
a frame observation is ``tracker.update``, a speech-only observation is ``tracker.set_speech`` and an empty
observation is ``tracker.tick``. ``Step.events`` are exactly the tracker log entries appended during the call, as
:class:`PresenceEvent` values; ``Step.snapshot`` is the unchanged legacy snapshot dict.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..config import Config, State
from .tracker import PresenceTracker
from .types import Observation, PresenceEvent, Step


class PresenceCore:
    def __init__(
        self,
        config: Config | None = None,
        clock: Callable[[], float] = time.monotonic,
        tracker: PresenceTracker | None = None,
    ):
        self.tracker = tracker if tracker is not None else PresenceTracker(config, clock)

    @property
    def cfg(self) -> Config:
        return self.tracker.cfg

    @property
    def state(self) -> State:
        return self.tracker.state

    def update(self, obs: Observation, now: float | None = None) -> Step:
        tr = self.tracker
        before = getattr(tr, "event_seq", 0)
        if obs.detections is not None:
            snap = tr.update(obs.detections, obs.speaking, now)
        elif obs.speaking is not None:
            snap = tr.set_speech(obs.speaking, now)
        else:
            snap = tr.tick(now)
        n = min(getattr(tr, "event_seq", 0) - before, len(tr.log))
        events = tuple(PresenceEvent.from_dict(e) for e in tr.log[len(tr.log) - n :]) if n > 0 else ()
        return Step(events=events, snapshot=snap)

    def snapshot(self, now: float | None = None) -> dict:
        return self.tracker.snapshot(now)

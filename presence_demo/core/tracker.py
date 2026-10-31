"""Pure-python PresenceTracker (no OpenCV dependency). Only numpy is required.

Coordinates are normalized [0,1] (x0, y0, x1, y1). The implementation is composed of
:class:`~presence_demo.core.tracking.TrackingMixin` (association + intent) and
:class:`~presence_demo.core.state_machine.PresenceStateMachineMixin` (presence + session states).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..config import Config, State
from .geometry import _norm
from .state_machine import PresenceStateMachineMixin
from .tracking import TrackingMixin
from .tracks import Detection


class PresenceTracker(TrackingMixin, PresenceStateMachineMixin):
    """Fuses per-frame face detections with VAD speech state into a presence state machine.

    Feed it with :meth:`update` (one call per processed frame) and :meth:`set_speech`
    (on VAD changes); call :meth:`tick` periodically. The clock is injectable for tests.
    """

    def __init__(self, config: Config | None = None, clock: Callable[[], float] = time.monotonic):
        self.cfg = config or Config()
        self.clock = clock
        self.log: list[dict] = []
        self._next_id = 1
        self.speech_source = "browser"  # browser (energy VAD) | clean (PureSound clean VAD, set by server)
        self._last_starter_t = -1e9  # last frame with a confident, near person in the ROI
        self._reset_session(self.clock(), initial=True)

    def set_speech(self, speaking: bool, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        onset = bool(speaking) and not self.speaking
        self.speaking = bool(speaking)
        if speaking:
            self.last_speech = now
            if self.state == State.ENDED and self._entry_only():
                if onset and now - self._last_starter_t <= 1.5:
                    keep = dict(self.tracks)
                    self._start(now, "entry: clean speech onset + visible near person")
                    self.tracks = keep
            elif self.state == State.ENDED and self._onset_ok(now):
                if self.cfg.priority_mode != "vision":
                    self._start(now, "speech detected" if self.speech_source != "clean" else "clean speech onset")
                elif onset and self.speech_source == "clean" and now - self._last_starter_t <= 1.5:
                    keep = dict(self.tracks)
                    self._start(now, "clean speech onset + visible candidate")
                    self.tracks = keep
        self._evaluate(now)
        return self.snapshot(now)

    def update(self, detections: list[Detection], speaking: bool | None = None, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        if speaking is not None:
            self.speaking = bool(speaking)
            if speaking:
                self.last_speech = now
        dets = [d for d in detections if self.det_in_roi(d)]
        for d in dets:
            d.embedding = _norm(d.embedding)
        confident = [d for d in dets if d.conf >= self._min_conf(d)]
        low = [d for d in dets if d.conf < self._min_conf(d)]
        starters = [d for d in confident if not self._det_far(d)]
        if starters:
            self._last_starter_t = now

        score = self.cfg.intent_model == "score"
        if self.state == State.ENDED and self._restart_needs_fresh and (not starters or self.speaking):
            # ROI seen without anyone confident (a later arrival is new), or speech (the person engages again)
            self._restart_absent_seen = True
        vision_start_ok = (not self._restart_needs_fresh or self._restart_absent_seen) and not self._entry_only()
        if self.state == State.ENDED and (
            (starters and not score and vision_start_ok)
            or (self.speaking and self.cfg.priority_mode != "vision" and self._onset_ok(now) and not self._entry_only())
        ):
            self._start(now, "person in ROI" if starters and not score else "speech detected")

        _t0 = time.perf_counter()
        self._intent_ms = 0.0
        if self.cfg.tracker_mode == "kalman":
            self._match_kalman(confident, now)
        else:
            self._match(confident, now)
        _t1 = time.perf_counter()
        if self.state == State.ENDED and score and vision_start_ok and any(t.participant for t in self.tracks.values()):
            # score mode: a session starts only when somebody shows interaction intent (not any passer-by)
            keep = dict(self.tracks)
            self._start(now, "intent from #" + ",".join(str(t.id) for t in keep.values() if t.participant))
            self.tracks = keep
            for t in keep.values():
                t.logodds = max(t.logodds, 0.0)
        near = any(t.visible and not t.far and t.visible_kind == "detected" for t in self.tracks.values())
        coasting = any(t.visible and not t.far and t.visible_kind == "predicted" for t in self.tracks.values())
        if confident and not near and not self._far_logged:
            self._event(now, "far", f"visible people beyond leave_distance {self.cfg.leave_distance:.1f} m")
        self._far_logged = bool(confident) and not near
        self._frame_confident = bool(confident)
        self._update_away_cue(now)
        if self.cfg.presence_model == "hmm":
            self._hmm_update(now, bool(low) and not confident)
        else:
            self._update_presence(now, near, bool(low), coasting)
        if self.state != State.ENDED:
            self._check_new_group(now)
        self._evaluate(now)
        _t2 = time.perf_counter()
        # per-stage timings (ms) of the last update, for benchmarks/stats; intent runs inside matching
        self.timings = {
            "tracker": (_t1 - _t0) * 1000 - self._intent_ms,
            "intent": self._intent_ms,
            "presence": (_t2 - _t1) * 1000,
        }
        return self.snapshot(now)

    def tick(self, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        self._evaluate(now)
        return self.snapshot(now)

    def snapshot(self, now=None) -> dict:
        now = self.clock() if now is None else now
        return {
            "state": self.state.value,
            "session": self.session,
            "speaking": self.speaking,
            "silence": 0.0 if self.speaking or self.state == State.ENDED else round(now - self.last_speech, 1),
            "path": getattr(self, "path", "ended"),
            "priority_mode": self.cfg.priority_mode,
            "present": self.present,
            "empty": self.empty,
            "timeout": self.timeout,
            "elapsed": round(getattr(self, "elapsed", 0.0), 1),
            "in_state": 0.0 if self.state == State.ENDED else round(now - self.state_since, 1),
            "roi": list(self.cfg.roi),  # legacy alias of face_roi
            "face_roi": list(self.cfg.face_roi),
            "pose_roi": list(self.cfg.pose_roi),
            "tracks": [
                {
                    "id": t.id,
                    "bbox": [round(x, 4) for x in t.bbox],
                    "participant": t.participant,
                    "visible": t.visible,
                    "similarity": round(t.similarity, 3),
                    "visible_kind": t.visible_kind,
                    "vis_conf": round(t.vis_conf, 2),
                    "p": round(t.p, 3) if self.cfg.presence_model == "hmm" else None,
                    "intent": None if t.intent is None else round(t.intent, 3),
                    "intent_cues": t.intent_cues if self.cfg.intent_model == "score" else None,
                    "role": "participant" if t.participant else "bystander",
                    "dwell_seconds": round(now - t.first_seen, 1),
                    "velocity": [round(v, 3) for v in t.kf.velocity] if t.kf is not None else None,
                    "conf": round(t.conf, 2),
                    "missing": round(now - t.last_seen, 1),
                    "has_face": t.has_face,
                    "body": [round(x, 4) for x in t.body.bbox] if t.body is not None else None,
                    "keypoints": {
                        k: [round(v[0], 4), round(v[1], 4), round(v[2], 2)] for k, v in t.body.keypoints.items()
                    }
                    if t.body is not None
                    else None,
                    "facing": t.facing,
                    "facing_score": t.facing_score,
                    "distance": round(t.distance, 2) if t.distance is not None and self.cfg.distance_enabled else None,
                    "trend": t.trend,
                    "far": t.far,
                    "zone": self.zone(t),
                }
                for t in self.tracks.values()
            ],
            "away_cue": self._away_cue_since is not None,
            "presence_model": self.cfg.presence_model,
            "intent_model": self.cfg.intent_model,
            "p_group": None if self.p_group is None or self.state == State.ENDED else round(self.p_group, 3),
            "p_tracks": None if self.p_tracks is None or self.state == State.ENDED else round(self.p_tracks, 3),
            "log": self.log[-30:],
        }

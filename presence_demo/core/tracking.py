"""Multi-person tracking (Layer 4) and interaction intent (Layer 3) of the core tracker.

Two association strategies, selected by ``tracker_mode``: ``legacy`` (greedy IoU + face cosine, :meth:`_match`) and
``kalman`` (Kalman prediction + gated cost + Hungarian + coasting, :meth:`_match_kalman`). Both then run the shared
join/drop rules (:meth:`_join_and_drop`). This is a mixin over the tracker's shared state; it is composed into
:class:`presence_demo.core.tracker.PresenceTracker` and has no public API of its own.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .. import intent as im
from ..body import estimate_distance, orientation, shoulder_width
from ..kalman import KalmanBox, appearance_similarity, hungarian
from .geometry import _norm, cosine, iou
from .tracks import Detection, Track


class TrackingMixin:
    """Track association, body/distance update, intent scoring and participant join/drop."""

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _center_in(bbox, r) -> bool:
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        return r[0] <= cx <= r[2] and r[1] <= cy <= r[3]

    def det_in_roi(self, d: Detection) -> bool:
        """Apply ``roi_rule``: face center in face_roi and/or body center in pose_roi."""
        c = self.cfg
        face_ok = d.face_bbox is not None and self._center_in(d.face_bbox, c.face_roi)
        body_ok = d.body is not None and self._center_in(d.body.bbox, c.pose_roi)
        if c.roi_rule == "face" or not c.pose_enabled:
            return face_ok
        if c.roi_rule == "body":
            return body_ok
        return face_ok or body_ok

    def _min_conf(self, d: Detection) -> float:
        """Body-only detections (no face) use ``body_min_conf``; anything with a face uses ``min_conf``."""
        return self.cfg.body_min_conf if d.face_bbox is None and d.body is not None else self.cfg.min_conf

    def _det_far(self, d: Detection) -> bool:
        """Single-frame distance check for a detection (used before it has a track)."""
        c = self.cfg
        if not c.distance_enabled:
            return False
        fw = (d.face_bbox[2] - d.face_bbox[0]) if d.face_bbox is not None else None
        sw = shoulder_width(d.body) if d.body is not None else None
        facing = orientation(d.body)[1] if d.body is not None else ("facing" if fw else None)
        raw = estimate_distance(fw, sw, facing, c.camera_hfov, c.face_width_m, c.shoulder_width_m, 1.0)
        return raw is not None and raw * c.distance_calib > c.leave_distance

    def _speech_join_mode(self) -> str:
        m = self.cfg.speech_join
        if m == "auto":
            if self.speech_source == "clean":
                return "single_candidate"  # clean near speech is trustworthy enough even in vision priority
            return "off" if self.cfg.priority_mode == "vision" else "single_candidate"
        return m

    def _speech_joinable(self, t: Track, cands: list[Track]) -> bool:
        """Whether speech lets track ``t`` join instantly (without waiting ``join_seconds``)."""
        if not self.speaking:
            return False
        mode = self._speech_join_mode()
        if mode == "all":
            return True
        if mode != "single_candidate" or len(cands) != 1:
            return False
        c = self.cfg
        face_ok = t.has_face and self._center_in(t.bbox, c.face_roi)
        near_ok = not c.distance_enabled or t.distance is None or t.distance <= c.zone_max
        return face_ok and t.facing == "facing" and near_ok

    # ------------------------------------------------------------------ tracking
    def _match(self, dets, now):
        c = self.cfg
        for t in self.tracks.values():
            t.visible, t.visible_kind, t.vis_conf = False, "none", 0.0
        pairs = []
        for di, d in enumerate(dets):
            for t in self.tracks.values():
                cs = cosine(d.embedding, t.embedding)
                ov = iou(d.bbox, t.bbox)
                if d.body is not None and t.body is not None and (d.embedding is None or t.embedding is None):
                    ov = max(ov, iou(d.body.bbox, t.body.bbox))  # identity continuity via body when no face
                if (
                    cs >= c.match_cos
                    or (ov >= c.iou_match and cs >= c.cos_weak)
                    or (ov >= c.iou_match and (d.embedding is None or t.embedding is None))
                ):
                    pairs.append((cs + 0.5 * ov, di, t.id, cs))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        for _, di, tid, cs in pairs:  # greedy assignment
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            self._apply_match(self.tracks[tid], dets[di], now, cs)
        self._new_tracks([d for di, d in enumerate(dets) if di not in used_d], now)
        self._join_and_drop(now)

    def _apply_match(self, t: Track, d: Detection, now: float, cs: float) -> None:
        c = self.cfg
        if not t.participant and now - t.last_seen > 2.5:  # gap -> restart join timer
            t.in_roi_since = now
        t.bbox, t.conf, t.last_seen, t.visible, t.similarity = d.bbox, d.conf, now, True, cs
        t.visible_kind, t.vis_conf = "detected", 1.0
        self._update_body(t, d, now)
        if d.embedding is not None:
            t.embedding = (
                d.embedding
                if t.embedding is None
                else _norm((1 - c.embed_ema) * t.embedding + c.embed_ema * d.embedding)
            )
        if d.appearance is not None:
            if t.appearance is None:
                t.appearance = d.appearance
            else:
                a = (1 - c.appearance_ema) * t.appearance + c.appearance_ema * d.appearance
                t.appearance = a / max(float(a.sum()), 1e-9)

    def _new_tracks(self, dets, now) -> None:
        for d in dets:
            t = Track(self._next_id, d.bbox, d.embedding, d.conf, now, now, now, appearance=d.appearance)
            if self.cfg.tracker_mode == "kalman":
                t.kf = KalmanBox(d.bbox, now)
            self._update_body(t, d, now)
            self._next_id += 1
            self.tracks[t.id] = t
            self._event(now, "track_new", "new face in ROI", track=t.id)

    def _match_kalman(self, dets, now):
        """Kalman predict -> gated cost (IoU, motion, face cosine, torso appearance) -> Hungarian -> coast."""
        c = self.cfg
        tracks = list(self.tracks.values())
        for t in tracks:
            if t.kf is None:
                t.kf = KalmanBox(t.bbox, t.last_seen)
            t.kf.predict_to(now)
        pred = [t.kf.bbox() for t in tracks]
        BIG = 1e6
        cost = np.full((len(dets), len(tracks)), BIG)
        cos_of = {}
        for i, d in enumerate(dets):
            for j, t in enumerate(tracks):
                ov = iou(d.bbox, pred[j])
                if d.body is not None and t.body is not None:
                    ov = max(ov, iou(d.body.bbox, t.body.bbox))
                m2 = t.kf.mahalanobis(d.bbox)
                cs = cosine(d.embedding, t.embedding) if d.embedding is not None and t.embedding is not None else None
                ap = appearance_similarity(d.appearance, t.appearance)
                strong_face = cs is not None and cs >= c.match_cos
                if m2 > c.gate_mahalanobis and ov < c.iou_match and not strong_face:
                    continue  # motion gate (a confident face match may re-identify anywhere)
                if cs is not None and cs < c.cos_weak:
                    continue  # faces say: different people
                if cs is None and ap is not None and ap < c.appearance_min_sim:
                    continue  # no face: clothing says different people
                comps = [(1.0, ov), (1.0, float(np.exp(-m2 / c.gate_mahalanobis)))]
                if cs is not None:
                    comps.append((2.0, min(1.0, max(0.0, cs) / max(c.match_cos, 1e-6))))
                if ap is not None:
                    comps.append((1.5, min(1.0, max(0.0, (ap - 0.5) / 0.5))))
                score = sum(w * v for w, v in comps) / sum(w for w, _ in comps)
                if score >= 0.3:
                    cost[i, j] = 1.0 - score
                    cos_of[i, j] = 0.0 if cs is None else cs
        used_d, used_t = set(), set()
        for i, j in hungarian(cost) if len(dets) and len(tracks) else []:
            if cost[i, j] >= BIG:
                continue
            used_d.add(i)
            used_t.add(j)
            t = tracks[j]
            t.kf.update(dets[i].bbox)
            self._apply_match(t, dets[i], now, cos_of[i, j])
        for j, t in enumerate(tracks):
            if j in used_t:
                continue
            missing = now - t.last_seen
            if c.coast_seconds > 0 and missing <= c.coast_seconds:
                t.bbox = pred[j]
                t.visible_kind = "predicted"
                t.vis_conf = max(0.0, 1.0 - missing / c.coast_seconds)
                t.visible = t.vis_conf >= c.coast_present_conf
            else:
                t.visible, t.visible_kind, t.vis_conf = False, "none", 0.0
        self._new_tracks([d for i, d in enumerate(dets) if i not in used_d], now)
        self._join_and_drop(now)

    def _update_intent(self, now: float) -> None:
        """Layer 3: per-track intent score (EMA over real dt) from facing/distance/dwell/motion/center/speech."""
        c = self.cfg
        cands = [t for t in self.tracks.values() if t.visible_kind == "detected" and not t.far]
        facing_cands = [t for t in cands if t.has_face and (t.frontal is None or t.frontal >= 0.5)]
        roi_c = c.face_roi
        for t in self.tracks.values():
            if t.visible_kind != "detected":
                # unseen: no new evidence; let the score sag slowly toward 0 (half the normal speed)
                if t.intent is not None and t.intent_t is not None:
                    t.intent = im.smooth(t.intent, 0.0, now - t.intent_t, c.intent_tau * 4)
                t.intent_t = now
                continue
            cx = (t.bbox[0] + t.bbox[2]) / 2
            size = (t.body.bbox[3] - t.body.bbox[1]) if t.body is not None else (t.bbox[2] - t.bbox[0])
            if t.kf is not None:
                t.vx = t.kf.velocity[0]
            if t._prev is not None and now > t._prev[0]:
                dt = now - t._prev[0]
                a = 1.0 - math.exp(-dt / 1.0)
                if t.kf is None:
                    t.vx += ((cx - t._prev[1]) / dt - t.vx) * a
                rel = (size - t._prev[2]) / max(t._prev[2], 1e-3) / dt
                t.size_rate += (rel - t.size_rate) * a
            t._prev = (now, cx, size)
            pose = t.facing_score if t.body is not None else None
            cues = {
                "facing": im.facing_cue(t.has_face, t.frontal, pose),
                "distance": im.distance_cue(t.distance, c.zone_max, c.leave_distance) if c.distance_enabled else None,
                "dwell": min(1.0, (now - (now if t.in_roi_since is None else t.in_roi_since)) / im.DWELL_FULL),
                "approach": im.approach_cue(t.size_rate),
                "still": im.still_cue(t.vx),
                "center": im.center_cue(cx, (t.bbox[1] + t.bbox[3]) / 2, roi_c),
            }
            weights = {
                "facing": c.w_facing,
                "distance": c.w_distance,
                "dwell": c.w_dwell,
                "approach": c.w_approach,
                "still": c.w_still,
                "center": c.w_center,
            }
            # speech: weak, only when attribution is plausible (single candidate, or the only one facing us)
            if self.speaking and (len(cands) == 1 or (len(facing_cands) == 1 and facing_cands[0] is t)):
                cues["speech"] = 1.0
                weights["speech"] = im.SPEECH_WEIGHT.get(c.priority_mode, 0.05)
            raw = im.gate(im.raw_score(cues, weights), cues["facing"])
            t.intent_cues = {k: (None if v is None else round(v, 2)) for k, v in cues.items()}
            if t.intent is None:
                t.intent = raw
            else:
                t.intent = im.smooth(t.intent, raw, now - (now if t.intent_t is None else t.intent_t), c.intent_tau)
            t.intent_t = now
        for t in self.tracks.values():
            if t.intent is None:
                continue
            if t.intent >= c.join_threshold and t.visible_kind == "detected":
                t.intent_since = t.intent_since if t.intent_since is not None else now
            elif t.intent < c.join_threshold:
                t.intent_since = None
            if t.participant and t.intent < c.drop_threshold:
                t.low_intent_since = t.low_intent_since if t.low_intent_since is not None else now
            else:
                t.low_intent_since = None

    def _intent_ok(self, t: Track, now: float) -> bool:
        """score mode: intent >= join_threshold sustained for join_seconds."""
        return t.intent_since is not None and now - t.intent_since >= self.cfg.join_seconds

    def _join_and_drop(self, now):
        c = self.cfg
        score = c.intent_model == "score"
        if score:
            _ti = time.perf_counter()
            self._update_intent(now)
            self._intent_ms = getattr(self, "_intent_ms", 0.0) + (time.perf_counter() - _ti) * 1000
            for t in self.tracks.values():
                # drop only while we can see the person being disengaged; absence is Layer 2's job
                if (
                    t.participant
                    and t.visible_kind == "detected"
                    and t.low_intent_since is not None
                    and now - t.low_intent_since >= c.drop_seconds
                    and len(self._participants()) > 1  # never drop the last one: absence is Layer 2's job
                ):
                    t.participant = False
                    t.low_intent_since = None
                    t.intent_since = None
                    reason = "intent %.2f < %.2f for %.0fs" % (t.intent, c.drop_threshold, c.drop_seconds)
                    self._event(now, "drop", reason, track=t.id)
        cands = [
            t
            for t in self.tracks.values()
            if t.visible and t.visible_kind == "detected" and not t.participant and not t.far
        ]
        for t in list(self.tracks.values()):
            by_speech = not score and t in cands and self._speech_joinable(t, cands)
            ready = self._intent_ok(t, now) if score else (now - t.in_roi_since >= c.join_seconds or by_speech)
            if (
                t.visible
                and t.visible_kind == "detected"
                and not t.participant
                and not t.far
                and ready
                and self._newgroup_since is None
            ) and (not self._old_group_missing_rules() or not self._participants()):  # join rule unchanged
                t.participant = True
                self._event(
                    now,
                    "join",
                    (
                        "intent %.2f >= %.2f for %.1fs" % (t.intent, c.join_threshold, now - t.intent_since)
                        if score
                        else "in ROI %.1fs" % (now - t.in_roi_since)
                        if not by_speech
                        else "speaking while in ROI"
                    ),
                    track=t.id,
                )
            if now - t.last_seen > c.grace_seconds:
                del self.tracks[t.id]
                self._event(
                    now,
                    "track_drop",
                    f"missing > grace {c.grace_seconds:.0f}s",
                    track=t.id,
                    participant=t.participant,
                )

    def _update_body(self, t: Track, d: Detection, now: float) -> None:
        """Pose, orientation, size trend and distance for a matched/new track."""
        c = self.cfg
        t.has_face = d.face_bbox is not None
        t.frontal = d.frontal if t.has_face else None
        t.body = d.body
        if d.body is not None:
            t.facing_score, t.facing = orientation(d.body)
        else:
            t.facing_score, t.facing = (1.0, "facing") if t.has_face else (0.0, None)
        sw = shoulder_width(d.body) if d.body is not None else None
        fw = (d.face_bbox[2] - d.face_bbox[0]) if d.face_bbox is not None else None
        # size for the trend: body height if available, else face width
        size = (d.body.bbox[3] - d.body.bbox[1]) if d.body is not None else fw
        if size:
            t.size_hist.append((now, size))
            old = [sz for ts, sz in t.size_hist if now - ts >= 1.5]
            if old:
                ratio = size / old[-1]
                t.trend = (
                    "receding"
                    if ratio < 1 - c.recede_ratio
                    else "approaching"
                    if ratio > 1 + c.recede_ratio
                    else "steady"
                )
        raw = estimate_distance(fw, sw, t.facing, c.camera_hfov, c.face_width_m, c.shoulder_width_m, 1.0)
        if raw is not None:
            t.dist_raw = raw
            dist = raw * c.distance_calib
            t.distance = dist if t.distance is None else (1 - c.distance_ema) * t.distance + c.distance_ema * dist
        t.far = bool(c.distance_enabled and t.distance is not None and t.distance > c.leave_distance)

    def zone(self, t: Track) -> str | None:
        """Conversation-zone label for a track (distance mode only)."""
        c = self.cfg
        if not c.distance_enabled or t.distance is None:
            return None
        if t.distance > c.leave_distance:
            return "left"
        if t.distance < c.zone_min:
            return "near"
        return "in" if t.distance <= c.zone_max else "far"

    def _participants(self):
        return [t for t in self.tracks.values() if t.participant]

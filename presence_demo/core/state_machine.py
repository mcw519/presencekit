"""Presence state machine (Layer 2 + session states) of the core tracker.

ACTIVE / IDLE / MAYBE_AWAY / AWAY / ENDED, plus the NEW_GROUP hand-over. Presence evidence comes from either the
``rules`` model (frame counters, :meth:`_update_presence`) or the ``hmm`` model (per-track log-odds,
:meth:`_hmm_update`); :meth:`_evaluate` fuses it with speech/silence according to ``priority_mode``. Every
transition carries a reason string. This is a mixin over the tracker's shared state; it is composed into
:class:`presence_demo.core.tracker.PresenceTracker`.
"""

from __future__ import annotations

from .. import presence_model as pm
from ..config import State
from .tracks import Track, asdict_shallow


class PresenceStateMachineMixin:
    """Session lifecycle, presence evidence and the state evaluation."""

    # ------------------------------------------------------------------ session
    def _reset_session(self, now, initial=False):
        self.tracks: dict[int, Track] = {}  # embeddings live only here (memory)
        self.state = State.ENDED if initial else State.ACTIVE
        self.state_since = now
        self.last_speech = now
        self.speaking = False
        self.present = False
        self.empty = True
        self._empty_streak = 0
        self._present_streak = 0
        self._low_streak = 0
        self._first_empty_at: float | None = None
        self._frame_confident = False
        self.empty_since: float | None = now
        self._newgroup_since: float | None = None
        self._away_cue_since: float | None = None
        self._far_logged = False
        self._hmm_t = now
        self._below_away_since: float | None = None
        self.p_group: float | None = None
        self.p_tracks: float | None = None
        self.timeout = self.cfg.base_timeout
        self.session = getattr(self, "session", 0) + (0 if initial else 1)
        # AWAY bookkeeping (balanced/audio, and vision after present_max_silence): a return needs *new* evidence,
        # not continued presence
        self._away_ids: set[int] = set()
        self._away_absent_seen = False
        self._away_capped = False  # AWAY entered through present_max_silence with people in view
        if initial:
            # after an away-timeout ENDED (balanced/audio), continued presence must not restart a session
            self._restart_needs_fresh = False
            self._restart_absent_seen = False

    def _event(self, now, kind, reason, **extra):
        e = {"t": round(now, 3), "event": kind, "reason": reason, **extra}
        self.event_seq = getattr(self, "event_seq", 0) + 1  # monotonic count (the log itself is capped)
        self.log.append(e)
        self.log = self.log[-200:]

    def _set_state(self, now, new, reason, capped: bool = False):
        if new == State.AWAY and self.state != State.AWAY:
            self._away_ids = {t.id for t in self.tracks.values() if t.visible}
            self._away_absent_seen = not self.present
            self._away_capped = capped
        if new != self.state:
            self._event(now, "transition", reason, frm=self.state.value, to=new.value)
            self.state = new
            self.state_since = now

    def _end(self, now, reason, away_timeout: bool = False):
        # balanced/audio (and vision after present_max_silence): a session that timed out while people may still
        # be in view must not restart from that same continued presence; a restart then needs speech or a vision
        # re-entry (seen absent first)
        self._restart_needs_fresh = away_timeout and (self.cfg.priority_mode != "vision" or self._away_capped)
        self._restart_absent_seen = not self.present
        self._set_state(now, State.ENDED, reason)
        self.tracks.clear()  # drop embeddings
        self.present, self.empty = False, True
        self._newgroup_since = None  # _check_new_group does not run in ENDED: a pending timer would block joins

    def _start(self, now, reason):
        self._restart_needs_fresh = False
        speaking = self.speaking
        self._reset_session(now)
        self.speaking = speaking
        self.state = State.ENDED
        self._set_state(now, State.ACTIVE, reason)

    def _entry_only(self) -> bool:
        """PureSound clean VAD active + onset_entry_only: vision alone (or speech alone) never opens a session;
        only a clean-speech onset together with a visible near candidate does (see set_speech)."""
        return self.speech_source == "clean" and self.cfg.onset_entry_only

    def _onset_ok(self, now) -> bool:
        """Clean-speech gating: with the PureSound clean VAD, speech may start a session / return from AWAY
        only when a confident near person was in the ROI within 1.5 s (TV / far talkers do not trigger), or
        when vision is uncertain (audio-only fallback)."""
        if self.speech_source != "clean" or not self.cfg.onset_needs_vision:
            return True
        return now - self._last_starter_t <= 1.5 or getattr(self, "path", "") == "uncertain" or self._low_streak > 0

    def _update_away_cue(self, now: float) -> None:
        """All visible participants turned away (side/away) and moving away -> start the early cue timer."""
        vis = [t for t in self._participants() if t.visible and not t.far]
        cue = (
            self.cfg.turned_away_early
            and bool(vis)
            and all(t.facing in ("side", "away") and t.trend == "receding" for t in vis)
        )
        if cue and self._away_cue_since is None:
            self._away_cue_since = now
            self._event(now, "turned_away", "participants turned away and moving away")
        elif not cue:
            self._away_cue_since = None

    def _away_newcomer(self) -> bool:
        """A near, detected person in view who was not visible when AWAY began."""
        return any(
            t.visible and t.visible_kind == "detected" and not t.far and t.id not in self._away_ids
            for t in self.tracks.values()
        )

    def _capped_return(self) -> tuple[bool, str]:
        """AWAY through present_max_silence: the same silent person still in view is no return (it would flip
        AWAY <-> ACTIVE on every evaluation and never end). Speech, a re-entry or a new person in view is."""
        speech = self.speaking or self.last_speech > self.state_since
        newcomer = self._away_newcomer()
        if speech:
            return True, "speech with people present"
        return self._away_absent_seen or newcomer, "user returned" if self._away_absent_seen else "new person in view"

    def _old_group_missing(self):
        if self.cfg.presence_model == "hmm":
            p = self._participants()
            return bool(p) and all(t.p < self.cfg.p_away for t in p)
        return self._old_group_missing_rules()

    def _old_group_missing_rules(self):
        """All participants confidently gone: none visible (and near), and this frame was not merely
        low-confidence (a confident detection or a confirmed-empty ROI is required)."""
        p = self._participants()
        evidence = self._frame_confident or (self.empty and self.empty_since is not None)
        return bool(p) and evidence and not any(t.visible and not t.far for t in p)

    def _check_new_group(self, now):
        c = self.cfg
        newcomers = [
            t
            for t in self.tracks.values()
            if not t.participant
            and t.visible
            and t.visible_kind == "detected"
            and not t.far
            and now - t.in_roi_since >= c.join_seconds
            and (c.presence_model != "hmm" or t.p >= c.p_active)
            and (c.intent_model != "score" or self._intent_ok(t, now))
        ]
        if self._old_group_missing() and newcomers:
            if self._newgroup_since is None:
                self._newgroup_since = now
            if now - self._newgroup_since >= c.new_group_seconds:
                ids = [t.id for t in newcomers]
                self._event(now, "NEW_GROUP", f"old participants gone, unmatched faces {ids} took over")
                keep = [Track(**{**asdict_shallow(t), "participant": True}) for t in newcomers]
                self._end(now, "new group replaced old group")
                self._start(now, "new group")
                for t in keep:
                    self.tracks[t.id] = t
                self.present, self.empty = True, False
        else:
            self._newgroup_since = None

    def _update_presence(self, now, confident: bool, low: bool, coasting: bool = False):
        """``coasting``: a predicted (kalman) track still above ``coast_present_conf``. It keeps presence
        and holds off the 'empty' decision; the empty clock starts when coasting ends."""
        c = self.cfg
        if not (low and not confident):
            self._low_streak = 0
        if confident or low:
            self._first_empty_at = None
        if confident:
            self._present_streak += 1
            self._empty_streak = 0
        elif low:
            # low confidence: neither confirms presence nor absence (down-weighted)
            self._present_streak = 0
            self._low_streak += 1
            if self._low_streak >= c.absent_frames and (self.present or self.empty):
                self._event(now, "uncertain", "%d low-confidence frames -> audio-only timeout" % self._low_streak)
                self.present, self.empty, self.empty_since = False, False, None
        else:
            if self._first_empty_at is None and not coasting:
                self._first_empty_at = now
            self._empty_streak += 1
            self._present_streak = 0
        if self._present_streak >= c.present_frames or (confident and self.present):
            if not self.present:
                self._event(now, "presence", "%d confident frames" % self._present_streak)
            self.present, self.empty, self.empty_since = True, False, None
        if self._empty_streak >= c.absent_frames and not coasting:
            if not self.empty:
                self._event(now, "empty", "%d consecutive empty frames" % self._empty_streak)
                self.empty_since = self._first_empty_at if self._first_empty_at is not None else now
            self.present, self.empty = False, True
        if low and not confident and not self.present:
            self.empty = False  # uncertain -> fall back to audio-only base timeout

    # ------------------------------------------------------------------ presence model (hmm)
    def _hmm_update(self, now: float, low_frame: bool) -> None:
        """Integrate this frame's evidence into every track's log-odds; derive present/empty flags."""
        c = self.cfg
        dt = now - self._hmm_t
        self._hmm_t = now
        for t in self.tracks.values():
            turned = c.turned_away_early and t.facing in ("side", "away") and t.trend == "receding"
            rate = pm.track_rate(
                t.visible_kind,
                t.conf,
                t.vis_conf,
                halflife=c.leave_halflife,
                detect_gain=c.detect_gain,
                lowconf_factor=c.lowconf_factor,
                low_frame=low_frame,
                turned_away=turned,
                far=t.far,
            )
            if t.first_seen == now:  # new track: start from the neutral prior + this frame's evidence
                t.logodds = pm.step(0.0, rate, 0.5)
            else:
                t.logodds = pm.step(t.logodds, rate, dt)
        p = self._track_group_p()
        self.p_tracks = p
        was_empty = self.empty
        self.present = p >= c.p_active or (self.present and p >= c.p_maybe)
        self.empty = p < c.p_away
        if self.empty and not was_empty:
            self.empty_since = now
        elif not self.empty:
            self.empty_since = None

    def _track_group_p(self) -> float:
        """Group presence from tracks: max P over participants (else over all tracks). With no track at all
        (e.g. a session started by speech) it is unknown (0.5), except in vision mode (nobody seen)."""
        c = self.cfg
        score = c.intent_model == "score"
        now = self._hmm_t
        ts = self._participants()
        if not ts:  # nobody joined yet: any track (rules) / tracks that show intent (score)
            ts = [t for t in self.tracks.values() if not score or (t.intent or 0) >= c.join_threshold]
        unknown = pm.P_MIN if c.priority_mode == "vision" else 0.5
        if score and c.intent_weighted_presence:
            p = max((t.p * min(1.0, (t.intent or 0) / c.join_threshold) for t in ts), default=unknown)
        else:
            p = max((t.p for t in ts), default=unknown)
        if self._participants() and p < c.p_away:
            # old group gone but a stable new candidate is here: a NEW_GROUP is forming, don't flap to AWAY
            forming = [
                t.p
                for t in self.tracks.values()
                if not t.participant
                and t.visible_kind == "detected"
                and not t.far
                and t.p >= c.p_active
                and now - t.in_roi_since >= c.join_seconds
                and (not score or self._intent_ok(t, now))
            ]
            p = max([p, *forming])
        return p

    def _evaluate_hmm(self, now, silence):
        c = self.cfg
        p_tr = self.p_tracks if self.p_tracks is not None else self._track_group_p()
        a = pm.audio_offset(c.priority_mode, self.speaking, silence, c.base_timeout)
        pg = pm.sigmoid(pm.logit(p_tr) + a)
        self.p_group = pg
        self.path = "present" if self.present else "empty" if self.empty else "uncertain"
        self.timeout, self.elapsed = 0.0, 0.0
        if pg < c.p_away:
            if self._below_away_since is None:
                self._below_away_since = now
        else:
            self._below_away_since = None
        # vision: only the cameras can bring the conversation back (audio never does)
        p_ret = p_tr if c.priority_mode == "vision" else pg
        if self.state == State.AWAY:
            if not self.present:
                self._away_absent_seen = True
            fresh, why = self._capped_return() if self._away_capped else (True, "user returned")
            if p_ret >= c.p_active and fresh:
                if self._away_capped:
                    self.last_speech = now  # fresh engagement: restart the silence clock (it reached the cap)
                self._set_state(now, State.ACTIVE, "%s (P=%.2f)" % (why, p_ret))
            elif now - self.state_since >= c.end_after_away:
                self._end(now, f"away for {c.end_after_away:.0f}s", away_timeout=True)
            return
        capped = c.priority_mode == "vision" and c.present_max_silence > 0 and silence >= c.present_max_silence
        if capped:
            self._set_state(now, State.AWAY, "silence %.0fs reached present_max_silence" % silence, capped=True)
        elif self._below_away_since is not None and now - self._below_away_since >= c.away_hold_seconds:
            self._set_state(now, State.AWAY, "P=%.2f < %.2f for %.1fs" % (pg, c.p_away, now - self._below_away_since))
        elif pg < c.p_maybe or (self.state == State.MAYBE_AWAY and pg < c.p_active):
            self._set_state(now, State.MAYBE_AWAY, "P=%.2f" % pg)
        elif silence < c.idle_after:
            self._set_state(now, State.ACTIVE, "speech")
        else:
            self._set_state(now, State.IDLE, f"silence {silence:.0f}s")

    # ------------------------------------------------------------------ fusion
    def _evaluate(self, now):
        c = self.cfg
        if self.state == State.ENDED:
            self.elapsed, self.path = 0.0, "ended"
            return
        if c.presence_model == "hmm":
            self._evaluate_hmm(now, 0.0 if self.speaking else now - self.last_speech)
            return
        silence = 0.0 if self.speaking else now - self.last_speech
        mode = c.priority_mode
        empty_conf = self.empty and self.empty_since is not None
        self.path = "present" if self.present else "empty" if empty_conf else "uncertain"
        vision_led = mode == "vision" and self.path != "uncertain"
        if mode == "vision" and self.present:
            # confident person in ROI: silence alone never ends the conversation (optional cap)
            self.timeout, elapsed = c.present_max_silence, silence
            why = "silence %.0fs reached present_max_silence with people present"
        elif mode == "vision" and empty_conf:
            # confidently empty ROI: leave after empty_timeout regardless of audio
            self.timeout, elapsed = c.empty_timeout, now - self.empty_since
            why = "ROI empty %.0fs (vision priority, audio ignored)"
        elif mode == "audio":
            k = 1.5 if self.present else 0.5 if empty_conf else 1.0
            self.timeout, elapsed = max(1.0, c.base_timeout * k), silence
            why = "silence %.0fs (audio priority, vision-adjusted)"
        elif self.present:
            self.timeout, elapsed, why = c.present_timeout, silence, "silence %.0fs with people present"
        elif empty_conf:
            self.timeout = c.empty_timeout
            elapsed = min(silence, now - self.empty_since)
            why = "silence + empty ROI %.0fs"
        else:
            self.timeout, elapsed, why = c.base_timeout, silence, "silence %.0fs (vision uncertain)"
        self.elapsed = elapsed
        # vision mode: only a confident person brings the conversation back (speech alone never does)
        if self.state == State.AWAY:
            if not self.present:
                self._away_absent_seen = True
            if mode == "vision" and self._away_capped:
                returned, why_ret = self._capped_return()
                returned = returned and self.present
            elif mode == "vision":
                returned, why_ret = self.present, "user returned"
            else:
                # balanced/audio: only NEW evidence brings the conversation back. Speech now (AWAY is only
                # entered after silence, so this is a speech onset), or a vision re-entry: present again after
                # the ROI was seen without anyone, or a person who was not visible when AWAY began.
                newcomer = self._away_newcomer()
                returned = (self.speaking and self._onset_ok(now)) or (
                    self.present and (self._away_absent_seen or newcomer)
                )
                why_ret = (
                    "speech onset"
                    if self.speaking
                    else "new person in view"
                    if newcomer and not self._away_absent_seen
                    else "user returned"
                )
            if returned:
                if mode != "vision" or self._away_capped:
                    # the return is fresh engagement: restart the silence clock, otherwise the same silence
                    # would send the session straight back to AWAY on the next evaluation
                    self.last_speech = now
                self._set_state(now, State.ACTIVE, why_ret)
            elif now - self.state_since >= c.end_after_away:
                self._end(now, f"away for {c.end_after_away:.0f}s", away_timeout=True)
            return
        early = (
            self._away_cue_since is not None
            and not self.speaking
            and now - self._away_cue_since >= c.turned_away_seconds
        )
        if self.timeout > 0 and elapsed >= self.timeout:
            self._set_state(now, State.AWAY, why % elapsed, capped=mode == "vision" and self.present)
        elif early:
            self._set_state(now, State.MAYBE_AWAY, "turned away + moving away %.1fs" % (now - self._away_cue_since))
        elif (
            self.timeout > 0
            and elapsed >= c.maybe_away_frac * self.timeout
            and not (vision_led and self.path == "present" and silence < c.idle_after)
        ):
            self._set_state(now, State.MAYBE_AWAY, why % elapsed)
        elif silence < c.idle_after:
            self._set_state(now, State.ACTIVE, "speech")
        else:
            self._set_state(now, State.IDLE, f"silence {silence:.0f}s")

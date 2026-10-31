"""Call budget for the remote identity matcher: hourly / total caps, decision cooldown and error breaker.

Pure bookkeeping (no I/O); :class:`~presencekit.identity.driver.ContinuationDriver` owns one per page.
"""

from __future__ import annotations

from collections import deque

BREAKER_ERRORS, BREAKER_PAUSE = 3, 60.0


class CallGuard:
    def __init__(self, breaker_errors: int = BREAKER_ERRORS, breaker_pause: float = BREAKER_PAUSE):
        self.breaker_errors, self.breaker_pause = breaker_errors, breaker_pause
        self.calls = self.tokens = self.errors = 0
        self.hour: deque = deque()
        self.paused_until = 0.0
        self.last_decision_t = -1e9

    def block_reason(self, now: float, cap_per_hour: float, cap_total: float, key_missing: bool) -> str | None:
        """``"breaker"`` / ``"cap"`` / ``"no_key"`` when a call must not be made now, else None."""
        while self.hour and now - self.hour[0] > 3600:
            self.hour.popleft()
        if now < self.paused_until:
            return "breaker"
        if len(self.hour) >= cap_per_hour or self.calls >= cap_total:
            return "cap"
        if key_missing:
            return "no_key"
        return None

    def reserve(self, now: float) -> None:
        """Count a call before awaiting it, so cancellation cannot evade the caps."""
        self.calls += 1
        self.hour.append(now)

    def succeeded(self, tokens: int) -> None:
        self.tokens += int(tokens or 0)
        self.errors = 0

    def failed(self, now: float) -> bool:
        """Record an error; True when the breaker opens (``breaker_pause`` seconds without calls)."""
        self.errors += 1
        if self.errors >= self.breaker_errors:
            self.paused_until = now + self.breaker_pause
            return True
        return False

    def cooling_down(self, now: float, cooldown: float) -> bool:
        return now - self.last_decision_t < cooldown

    def decided(self, now: float) -> None:
        self.last_decision_t = now

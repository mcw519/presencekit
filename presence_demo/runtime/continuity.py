"""Ephemeral page continuity and observation time; no persistent identity storage.

A cryptographically random page capability lives only in the browser's JS memory.
The normal WebSocket authentication still applies. Replacing a transport never
counts as evidence that a person left, nor refreshes historical conversation TTLs.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..identity.store import ReferenceStore

PAGE_TOKEN = re.compile(r"^[0-9a-f]{32}$")
PAGE_TTL = 600.0


class ObservationClock:
    """Freeze presence time during missing sensor input, preserving ordinary time otherwise."""

    def __init__(self, clock=time.monotonic):
        self.wall = clock
        self.offset = 0.0
        self.paused_at = None
        self.last = float("-inf")
        self.valid_until = None

    def __call__(self):
        if self.valid_until is not None and self.wall() > self.valid_until:
            self.pause(at=self.valid_until)
        self.last = max(self.last, (self.wall() if self.paused_at is None else self.paused_at) - self.offset)
        return self.last

    def pause(self, at=None):
        if self.paused_at is None:
            self.paused_at = max(self.last + self.offset, self.wall() if at is None else at)

    def resume(self):
        if self.paused_at is not None:
            self.offset += self.wall() - self.paused_at
            self.paused_at = None
        self.valid_until = None

    def observed(self, grace=3.0):
        """Renew sensor freshness; all readers freeze automatically after this lease."""
        self()  # Honor an expired old lease even if this frame wakes before every other reader.
        self.resume()
        self.valid_until = self.wall() + grace


@dataclass
class PageState:
    store: ReferenceStore = field(default_factory=ReferenceStore, repr=False)
    session: int = 0
    generation: int = 0
    detached_at: float | None = None
    interrupt: object = field(default=None, repr=False)
    disconnect: object = field(default=None, repr=False)


class PageRegistry:
    """Single event-loop registry, scoped by an unguessable per-page capability.

    Live transports and detached leases are bounded; expired leases are wiped.
    Browser refresh produces a new capability and cannot access an old page's data.
    """

    def __init__(self, clock=time.monotonic, limit=64):
        self.clock, self.limit = clock, limit
        self.pages: dict[str, PageState] = {}

    def prune(self):
        now = self.clock()
        for token, page in list(self.pages.items()):
            page.store.expire(now, PAGE_TTL)
            if page.detached_at is not None and now - page.detached_at >= PAGE_TTL:
                page.store.wipe()
                del self.pages[token]

    def acquire(self, token):
        if not PAGE_TOKEN.fullmatch(token):
            raise ValueError("invalid page capability")
        self.prune()
        if token not in self.pages:
            if len(self.pages) >= self.limit:
                detached = [(p.detached_at, key) for key, p in self.pages.items() if p.detached_at is not None]
                if not detached:
                    raise ValueError("page capacity reached")
                _, oldest = min(detached)
                self.pages.pop(oldest).store.wipe()
            self.pages[token] = PageState()
        page = self.pages[token]
        # Invalidate old work before handing its store to the replacement transport.
        if page.interrupt:
            page.interrupt()
        if page.disconnect:
            page.disconnect()
        page.interrupt = page.disconnect = None
        page.generation += 1
        page.detached_at = None
        return page, page.generation

    def release(self, page, generation):
        if page.generation != generation:
            return False
        if page.interrupt:
            page.interrupt()
        page.interrupt = page.disconnect = None
        page.detached_at = self.clock()
        return True

    def forget(self, token):
        page = self.pages.pop(token, None)
        if page is None:
            return False
        page.generation += 1
        page.store.wipe()
        page.interrupt = page.disconnect = None
        return True


PAGES = PageRegistry()

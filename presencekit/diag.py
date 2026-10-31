"""Privacy-safe structured logging, an in-memory event ring buffer, stack dumps and task supervision.

Never log image/audio data, tokens, API keys or transcript text: only counters, timings, states and config values
(with secret-looking keys masked).
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import logging.handlers
import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Awaitable, Callable

log = logging.getLogger("presence")
EVENTS: collections.deque = collections.deque(maxlen=int(os.environ.get("PRESENCE_EVENT_BUFFER", "500")))
_ev_lock = threading.Lock()
_listeners: list[Callable[[dict], None]] = []
_SECRET = re.compile(r"(key|token|secret|password|authorization)", re.I)
_TOKEN_RE = re.compile(r"((?:token|key)=)[^&\s\"']+", re.I)


class Redact(logging.Filter):
    """Mask ``token=…`` / ``key=…`` in any log line (e.g. uvicorn's WebSocket handshake line)."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        masked = _TOKEN_RE.sub(r"\1***", msg)
        masked = re.sub(r"page-[0-9a-f]{32}\b", "page-***", masked)
        if masked != msg:
            record.msg, record.args = masked, ()
        return True


def setup_logging() -> None:
    """Root logging: level ``PRESENCE_LOG_LEVEL`` (INFO); file ``PRESENCE_LOG_FILE`` with size rotation
    (``PRESENCE_LOG_MAX_BYTES`` 5 MB × ``PRESENCE_LOG_BACKUPS`` 3) or stderr when unset."""
    level = getattr(logging, os.environ.get("PRESENCE_LOG_LEVEL", "INFO").upper(), logging.INFO)
    path = os.environ.get("PRESENCE_LOG_FILE", "")
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        h: logging.Handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=int(os.environ.get("PRESENCE_LOG_MAX_BYTES", str(5 << 20))),
            backupCount=int(os.environ.get("PRESENCE_LOG_BACKUPS", "3")),
            encoding="utf-8",
        )
    else:
        h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    h.addFilter(Redact())
    root = logging.getLogger()
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(h)
    root.setLevel(level)
    logging.getLogger("websockets").setLevel(logging.WARNING)  # its DEBUG lines can contain headers


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    s = str(v)
    return f'"{s}"' if (" " in s or not s) else s


def safe(d: dict) -> dict:
    """Copy of ``d`` with secret-looking keys masked and long strings cut."""
    out = {}
    for k, v in d.items():
        if _SECRET.search(str(k)):
            out[k] = "***" if v else ""
        elif isinstance(v, str) and len(v) > 200:
            out[k] = v[:200] + "…"
        else:
            out[k] = v
    return out


def event(_event: str, level: str = "info", conn: str | None = None, **kv) -> dict:
    """Log one structured event (``event=name conn=… k=v``) and keep it in the ring buffer."""
    kv = safe(kv)
    name = _event
    rec = {"t": round(time.time(), 3), "level": level, "event": name, **({"conn": conn} if conn else {}), **kv}
    with _ev_lock:
        EVENTS.append(rec)
        ls = list(_listeners)
    msg = " ".join([f"event={name}"] + ([f"conn={conn}"] if conn else []) + [f"{k}={_fmt(v)}" for k, v in kv.items()])
    log.log(getattr(logging, level.upper(), logging.INFO), msg)
    for f in ls:
        with contextlib.suppress(Exception):
            f(rec)
    return rec


def subscribe(f: Callable[[dict], None]) -> Callable[[], None]:
    with _ev_lock:
        _listeners.append(f)

    def unsub():
        with _ev_lock:
            if f in _listeners:
                _listeners.remove(f)

    return unsub


def recent(n: int = 200, conn: str | None = None) -> list[dict]:
    with _ev_lock:
        ev = list(EVENTS)
    if conn:
        ev = [e for e in ev if e.get("conn") in (conn, None)]
    return ev[-n:]


def dump_stacks(limit: int = 25) -> str:
    """All Python thread stacks (like faulthandler, but returned as text for the log)."""
    names = {t.ident: t.name for t in threading.enumerate()}
    parts = []
    for tid, frame in sys._current_frames().items():
        parts.append(
            f"--- thread {names.get(tid, '?')} ({tid})\n" + "".join(traceback.format_stack(frame, limit=limit))
        )
    return "\n".join(parts)


async def supervise(
    name: str,
    factory: Callable[[], Awaitable[None]],
    conn: str,
    on_crash: Callable[[str, str], None] | None = None,
    max_restarts: int = 20,
) -> None:
    """Run ``factory()`` forever; log any exception with its full traceback and restart (no silent failures)."""
    restarts = 0
    while True:
        try:
            await factory()
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            restarts += 1
            tb = traceback.format_exc()
            event("task_crash", "error", conn, task=name, error=f"{type(e).__name__}: {e}", restarts=restarts)
            log.error("traceback task=%s conn=%s\n%s", name, conn, tb)
            if on_crash:
                on_crash(name, f"{type(e).__name__}: {e}")
            if restarts > max_restarts:
                event("task_gave_up", "error", conn, task=name)
                return
            await asyncio.sleep(min(5.0, 0.2 * restarts))

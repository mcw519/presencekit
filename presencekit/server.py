"""FastAPI server: serves the demo page and runs one PresenceTracker per WebSocket."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import faulthandler
import logging
import secrets
from dataclasses import asdict
from pathlib import Path

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as e:  # the SDK installs without the web stack
    raise SystemExit("the web demo needs the web extra: pip install 'presencekit[web]' (or uv sync)") from e

from . import asr as asrmod
from . import diag
from . import reid as reidmod
from .audio import puresound_available
from .tracker import RANGES, Config
from .web import session
from .web.outbox import SEND_AUDIO_MAX, SEND_AUDIO_STALE, SEND_JSON_MAX, Outbox  # noqa: F401  (re-exported)
from .web.runtime import (  # noqa: F401  (re-exported for callers/tests)
    AUDIO_POOL,
    TOKEN,
    VISION_LOCK,
    VISION_LOCK_TIMEOUT,
    VisionBusy,
    _vision_job,
    calibrate,
    pipeline,
    pose_detector,
)

STATIC = Path(__file__).parent / "static"


async def _expire_pages():
    # App-owned: still runs after the last WebSocket and its ticker have closed.
    from .web.continuity import PAGES

    while True:
        PAGES.prune()
        await asyncio.sleep(1)


@contextlib.asynccontextmanager
async def lifespan(app):
    cleaner = asyncio.create_task(_expire_pages())
    try:
        yield
    finally:
        cleaner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleaner
        from .web.continuity import PAGES

        for token in list(PAGES.pages):
            PAGES.forget(token)


app = FastAPI(title="PresenceKit", lifespan=lifespan)
PUBLIC_PREFIXES = ("/static/vendor/", "/static/js/", "/static/i18n/")


@app.middleware("http")
async def require_token(request: Request, call_next):
    # vendored design-system assets (public CSS + fonts, no data) load without the token
    # and the front-end code (static JS and string tables, no data) load without the token
    public = request.url.path.startswith(PUBLIC_PREFIXES) or request.url.path == "/static/app.css"
    given = request.query_params.get("token", "").encode()  # bytes: non-ASCII str would raise TypeError
    if TOKEN and not public and not secrets.compare_digest(given, TOKEN.encode()):
        return PlainTextResponse("forbidden", status_code=403)
    resp = await call_next(request)
    if request.url.path.startswith(("/static/js/", "/static/i18n/")):
        resp.headers["Cache-Control"] = "no-cache"  # always revalidate the app code (ETag) after an upgrade
    return resp


app.mount("/static/vendor", StaticFiles(directory=STATIC / "vendor"), name="vendor")
app.mount("/static/js", StaticFiles(directory=STATIC / "js"), name="js")
app.mount("/static/i18n", StaticFiles(directory=STATIC / "i18n"), name="i18n")


@app.get("/static/app.css")
def app_css():
    return FileResponse(STATIC / "app.css", media_type="text/css", headers={"Cache-Control": "no-cache"})


@app.get("/")
def index():
    html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__VERSION__", _version())
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("presencekit")
    except Exception:  # noqa: BLE001
        return "dev"


@app.get("/api/config")
def default_config():
    return {
        "defaults": asdict(Config()),
        "ranges": {
            k: [lo, hi, t if isinstance(t, str) else (t.__name__ if t else "choice")]
            for k, (lo, hi, t) in RANGES.items()
        },
        "puresound": dict(zip(("available", "reason"), puresound_available(), strict=True)),
    }


@app.get("/api/asr_status")
def asr_status():
    return asrmod.key_status()  # never the key itself


@app.post("/api/asr_key")
async def asr_key(request: Request):
    """Set/clear the AssemblyAI key in server memory (token-protected by the middleware). Not logged/stored."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    return asrmod.set_key(str(body.get("key") or "")[:256])


@app.get("/api/reid_status")
def reid_status():
    return reidmod.key_status()  # never the key itself


@app.post("/api/reid_key")
async def reid_key(request: Request):
    """Set/clear the re-ID backend key in server memory (token-protected). Not logged/stored."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    return reidmod.set_key(str(body.get("key") or "")[:512])


@app.post("/api/reid_reset")
def reid_reset():
    """Wipe every connection's in-memory reference faces."""
    n = reidmod.wipe_all()
    diag.event("reid_store", action="wipe", reason="api", count=n)
    return {"wiped": n}


@app.post("/api/reid_page_close")
async def reid_page_close(request: Request):
    """Best-effort page unload wipe; requires the in-memory page capability, never logged."""
    from .web.continuity import PAGE_TOKEN, PAGES

    try:
        body = await request.json()
        capability = body.get("page_token") if isinstance(body, dict) else None
    except ValueError:
        capability = None
    if isinstance(capability, str) and PAGE_TOKEN.fullmatch(capability):
        PAGES.forget(capability)
    return {"ok": True}  # do not reveal whether a page capability exists


@app.get("/debug/events")
def debug_events(n: int = 200, conn: str | None = None):
    """Recent structured events (token-protected by the middleware; no media, keys or transcript text)."""
    return {"events": diag.recent(max(1, min(n, 2000)), conn)}


app.websocket("/ws")(session.ws)
atexit.register(reidmod.wipe_all)  # shutdown: wipe in-memory reference faces


def main():
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    diag.setup_logging()
    faulthandler.enable()  # hard crashes (segfault) still print all thread stacks to stderr
    # access_log=False: request lines would contain the ?token= query string. Frames travel over the
    # WebSocket and are never logged. log_config=None: uvicorn logs go through our handler (rotation, redaction).
    diag.event("server_start", host=a.host, port=a.port, log_level=logging.getLevelName(logging.getLogger().level))
    uvicorn.run(
        app,
        host=a.host,
        port=a.port,
        access_log=False,
        log_config=None,
        ws_max_size=8 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()

_RedactToken = diag.Redact

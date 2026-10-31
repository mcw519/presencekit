"""Helpers for source-level tests: the web UI (markup + scripts + string tables) and the server code."""

from pathlib import Path

PKG = Path(__file__).parents[1] / "presencekit"
STATIC = PKG / "static"


def ui_files() -> list[Path]:
    return [STATIC / "index.html", *sorted((STATIC / "js").glob("*.js")), *sorted((STATIC / "i18n").glob("*.js"))]


def ui_source() -> str:
    """index.html followed by every front-end script and string table (one string, for grep-style checks)."""
    return "\n".join(p.read_text(encoding="utf-8") for p in ui_files())


def server_source() -> str:
    """server.py plus the web transport package and the runtime package it drives (v0.13 layering)."""
    files = [PKG / "server.py", *sorted((PKG / "web").glob("*.py")), *sorted((PKG / "runtime").glob("*.py"))]
    return "\n".join(p.read_text(encoding="utf-8") for p in files)

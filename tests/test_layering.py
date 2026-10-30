"""Dependency direction of the v0.13 layers (static import check, no imports executed).

core      -> numpy + pure-python helpers only (no OpenCV, no web framework, no runtime/web modules)
detectors -> may use OpenCV, never the web framework
identity  -> never the web framework, never the web/runtime layers
runtime   -> never the web framework (FastAPI/Starlette/uvicorn)
"""

from __future__ import annotations

import ast
from pathlib import Path

from presence_demo.core import Observation, PresenceEvent, Step, VideoFrame

PKG = Path(__file__).resolve().parent.parent / "presence_demo"
WEB_FRAMEWORK = {"fastapi", "starlette", "uvicorn"}


def _imports(layer: str) -> dict[str, set[str]]:
    """Absolute module names imported by every file of ``presence_demo/<layer>`` (relative imports resolved)."""
    out = {}
    for path in sorted((PKG / layer).rglob("*.py")):
        rel = path.relative_to(PKG.parent).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        pkg = parts if path.name == "__init__.py" else parts[:-1]
        names = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg[: len(pkg) - node.level + 1]
                    mod = ".".join(base + ([node.module] if node.module else []))
                    names.add(mod)
                    if not node.module:
                        names.update(f"{mod}.{a.name}" for a in node.names)
                else:
                    names.add(node.module)
        out[str(path.relative_to(PKG))] = names
    assert out, layer
    return out


def _violations(layer: str, banned_roots: set[str], banned_prefixes: tuple[str, ...] = ()) -> list[str]:
    bad = []
    for f, names in _imports(layer).items():
        for n in names:
            if n.split(".")[0] in banned_roots or n.startswith(banned_prefixes):
                bad.append(f"{f}: {n}")
    return bad


def test_core_imports_only_numpy_and_pure_helpers():
    banned = {"cv2", "ai_edge_litert", "psutil", *WEB_FRAMEWORK}
    prefixes = (
        "presence_demo.web",
        "presence_demo.runtime",
        "presence_demo.detectors",
        "presence_demo.identity",
        "presence_demo.vision",
        "presence_demo.optimizer",
        "presence_demo.reid",
        "presence_demo.audio",
        "presence_demo.asr",
        "presence_demo.server",
        "presence_demo.appearance",
        "presence_demo.embed_backends",
    )
    assert not _violations("core", banned, prefixes)


def test_core_types_round_trip_the_legacy_dicts():
    e = {"t": 1.5, "event": "transition", "reason": "speech", "frm": "IDLE", "to": "ACTIVE"}
    assert PresenceEvent.from_dict(e).to_dict() == e
    assert list(PresenceEvent.from_dict(e).to_dict()) == list(e)
    f = {"data": b"\xff\xd8..\xff\xd9", "t": 12.0, "speech": True}
    assert VideoFrame.from_dict(f).to_dict() == f
    assert Observation().kind == "tick" and Observation(speaking=True).kind == "speech"
    assert Observation(detections=[]).kind == "frame"
    assert Step(events=(), snapshot={}).events == ()

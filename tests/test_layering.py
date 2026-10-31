"""Dependency direction of the layers (static import check, no imports executed).

core      -> numpy + pure-python helpers only (no OpenCV, no web framework, no runtime/web modules)
inference -> numpy + onnxruntime only (no OpenCV, no other presence_demo layer)
detectors -> may use OpenCV and inference, never the web framework
identity  -> never the web framework, never the web/runtime layers
runtime   -> never the web framework (FastAPI/Starlette/uvicorn), never presence_demo.web / server
"""

from __future__ import annotations

import ast
import sys
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
    banned = {"cv2", "ai_edge_litert", "onnxruntime", "psutil", *WEB_FRAMEWORK}
    prefixes = (
        "presence_demo.inference",
        "presence_demo.sdk",
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


def test_inference_depends_on_numpy_and_onnxruntime_only():
    for f, names in _imports("inference").items():
        for n in names:
            root = n.split(".")[0]
            assert (
                root in {"numpy", "onnxruntime", "__future__"}
                or root in sys.stdlib_module_names
                or (n.startswith("presence_demo.inference"))
            ), f"{f}: {n}"


def test_sdk_never_imports_the_web_framework_or_the_web_runtime_layers():
    path = PKG / "sdk.py"
    names = {
        (node.module if isinstance(node, ast.ImportFrom) else a.name)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for a in (node.names if isinstance(node, ast.Import) else [None])
    }
    assert not {n for n in names if n and n.split(".")[0] in WEB_FRAMEWORK | {"runtime", "web", "server", "stats"}}


def test_detectors_never_import_the_web_framework_or_upper_layers():
    prefixes = ("presence_demo.web", "presence_demo.runtime", "presence_demo.server", "presence_demo.identity")
    assert not _violations("detectors", WEB_FRAMEWORK, prefixes)


def test_identity_never_imports_the_web_framework_or_the_web_runtime_layers():
    prefixes = ("presence_demo.web", "presence_demo.runtime", "presence_demo.server")
    assert not _violations("identity", WEB_FRAMEWORK, prefixes)


def test_runtime_never_imports_the_web_framework_or_the_web_layer():
    prefixes = ("presence_demo.web", "presence_demo.server")
    assert not _violations("runtime", WEB_FRAMEWORK, prefixes)


def test_lower_layers_load_without_the_web_framework():
    import subprocess
    import sys

    code = (
        "import sys; import presence_demo.core, presence_demo.inference, presence_demo.detectors, "
        "presence_demo.identity, presence_demo.sdk, presence_demo.runtime; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
        "('fastapi', 'starlette', 'uvicorn') or m.startswith(('presence_demo.web', 'presence_demo.server'))); "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == ""

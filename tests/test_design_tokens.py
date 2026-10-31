"""The UI uses the PureSound web design system (vendored, pinned) instead of its own token copy."""

import hashlib
import re
from pathlib import Path

from fastapi.testclient import TestClient
from sources import ui_source

from presencekit import server

VENDOR = Path(server.STATIC) / "vendor" / "puresound"
# unmodified upstream files at puresound 121bf623b491cb7f18805bec65099543ecc5323f (see vendor/puresound/README.md)
PINNED = {
    "tokens.css": "d788571678a05894d084e8bda54f4445b8d1c3072ff6e1ecf3718c8934c3822b",
    "components.css": "cbb5a2580dd69b3451e7e77f2b2794bf86a4559835aa2addd1cadd87b0a590e3",
    "shell.css": "de94af088d54e9fca0beda35be7d8338cd40fca6d68aa48605d647ec78919399",
}


def test_vendored_tokens_match_pinned_upstream():
    for name, digest in PINNED.items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name
    for f in ("LICENSE", "NOTICE", "README.md", "fonts/LICENSE.txt"):
        assert (VENDOR / f).is_file(), f


def test_vendor_and_sdk_revisions_are_pinned_independently():
    rev = "121bf623b491cb7f18805bec65099543ecc5323f"
    assert rev in (VENDOR / "README.md").read_text()
    assert "independently" in (VENDOR / "README.md").read_text()
    root = Path(server.__file__).parents[1]
    source = re.search(r'puresound-streaming = \{[^\n]+rev = "([0-9a-f]{40})"', (root / "pyproject.toml").read_text())
    assert source, "SDK must remain pinned to an immutable commit"
    sdk_rev = source.group(1)
    assert f"rev={sdk_rev}#{sdk_rev}" in (root / "uv.lock").read_text()


def test_index_uses_shared_tokens_and_component_classes():
    html = ui_source()
    for href in ("tokens.css", "base.css", "components.css", "shell.css", "screens-shared.css"):
        assert f"/static/vendor/puresound/{href}" in html
    assert 'class="button button-primary"' in html
    assert 'class="panel panel-pad"' in html
    assert 'class="inspector"' in html and "inspector-section" in html
    assert 'class="segmented"' in html and 'class="toast"' in html
    assert 'class="toggle-row"' in html or "toggle-row" in html  # built in rowHTML
    css = (Path(server.STATIC) / "app.css").read_text()
    for role in ("--page:", "--canvas:", "--ink:", "--accent:", "--good:", "--bad:", "--font-sans:"):
        assert role not in css, role
    # presence must use PureSound button/panel classes, not reinvent them
    assert ".button {" not in css and ".panel {" not in css


def test_vendor_assets_public_but_app_still_protected(monkeypatch):
    monkeypatch.setattr(server, "TOKEN", "tok")
    c = TestClient(server.app)
    r = c.get("/static/vendor/puresound/tokens.css")
    assert r.status_code == 200 and "--accent" in r.text
    assert c.get("/static/vendor/puresound/fonts/IBMPlexSans-400.woff2").status_code == 200
    assert c.get("/static/app.css").status_code == 200
    assert c.get("/").status_code == 403
    r = c.get("/?token=tok")
    assert r.status_code == 200 and "__VERSION__" not in r.text
    assert c.get("/static/vendor/../index.html").status_code in (403, 404)

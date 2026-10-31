"""Settings UI: every config key belongs to a module; server config keys and client defaults agree."""

import re
from dataclasses import fields

from sources import STATIC, ui_source

from presencekit.tracker import Config

HTML = ui_source()


def _def_keys():
    body = re.search(r"const DEF=\{(.*?)\};", HTML, re.S).group(1)
    return set(re.findall(r"(?:^|[,{\s])([a-z_0-9]+):", body))


def test_every_setting_is_in_a_module():
    mods = HTML[HTML.index("const MODULES=[") : HTML.index("const MOD_PRESETS")]
    placed = set(re.findall(r"'([a-z_0-9]+)'", mods))
    missing = {k for k in _def_keys() if k != "monitor" and k not in placed}
    assert not missing, missing


def test_client_defaults_cover_server_config():
    client_only = {"monitor", "mirror_view", "fps", "vad_k", "vad_min", "vad_hang", "show_face_roi", "show_pose_roi"}
    server = {f.name for f in fields(Config)}
    ui = _def_keys() - client_only
    assert ui <= server, ui - server


def test_settings_key_rows_styled():
    # API-key rows share one builder: PureSound buttons, .control input, padded module body, status badge
    assert "function keyRow(" in HTML and "keyRow('asr'" in HTML and "keyRow('reid'" in HTML
    assert "det.querySelector('.mbody')" in HTML and 'class="kbadge"' in HTML
    assert (
        re.search(
            r"<button(?![^>]*class=)[^>]*>",
            HTML[HTML.index("function addCalib") : HTML.index("function addCalib") + 900]
            + HTML[HTML.index("function keyRow") : HTML.index("$('reidViewClr')")],
        )
        is None
    )
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert ".mod > .row{padding:10px 12px}" in css and ".kbadge" in css and ".rv-row" in css
    assert 'id="reidCard"' in HTML and "RV_MAX=4" in HTML and "reid_show_crops:true" in HTML

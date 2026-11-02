"""UI: audio panel under the camera and the live speech-recognition caption overlay."""

import re
from pathlib import Path

from sources import ui_source

ROOT = Path(__file__).parents[1] / "presencekit" / "static"
HTML = ui_source()
CSS = (ROOT / "app.css").read_text(encoding="utf-8")


def test_audio_panel_is_in_left_column_below_camera():
    main = HTML.index('class="workbench-main presence-stack"')
    view, audio, right = HTML.index('id="view"'), HTML.index('id="aCard"'), HTML.index('<div class="presence-stack">')
    assert main < view < audio < right


def test_caption_overlay_outside_mirrored_layers():
    view = re.search(r'<div id="view">(.*?)</div>\s*<div class="sig-stack"', HTML, re.S).group(1)
    assert 'id="cap"' in view and "hidden" in view
    assert "#view video.mirror" in CSS and ".caption" in CSS and "scaleX" not in CSS[CSS.index("v0.10.5") :]


def test_caption_logic():
    assert "caption(m);" in HTML  # every partial/final updates in place
    assert "m.kind==='final'?CAP_HOLD_MS" in HTML  # fade out after final
    assert "if(cfg.asr_provider==='none')captionOff();" in HTML  # hidden when ASR off
    assert "textContent=m.text" in HTML  # no HTML injection from transcripts


def test_monitor_stage_holds_only_video_and_tracks_0106():
    a = HTML.index('id="stage"')
    b = HTML.index("</section>", a)
    stage = HTML[a:b]
    for x in ('id="view"', 'id="cap"', 'id="grpRaw"', 'id="grpClean"'):
        assert x in stage
    for x in ('id="roiPick"', 'id="tl"', 'id="monBtns"', 'id="recBtn"'):
        assert x not in stage
    assert stage.index('id="view"') < stage.index('class="sig-stack"')
    assert "100vh" in CSS[CSS.index("v0.10.6") :]


def test_ids_referenced_by_script_exist():
    """Every static $('id') lookup has a matching element (catches broken layout moves)."""
    dynamic = ("st_", "se_", "fs_", "mod_", "row_", "reidKey", "asrKey")  # key rows built by keyRow()
    missing = [
        i
        for i in set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", HTML))
        if not i.startswith(dynamic) and f'id="{i}"' not in HTML
    ]
    assert not missing, missing


def test_tracks_stacked_with_view_switch_0107():
    stage = HTML[HTML.index('id="stage"') : HTML.index("</section>", HTML.index('id="stage"'))]
    assert stage.index('id="grpRaw"') < stage.index('id="grpClean"')  # noisy on top, output below
    assert "sig-pair" not in HTML and "grid-template-columns:1fr 1fr" not in CSS[CSS.index("v0.10.6") :]
    pick = re.search(r'id="trkView".*?</div>', HTML, re.S).group(0)
    assert all(f'data-v="{v}"' in pick for v in ("wave", "spec", "both"))
    assert "localStorage.setItem(TRK_KEY" in HTML
    assert "if(!running||!showWave())return;wave(" in HTML and "if(showSpec()){sg(" in HTML  # hidden canvases not drawn
    for v in ("wave", "spec"):
        assert f".monitor-stage[data-view={v}]" in CSS  # per-mode track heights keep the frame in the viewport


def test_view_switch_outside_frame_and_lane_gutters_0108():
    a = HTML.index('id="stage"')
    stage = HTML[a : HTML.index("</section>", a)]
    assert 'id="trkView"' not in stage and HTML.index('id="trkView"') < a  # switch sits in the header above the frame
    for lane, db in (("grpRaw", "dbRaw"), ("grpClean", "dbClean")):
        block = stage[stage.index(f'id="{lane}"') :]
        assert block.index('class="lane-label"') < block.index('class="lane-plots"') and f'id="{db}"' in block
    css = CSS[CSS.index("v0.10.6") :]
    assert "grid-template-columns:var(--lane-gutter) minmax(0,1fr)" in css  # same gutter => aligned time axes
    assert ".monitor-stage{display:grid;gap:10px;padding:12px" in css  # inner gutter, not full-bleed

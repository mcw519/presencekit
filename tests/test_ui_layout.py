"""UI layout: state machine axis, audio signal bundles, draggable face/body ROI with mirror-safe coordinates."""

import json
import re
import shutil
import subprocess

import pytest
from sources import ui_source

HTML = ui_source()


def test_state_machine_is_an_axis_with_history():
    assert 'class="fsm-axis" id="sm"' in HTML and 'id="smHist"' in HTML
    assert "fsm-edge" in HTML and "is-current" in HTML
    assert "segmented sm" not in HTML


def _group(cls):
    a = HTML.index('<div class="sig-lane %s"' % cls)
    b = HTML.index('<div class="sig-lane', a + 10) if cls == "raw" else HTML.index("</section>", a)
    return HTML[a:b]


def test_audio_bundles_wave_and_spectrogram_per_signal():
    raw, clean = _group("raw"), _group("clean")
    assert 'id="wRaw"' in raw and 'id="sgRaw"' in raw and "Clean" not in raw
    assert 'id="wClean"' in clean and 'id="sgClean"' in clean and "Raw" not in clean


def test_roi_picker_covers_both_rois():
    pick = re.search(r'id="roiPick".*?</div>', HTML, re.S).group(0)
    assert 'data-roi="face_roi"' in pick and 'data-roi="pose_roi"' in pick


def test_distance_calibration_card_is_on_main_workbench():
    assert HTML.index('id="roiCard"') < HTML.index('id="calibCard"') < HTML.index('id="aCard"')
    presets = re.search(r'id="calPresets".*?</div>', HTML, re.S).group(0)
    assert all(f'data-m="{m}"' in presets for m in ("0.5", "1.0", "1.5"))
    for eid in ('id="calM"', 'id="calB"', 'id="calDelayB"', 'id="calReset"', 'id="calEnable"', 'id="calBadge"'):
        assert eid in HTML


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_roi_math_mirror_roundtrip():
    body = re.search(r"const roiMath=(.*?)\};\n", HTML, re.S).group(1) + "}"
    js = f"""const roiMath={body};
const out={{}};
out.toRaw=[roiMath.toRaw(0.1,true),roiMath.toRaw(0.1,false)];
out.view=roiMath.boxToView([0.1,0.2,0.4,0.9],true);
out.back=roiMath.boxToView(out.view,true);
// drag on mirrored view from view-x 0.9 -> 0.6 (raw 0.1 -> 0.4)
out.newbox=roiMath.norm([roiMath.toRaw(0.9,true),0.2],[roiMath.toRaw(0.6,true),0.8]);
out.tiny=roiMath.norm([0.99,0.5],[0.99,0.5]);
console.log(JSON.stringify(out));"""
    r = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert r["toRaw"][0] == pytest.approx(0.9) and r["toRaw"][1] == 0.1
    assert r["view"] == pytest.approx([0.6, 0.2, 0.9, 0.9])
    assert r["back"] == pytest.approx([0.1, 0.2, 0.4, 0.9])
    assert r["newbox"] == pytest.approx([0.1, 0.2, 0.4, 0.8])
    x0, y0, x1, y1 = r["tiny"]
    assert 0 <= x0 < x1 <= 1 and x1 - x0 >= 0.049 and y1 - y0 >= 0.049

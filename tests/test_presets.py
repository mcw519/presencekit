"""Tuned defaults and the three UI modes."""

import json
import re
import shutil
import subprocess
from dataclasses import asdict

import pytest
from sources import ui_source

from presencekit.tracker import Config

HTML = ui_source()
CLIENT_ONLY = {"fps", "vad_k", "vad_min", "vad_hang"}  # browser-side send rate / VAD


def test_tuned_defaults():
    c = Config()
    assert (c.min_conf, c.leave_halflife, c.force_detect_seconds) == (0.4, 1.5, 2.0)
    assert (c.reid_decision_timeout, c.reid_face_wait) == (10.0, 3.0)
    for k, v in (("min_conf", 0.4), ("leave_halflife", 1.5), ("force_detect_seconds", 2), ("reid_face_wait", 3)):
        assert re.search(rf"[,{{\s]{k}:{v}[,}}]", HTML), k  # client defaults agree


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_three_modes_are_valid_and_near_uses_audio_recovery():
    start = HTML.index("const DEF=")
    src = HTML[start : HTML.index("};", start) + 2]
    modes = HTML[HTML.index("const modeOverrides=") : HTML.index("const KEY=")]
    js = src + modes + "console.log(JSON.stringify(MODES))"
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert list(out) == re.findall(r'data-p="([^"]+)"', HTML) == ["standard", "near", "lowpower"]
    assert all(set(values) == set(out["standard"]) for values in out.values())
    assert not ({"face_roi", "pose_roi", "distance_calib", "asr_provider", "reid_consent"} & out["standard"].keys())

    near = out["near"]
    want = dict(
        tracker_mode="kalman",
        presence_model="hmm",
        intent_model="score",
        pose_enabled=True,
        distance_enabled=True,
        zone_max=1.3,
        audio_pipeline="puresound",
        near_range=1.3,
        iso_dry_blend=1,
        iso_onset_guard=False,
        xcheck_recovery_mode="audio",
        xcheck_recover_mix=0.3,
        adaptive_fps=False,
        fps=2,
    )
    assert {k: near[k] for k in want} == want
    assert out["standard"]["audio_pipeline"] == out["lowpower"]["audio_pipeline"] == "energy_vad"
    assert out["lowpower"]["fps_high"] == 1 and out["lowpower"]["max_width"] == 480
    for name, settings in out.items():
        cfg = Config()
        settings = {k: v for k, v in settings.items() if k not in CLIENT_ONLY}
        assert cfg.update(settings) == [], name
        actual = asdict(cfg)
        assert settings == {k: actual[k] for k in settings}


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
@pytest.mark.parametrize(
    "saved,expected",
    [
        (None, ("puresound", "audio", 2)),
        ({}, ("energy_vad", "legacy", 1)),
        (
            {"_v091": 1, "audio_pipeline": "energy_vad", "xcheck_recovery_mode": "audiovisual", "fps": 3},
            ("energy_vad", "audiovisual", 3),
        ),
    ],
)
def test_near_mode_defaults_only_for_browsers_without_saved_settings(saved, expected):
    start = HTML.index("const DEF=")
    defaults = HTML[start : HTML.index("};", start) + 2]
    startup = HTML[HTML.index("const modeOverrides=") : HTML.index("const save=")]
    stored = json.dumps(saved) if saved is not None else None
    js = f"const localStorage={{getItem:()=>{json.dumps(stored)}}};" + defaults + startup
    js += "console.log(JSON.stringify([cfg.audio_pipeline,cfg.xcheck_recovery_mode,cfg.fps]));"
    result = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
    assert tuple(json.loads(result.stdout)) == expected

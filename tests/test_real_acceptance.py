"""Offline pack safety and operator-facing contracts."""

import json
import plistlib
import subprocess
import sys

import pytest

from presencekit import real_acceptance as real
from presencekit.config import Config


def test_profile_is_config_valid_and_cloud_off():
    profile = real.profile()
    client = {"fps", "monitor"}
    config = Config()
    assert config.update({k: v for k, v in profile.items() if k not in client}) == []
    for key, value in profile.items():
        if key not in client:
            actual = getattr(config, key)
            assert list(actual) == value if isinstance(actual, tuple) else actual == value
    assert profile["fps"] == 2
    assert profile["pose_enabled"]
    assert not any(profile[k] for k in ("reid_enabled", "reid_consent", "reid_show_crops"))
    assert profile["reid_threshold"] == config.reid_threshold
    assert profile["audio_pipeline"] == "puresound"
    assert profile["asr_provider"] == "none"


def test_prepare_never_marks_real_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(real, "preflight", lambda: {"status": "pending_real"})
    directory = real.prepare(tmp_path / "new-run")
    assert {p.name for p in directory.iterdir()} == {
        "profile.json",
        "preflight.json",
        "manual-results.json",
        "README.md",
    }
    result = json.loads((directory / "manual-results.json").read_text())
    assert result["status"] == "pending_real"
    assert len(result["cases"]) >= 19
    assert all(case["status"] == "pending_real" for case in result["cases"])
    assert all(not value for value in result["confirmations"].values())
    assert all(case["ground_truth"] is None for case in result["cases"])
    assert all(value is None for case in result["cases"] for value in case["metrics"].values())
    assert "api_key" not in json.dumps(result).lower()


@pytest.mark.parametrize("populated", [False, True])
def test_existing_directory_untouched_before_preflight(tmp_path, monkeypatch, populated):
    output = tmp_path / "run"
    output.mkdir()
    if populated:
        (output / "manual-results.json").write_text("operator evidence")
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    monkeypatch.setattr(real, "preflight", lambda: pytest.fail("must refuse before collecting metadata"))
    with pytest.raises(FileExistsError):
        real.prepare(output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_failure_before_write_leaves_no_output(tmp_path, monkeypatch):
    def fail():
        raise ValueError("invalid metadata")

    monkeypatch.setattr(real, "preflight", fail)
    with pytest.raises(ValueError):
        real.prepare(tmp_path / "run")
    assert list(tmp_path.iterdir()) == []


def test_chrome_metadata_only_and_invalid_version(tmp_path):
    path = tmp_path / "Info.plist"
    path.write_bytes(plistlib.dumps({"CFBundleShortVersionString": "130.0.1.2", "Secret": "DO_NOT_COPY"}))
    assert real.chrome_version((path,)) == {"status": "installed_metadata_only", "version": "130.0.1.2"}
    path.write_bytes(plistlib.dumps({"CFBundleShortVersionString": "unexpected private content"}))
    assert real.chrome_version((path,)) == {"status": "unverified", "version": None}
    path.write_bytes(b"invalid plist")
    assert real.chrome_version((path,)) == {"status": "unverified", "version": None}


def test_preflight_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(real, "ROOT", tmp_path)
    monkeypatch.setattr(real, "chrome_version", lambda: {"status": "unverified", "version": None})
    output = real.preflight()
    assert set(output["packages"]) == set(real.PACKAGES)
    assert set(output["models_default_directory_only"]) == set(real.MODEL_FILES)
    assert not any(output["models_default_directory_only"].values())
    assert set(output["manual_checks"].values()) == {"unverified"}
    assert "node" not in output["os"]  # host name is not evidence


def test_stdlib_only_import_and_help():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'cv2','numpy','onnxruntime','puresound_streaming'} or fullname in {
            'presencekit.config', 'presencekit.tracker', 'presencekit.reid'}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
from presencekit.real_acceptance import main
main(['--help'])
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=real.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout


def test_import_profile_against_actual_ui(tmp_path):
    """Execute the UI DEF/META/clamp import path without DOM, camera or browser."""
    import shutil

    if not shutil.which("node"):
        pytest.skip("node unavailable")
    settings = (real.ROOT / "presencekit/static/js/settings.js").read_text()
    start = settings.index("const DEF=")
    end = settings.index("const REC=")
    # DEF/META are declarations only; the helper is copied from its literal declaration.
    declarations = settings[start:end]
    clamp = next(line for line in settings.splitlines() if line.startswith("const clamp="))
    script = (
        declarations
        + "\n"
        + clamp
        + "\nconst j="
        + json.dumps(real.profile())
        + ";\n"
        + """
for (const [k,v] of Object.entries(j)) {
  if (!(k in DEF)) throw new Error('unknown UI key '+k);
  const actual=BOXES.includes(k)?v.slice(0,4).map(x=>Math.min(1,Math.max(0,+x))):clamp(k,v);
  if (JSON.stringify(actual)!==JSON.stringify(v)) throw new Error('UI clamps '+k);
}
"""
    )
    path = tmp_path / "check.js"
    path.write_text(script)
    result = subprocess.run(["node", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_real_device_cases_keep_cloud_disabled_semantics():
    cases = {case["id"]: case for case in real.results_template()["cases"]}
    assert cases["device.same_page_reconnect"]["expected"] == "new_session"
    assert cases["device.refresh_clears"]["expected"] == "cleared"
    assert cases["device.server_restart_clears"]["expected"] == "cleared"
    for name in ("device.refresh_clears", "device.server_restart_clears"):
        assert cases[name]["prerequisites"] == [
            "approved_jev_phase",
            "nonempty_reference_history_verified",
        ]
    assert cases["jev.same_person"]["expected"] == "resume"
    assert cases["jev.service_error"]["expected"] == "unassociated"
    assert cases["jev.ambiguous_overlap"]["participants"] == ["A", "B"]
    for count in (1, 2, 3):
        name = "device.quiet_120s" + (f"_{count}" if count > 1 else "")
        assert len(cases[name]["participants"]) == count
        assert cases[name]["scenario_seconds"] == 120
    for name in ("turned_head_3s", "bystander_not_extend", "background_speech_not_extend"):
        assert f"device.{name}" in cases
    allowed = real.results_template()["schema"]["ground_truth_allowed"]
    assert all(case["expected"] in allowed for case in cases.values())


def test_numeric_evidence_fields_and_response_version_are_bounded():
    import re

    template = real.results_template()
    bounds = template["schema"]["metric_bounds"]
    assert bounds["sample_coverage"] == [0, 1]
    for name in (
        "valid_sample_count",
        "queue_peak",
        "send_audio_dropped",
        "jev_calls",
        "jev_input_tokens",
        "jev_latency_p95_ms",
        "true_match_count",
        "false_match_count",
        "true_nonmatch_count",
        "false_nonmatch_count",
    ):
        assert bounds[name][0] == 0
        assert isinstance(bounds[name][1], int)
    pattern = template["schema"]["jev_response_model_pattern"]
    assert re.fullmatch(pattern, "jev-1.13.0")
    assert not re.fullmatch(pattern, "operator free text")
    assert template["manual_environment"]["jev_response_model"] is None

"""Observable report/CLI contracts; scenario semantics live in acceptance_cases tests."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from presencekit.acceptance import exit_code, main, run_acceptance

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def adapter(monkeypatch):
    module = ModuleType("presencekit.acceptance_cases")
    module.PROFILE_CONFIG = {"priority_mode": "vision", "cloud_asr": False}

    def cases(seeds):
        return [
            {
                "id": "quiet_present",
                "suite": "functional",
                "status": "passed",
                "expected": "not ENDED",
                "actual": "IDLE",
                "details": {"seeds": list(range(seeds))},
            }
        ]

    module.functional_cases = cases
    monkeypatch.setitem(sys.modules, "presencekit.acceptance_cases", module)
    # Default fake has no pipeline adapters; unit tests must never start a server.
    monkeypatch.setitem(sys.modules, "presencekit.acceptance_pipeline", ModuleType("presencekit.acceptance_pipeline"))
    monkeypatch.setitem(sys.modules, "presencekit.acceptance_browser", ModuleType("presencekit.acceptance_browser"))
    monkeypatch.setitem(
        sys.modules, "presencekit.acceptance_browser_reid", ModuleType("presencekit.acceptance_browser_reid")
    )
    return module


def test_report_distinguishes_coverage_and_writes_both_formats(adapter, tmp_path):
    report = run_acceptance(seeds=3, output_dir=tmp_path)
    assert report["summary"] == {
        "status": "incomplete",
        "counts": {"passed": 1, "failed": 0, "not_run": 4, "pending_real": 2},
        "total": 7,
        "poc_accepted": False,
    }
    assert json.loads((tmp_path / "acceptance.json").read_text()) == report
    markdown = (tmp_path / "acceptance.md").read_text()
    for status in ("passed", "not_run", "pending_real"):
        assert status in markdown
    assert "real_chrome_devices" in markdown
    assert "real_jev_quality" in markdown
    assert report["profile"]["suite_configs"]["functional"] == adapter.PROFILE_CONFIG
    assert report["thresholds"]["silent_present_seconds"] == 120


def test_failed_cases_take_precedence_over_incomplete(adapter):
    original = adapter.functional_cases

    def failed(seeds):
        cases = original(seeds)
        cases[0]["status"] = "failed"
        cases[0]["actual"] = "ENDED"
        return cases

    adapter.functional_cases = failed
    report = run_acceptance()
    assert report["summary"]["status"] == "failed"
    assert exit_code(report) == 1
    assert exit_code(report, strict=True) == 1


def test_unimplemented_and_empty_suites_cannot_pass(adapter):
    report = run_acceptance(suite="all")
    statuses = {case["suite"]: case["status"] for case in report["cases"]}
    assert statuses["pipeline"] == statuses["browser"] == statuses["browser-reid"] == statuses["soak"] == "not_run"
    assert report["summary"]["poc_accepted"] is False
    adapter.functional_cases = lambda seeds: []
    report = run_acceptance()
    assert report["summary"]["counts"]["passed"] == 0
    assert report["summary"]["counts"]["not_run"] == 5


def test_selecting_pipeline_never_executes_functional_cases(adapter):
    def forbidden(seeds):
        pytest.fail("unselected suite must not execute")

    adapter.functional_cases = forbidden
    report = run_acceptance(suite="pipeline")
    assert report["summary"]["counts"]["failed"] == 0
    assert report["summary"]["counts"]["not_run"] == 5


def test_pipeline_and_soak_receive_requested_durations(adapter):
    pipeline_module = sys.modules["presencekit.acceptance_pipeline"]
    calls = []

    def suite_case(suite, seeds, seconds):
        calls.append((suite, seeds, seconds))
        return [
            {
                "id": suite + ".short",
                "suite": suite,
                "status": "not_run",
                "expected": "Full duration",
                "actual": seconds,
                "details": {"qualification": False},
            }
        ]

    pipeline_module.run_pipeline = lambda seeds, seconds: suite_case("pipeline", seeds, seconds)
    pipeline_module.run_soak = lambda seeds, seconds: suite_case("soak", seeds, seconds)
    report = run_acceptance(suite="all", seeds=2, pipeline_seconds=5, soak_seconds=6)
    assert calls == [("pipeline", 2, 5), ("soak", 2, 6)]
    assert report["run"]["pipeline_seconds"] == 5
    assert report["run"]["soak_seconds"] == 6
    assert report["summary"]["status"] == "incomplete"


@pytest.mark.parametrize(
    "arg,value",
    [
        ("--pipeline-seconds", "nan"),
        ("--browser-seconds", "nan"),
        ("--browser-seconds", "0"),
        ("--soak-seconds", "0"),
        ("--soak-seconds", "3601"),
    ],
)
def test_invalid_cli_durations_fail_without_running(adapter, arg, value, tmp_path):
    with pytest.raises(SystemExit) as result:
        main([arg, value, "--output-dir", str(tmp_path / "never")])
    assert result.value.code == 2
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize(
    "bad_result",
    [
        None,
        [{"id": "missing-fields"}],
        [{"id": "wrong-suite", "suite": "soak", "status": "passed", "expected": True, "actual": True, "details": {}}],
    ],
)
def test_malformed_adapter_cannot_claim_success(adapter, bad_result):
    adapter.functional_cases = lambda seeds: bad_result
    report = run_acceptance()
    assert report["summary"]["counts"]["failed"] == 1
    assert report["summary"]["counts"]["passed"] == 0
    assert exit_code(report) == 1


def test_adapter_crash_is_a_reported_failure(adapter, tmp_path):
    def broken(seeds):
        raise RuntimeError("fixture adapter failed")

    adapter.functional_cases = broken
    report = run_acceptance(output_dir=tmp_path)
    assert report["summary"]["counts"]["failed"] == 1
    assert report["cases"][0]["actual"] == "RuntimeError"
    assert "fixture adapter failed" in (tmp_path / "acceptance.md").read_text()


@pytest.mark.parametrize("seeds", [0, -1, 1.5, True])
def test_invalid_seed_counts_never_run_cases(adapter, seeds, tmp_path):
    def forbidden(seeds):
        pytest.fail("invalid input must not execute any suite")

    adapter.functional_cases = forbidden
    with pytest.raises(ValueError, match="positive integer"):
        run_acceptance(seeds=seeds, output_dir=tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_fixed_inputs_are_reproducible_without_leaking_environment(adapter, monkeypatch):
    monkeypatch.setenv("ACCEPTANCE_TEST_SECRET", "secret-that-must-not-be-in-report")
    first = run_acceptance(seeds=4)
    second = run_acceptance(seeds=4)
    for key in ("cases", "profile", "thresholds", "run", "summary", "environment"):
        assert first[key] == second[key]
    assert "secret-that-must-not-be-in-report" not in json.dumps(first)
    assert "environ" not in first["environment"]
    assert first["created_at"]
    assert first["elapsed_seconds"] >= 0


def test_cli_exit_codes_do_not_claim_full_acceptance(adapter, tmp_path, capsys):
    assert main(["--seeds", "1", "--output-dir", str(tmp_path)]) == 0
    assert "Acceptance: incomplete" in capsys.readouterr().out
    assert main(["--strict", "--output-dir", str(tmp_path)]) == 2
    adapter.functional_cases = lambda seeds: [
        {"id": "failure", "suite": "functional", "status": "failed", "expected": True, "actual": False, "details": {}}
    ]
    assert main(["--output-dir", str(tmp_path)]) == 1


def test_cli_help_and_import_are_offline_and_leave_no_artifacts(tmp_path):
    # Deny sockets and model/device modules before import/help, rather than trusting a printed assurance.
    script = ROOT / "scripts" / "run_acceptance.py"
    bootstrap = """
import builtins, runpy, socket, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'cv2', 'onnxruntime', 'puresound', 'puresound_streaming'}:
        raise AssertionError('device/model import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
socket.socket = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('network access'))
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", bootstrap, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--suite" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_invalid_cli_arguments_exit_before_writing(tmp_path):
    with pytest.raises(SystemExit) as result:
        main(["--seeds", "0", "--output-dir", str(tmp_path / "never")])
    assert result.value.code == 2
    assert not (tmp_path / "never").exists()


def test_real_functional_reconnect_and_sensor_contracts():
    from presencekit.acceptance_cases import functional_cases

    cases = {case["id"]: case for case in functional_cases(seeds=1)}
    reconnect = cases["transport.same_page_reconnect"]
    assert reconnect["status"] == "passed", reconnect
    for contract in (
        "new_session",
        "resumed_original_conversation",
        "stale_release_rejected",
        "current_owner_unchanged",
        "other_page_isolated",
        "normal_end_ttl_not_refreshed",
        "expired_media_cleared",
        "active_interruption_preserved",
        "interruption_not_normal_end",
    ):
        assert reconnect["actual"][contract] is True
    sensor_gap = cases["transport.sensor_gap"]
    assert sensor_gap["status"] == "passed", sensor_gap
    assert sensor_gap["actual"]["false_departures"] == 0
    assert sensor_gap["actual"]["same_session_after_recovery"] is True


def test_functional_presence_contract_coverage_is_explicit():
    from presencekit.acceptance_cases import functional_cases

    cases = {case["id"]: case for case in functional_cases(seeds=1)}
    for people, case_id in (
        (1, "presence.quiet_120s.seed_0"),
        (2, "presence.quiet_120s.people_2.seed_0"),
        (3, "presence.quiet_120s.people_3.seed_0"),
    ):
        case = cases[case_id]
        assert case["status"] == "passed", case
        assert case["actual"]["joined"] == people
        assert case["actual"]["false_endings"] == 0
        assert case["details"]["simulated_seconds"] == 130
    for suffix in ("", ".remaining_1", ".remaining_2"):
        case = cases[f"presence.remaining_participant{suffix}.seed_0"]
        assert case["status"] == "passed", case
        assert case["actual"]["joined"] == 3
        assert case["actual"]["false_endings"] == 0
    for scenario in ("unjoined_bystander", "unjoined_passerby", "background_speech", "tv_audio_flag"):
        case = cases[f"presence.{scenario}.seed_0"]
        assert case["status"] == "passed", case
        assert case["actual"]["ended_at_s"] <= 20
        assert case["actual"]["candidate_joined"] is False
        assert case["actual"]["new_sessions"] == 0
    for scenario in ("turned_head_3s", "pose_only_3s"):
        case = cases[f"presence.{scenario}.seed_0"]
        assert case["status"] == "passed", case
        assert case["actual"]["false_endings"] == 0
        assert case["actual"]["same_participant_recovered"] is True
    assert cases["presence.pose_only_3s.seed_0"]["actual"]["pose_only_participant_observed"] is True
    assert cases["presence.tv_visual_rejection"]["status"] == "not_run"
    assert "no screen-person semantic label" in cases["presence.tv_visual_rejection"]["details"]["reason"]


@pytest.mark.parametrize(
    "suite,module_name,adapter_name",
    [
        ("browser", "presencekit.acceptance_browser", "run_browser"),
        ("browser-reid", "presencekit.acceptance_browser_reid", "run_browser_reid"),
    ],
)
def test_browser_suite_is_isolated_and_does_not_accept_real_devices(
    adapter, monkeypatch, tmp_path, suite, module_name, adapter_name
):
    browser = sys.modules[module_name]
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("browser suite must not execute another suite")

    adapter.functional_cases = forbidden
    pipeline = sys.modules["presencekit.acceptance_pipeline"]
    pipeline.run_pipeline = pipeline.run_soak = forbidden

    def run_browser(seeds, seconds):
        calls.append((seeds, seconds))
        return [
            {
                "id": suite + ".synthetic",
                "suite": suite,
                "status": "passed",
                "expected": True,
                "actual": True,
                "details": {"synthetic": True},
            }
        ]

    setattr(browser, adapter_name, run_browser)
    report = run_acceptance(suite=suite, seeds=2, browser_seconds=7, output_dir=tmp_path)
    assert calls == [(2, 7)]
    assert report["run"]["selected_suites"] == [suite]
    assert report["run"]["browser_seconds"] == 7
    assert report["summary"]["counts"] == {"passed": 1, "failed": 0, "not_run": 4, "pending_real": 2}
    assert report["summary"]["poc_accepted"] is False
    assert exit_code(report, strict=True) == 2
    assert all(case["status"] == "pending_real" for case in report["cases"] if case["suite"] == "real")
    assert main(["--suite", suite, "--browser-seconds", "7", "--output-dir", str(tmp_path)]) == 0
    assert calls[-1] == (20, 7)


def test_all_includes_browser_in_order_and_propagates_failure(adapter):
    calls = []

    def run(suite, seeds, seconds):
        calls.append((suite, seconds))
        return [
            {
                "id": suite + ".probe",
                "suite": suite,
                "status": "failed" if suite == "browser" else "passed",
                "expected": True,
                "actual": suite != "browser",
                "details": {},
            }
        ]

    pipeline = sys.modules["presencekit.acceptance_pipeline"]
    pipeline.run_pipeline = lambda seeds, seconds: run("pipeline", seeds, seconds)
    pipeline.run_soak = lambda seeds, seconds: run("soak", seeds, seconds)
    sys.modules["presencekit.acceptance_browser"].run_browser = lambda seeds, seconds: run("browser", seeds, seconds)
    sys.modules["presencekit.acceptance_browser_reid"].run_browser_reid = lambda seeds, seconds: run(
        "browser-reid", seeds, seconds
    )
    report = run_acceptance(suite="all", pipeline_seconds=3, browser_seconds=4, soak_seconds=5)
    assert calls == [("pipeline", 3), ("browser", 4), ("browser-reid", 4), ("soak", 5)]
    assert report["summary"]["status"] == "failed"
    assert exit_code(report) == 1


@pytest.mark.parametrize("seconds", [True, float("nan"), float("inf"), 0, 3601])
def test_invalid_browser_duration_cannot_execute(adapter, seconds):
    with pytest.raises(ValueError, match="browser_seconds"):
        run_acceptance(browser_seconds=seconds)

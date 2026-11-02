"""Offline PoC acceptance orchestration and reports.

Importing this module does not load models, open devices, or contact services.
Cases are loaded only when a requested suite is executed.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
SUITES = ("functional", "pipeline", "browser", "browser-reid", "soak")
STATUSES = ("passed", "failed", "not_run", "pending_real")
THRESHOLDS = {
    "silent_present_seconds": 120,
    "occlusion_seconds": 3,
    "last_participant_leave_to_ended_seconds_max": 20,
    "reid_deadline_seconds": 10,
    "history_ttl_seconds": 600,
    "history_conversations_max": 3,
    "participants_max": 3,
    "soak_seconds": 1800,
    "recent_5s_rtf_p95_max": 0.5,
    "audio_dropped_max": 0,
    "uncaught_errors_max": 0,
    "rss_last_vs_first_10min_growth_mb_max": 30,
}
LIMITATIONS = [
    "Synthetic detections and mock identity responses do not establish camera or biometric accuracy.",
    "Real Chrome camera/microphone and consented Jev quality tests require separate acceptance.",
    "Functional virtual time does not measure real-time throughput or memory stability.",
    "Only an executed pipeline/soak adapter can establish pipeline or 30-minute stability results.",
    "Headless Chrome with synthetic media validates browser integration, not physical devices or Jev quality.",
    "Browser re-ID uses a local mock comparison backend and does not establish biometric recognition quality.",
    "Seeds and profile are fixed inputs; timestamps, elapsed time and host metadata may vary.",
    "Audio RTF 0.37 has not been fully explained and is not claimed fixed by this report.",
]
DEFAULT_PROFILE = {
    "name": "m4-chrome-leave-poc-v1",
    "target": "Apple M4 + Chrome + local server + one active connection",
    "input": "synthetic/mock; no physical camera/microphone or cloud calls",
    "priority": "avoid falsely ending an existing participant's conversation",
    "session_policy": "new session, provisional conversation, explicit handoff only on unique overlap",
    "cloud_asr": False,
}


def _command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def environment_metadata() -> dict:
    """Collect an allowlist of reproducibility metadata, never environment variables."""
    root = Path(__file__).resolve().parent.parent
    versions = {}
    for package in ("presencekit", "numpy", "onnxruntime", "puresound-streaming", "opencv-python-headless", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    dirty = _command(["git", "status", "--porcelain", "--untracked-files=normal"], root)
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        cpu = _command(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
    return {
        "git_head": _command(["git", "rev-parse", "HEAD"], root),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "python": platform.python_version(),
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "cpu": cpu,
        "packages": versions,
    }


def _case(case_id: str, suite: str, status: str, expected, actual, details) -> dict:
    return {"id": case_id, "suite": suite, "status": status, "expected": expected, "actual": actual, "details": details}


def _validate_cases(cases, suite: str) -> list[dict]:
    if not isinstance(cases, list):
        raise ValueError("suite adapter must return a list of case dictionaries")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not all(
            key in case for key in ("id", "suite", "status", "expected", "actual", "details")
        ):
            raise ValueError("each case requires id, suite, status, expected, actual and details")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in seen:
            raise ValueError("case IDs must be nonempty and unique within a suite")
        if case["suite"] != suite or case["status"] not in STATUSES:
            raise ValueError("invalid case suite or status")
        seen.add(case["id"])
    # Reject values that cannot be represented by the report, including NaN.
    json.dumps(cases, allow_nan=False)
    return cases


def _execute_suite(
    suite: str, seeds: int, *, pipeline_seconds: float, browser_seconds: float, soak_seconds: float
) -> tuple[list[dict], dict]:
    try:
        modules = {
            "functional": "presencekit.acceptance_cases",
            "browser": "presencekit.acceptance_browser",
            "browser-reid": "presencekit.acceptance_browser_reid",
            "pipeline": "presencekit.acceptance_pipeline",
            "soak": "presencekit.acceptance_pipeline",
        }
        module = importlib.import_module(modules[suite])
        profile = getattr(module, "PROFILE_CONFIG", {})
        adapter = getattr(
            module, "functional_cases" if suite == "functional" else f"run_{suite.replace('-', '_')}", None
        )
        if adapter is None:
            return [
                _case(
                    f"{suite}_not_implemented",
                    suite,
                    "not_run",
                    "Executed acceptance measurements",
                    None,
                    "Adapter is not available; no performance or stability result is claimed.",
                )
            ], profile
        if suite == "functional":
            results = adapter(seeds)
        else:
            seconds = {
                "pipeline": pipeline_seconds,
                "browser": browser_seconds,
                "browser-reid": browser_seconds,
                "soak": soak_seconds,
            }[suite]
            results = adapter(seeds, seconds=seconds)
            profile = {"seconds": seconds, "method": getattr(module, "METHOD", {})}
        cases = _validate_cases(results, suite)
        if not cases:
            cases = [_case(f"{suite}_empty", suite, "not_run", "At least one executed case", [], "No cases returned.")]
        return cases, profile
    except Exception as exc:
        return [
            _case(
                f"{suite}_execution_error",
                suite,
                "failed",
                "Suite completes without an exception",
                type(exc).__name__,
                str(exc),
            )
        ], {}


def _real_cases() -> list[dict]:
    return [
        _case(
            "real_chrome_devices",
            "real",
            "pending_real",
            "M4 + Chrome with real camera/microphone passes contract",
            None,
            "Requires separately scheduled physical-device acceptance; never inferred from synthetic tests.",
        ),
        _case(
            "real_jev_quality",
            "real",
            "pending_real",
            "Consented real subjects satisfy re-ID quality and deadline",
            None,
            "Requires consented samples and the real Jev backend; no cloud request is made by this runner.",
        ),
    ]


def run_acceptance(
    suite: str = "functional",
    seeds: int = 20,
    output_dir: str | Path | None = None,
    strict: bool = False,
    *,
    pipeline_seconds: float = 10,
    browser_seconds: float = 10,
    soak_seconds: float = 1800,
) -> dict:
    if suite not in (*SUITES, "all"):
        raise ValueError(f"Unknown suite: {suite}")
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise ValueError("seeds must be a positive integer")
    for name, duration in (
        ("pipeline_seconds", pipeline_seconds),
        ("browser_seconds", browser_seconds),
        ("soak_seconds", soak_seconds),
    ):
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or not 3 <= duration <= 3600
        ):
            raise ValueError(f"{name} must be finite and in 3..3600")
    started = time.perf_counter()
    selected = list(SUITES) if suite == "all" else [suite]
    cases = []
    profiles = {}
    for selected_suite in selected:
        results, profile = _execute_suite(
            selected_suite,
            seeds,
            pipeline_seconds=pipeline_seconds,
            browser_seconds=browser_seconds,
            soak_seconds=soak_seconds,
        )
        cases.extend(results)
        if profile:
            profiles[selected_suite] = profile
    # Unrequested automated suites remain visible, rather than silently implying coverage.
    for unselected in SUITES:
        if unselected not in selected:
            cases.append(
                _case(
                    f"{unselected}_not_selected",
                    unselected,
                    "not_run",
                    "Suite executed",
                    None,
                    "Not selected for this invocation.",
                )
            )
    cases.extend(_real_cases())
    counts = {status: 0 for status in STATUSES}
    counts.update(Counter(case["status"] for case in cases))
    status = "failed" if counts["failed"] else "incomplete" if counts["not_run"] or counts["pending_real"] else "passed"
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "environment": environment_metadata(),
        "run": {
            "suite": suite,
            "selected_suites": selected,
            "seeds": seeds,
            "strict": strict,
            "pipeline_seconds": pipeline_seconds,
            "browser_seconds": browser_seconds,
            "soak_seconds": soak_seconds,
        },
        "profile": {**DEFAULT_PROFILE, "suite_configs": profiles},
        "thresholds": dict(THRESHOLDS),
        "summary": {"status": status, "counts": counts, "total": len(cases), "poc_accepted": status == "passed"},
        "cases": cases,
        "limitations": list(LIMITATIONS),
    }
    if output_dir is not None:
        write_report(report, output_dir)
    return report


def _cell(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text.replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def markdown_report(report: dict) -> str:
    counts = report["summary"]["counts"]
    lines = [
        "# PoC acceptance report",
        "",
        f"Overall status: **{report['summary']['status']}**.",
        "",
        "Full PoC acceptance requires real-device and consented Jev tests; synthetic passes do not establish it.",
        "",
        f"Created: {report['created_at']} · elapsed: {report['elapsed_seconds']:.3f} s",
        "",
        "| Status | Count |",
        "| --- | ---: |",
        *[f"| {status} | {counts[status]} |" for status in STATUSES],
        "",
        "## Run and environment",
        "",
        "```json",
        json.dumps({"run": report["run"], "environment": report["environment"]}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Profile and thresholds",
        "",
        "```json",
        json.dumps({"profile": report["profile"], "thresholds": report["thresholds"]}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Cases",
        "",
        "| Suite | Case | Status | Expected | Actual | Details |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        lines.append(
            "| "
            + " | ".join(_cell(case[key]) for key in ("suite", "id", "status", "expected", "actual", "details"))
            + " |"
        )
    lines.extend(["", "## Method and limitations", "", *[f"- {item}" for item in report["limitations"]], ""])
    return "\n".join(lines)


def write_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    # Serialize before touching the filesystem, so bad adapter data cannot leave half a report.
    json_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown = markdown_report(report)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = destination / "acceptance.json", destination / "acceptance.md"
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path


def exit_code(report: dict, strict: bool = False) -> int:
    if report["summary"]["counts"]["failed"]:
        return 1
    return 2 if strict and report["summary"]["status"] != "passed" else 0


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _duration(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be finite and in 3..3600 seconds") from exc
    if not math.isfinite(seconds) or not 3 <= seconds <= 3600:
        raise argparse.ArgumentTypeError("must be finite and in 3..3600 seconds")
    return seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="functional")
    parser.add_argument(
        "--seeds", type=_positive_int, default=20, help="Number of deterministic scenario seeds (default: 20)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/acceptance"))
    parser.add_argument("--pipeline-seconds", type=_duration, default=10, help="Measured pipeline seconds after warmup")
    parser.add_argument(
        "--browser-seconds",
        type=_duration,
        default=10,
        help="Synthetic Chrome streaming seconds for browser/browser-reid; fault checks take additional time",
    )
    parser.add_argument(
        "--soak-seconds",
        type=_duration,
        default=1800,
        help="Measured soak seconds after warmup; below 1800 cannot qualify 30-minute gates",
    )
    parser.add_argument("--strict", action="store_true", help="Exit 2 when any acceptance remains not_run/pending_real")
    args = parser.parse_args(argv)
    report = run_acceptance(
        args.suite,
        args.seeds,
        args.output_dir,
        args.strict,
        pipeline_seconds=args.pipeline_seconds,
        browser_seconds=args.browser_seconds,
        soak_seconds=args.soak_seconds,
    )
    counts = report["summary"]["counts"]
    print(f"Acceptance: {report['summary']['status']}; " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"Reports: {args.output_dir / 'acceptance.json'} and {args.output_dir / 'acceptance.md'}")
    return exit_code(report, args.strict)

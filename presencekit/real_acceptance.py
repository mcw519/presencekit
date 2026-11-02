"""Offline preparation for operator-run device/Jev acceptance. Standard library only.

This module never captures media, imports inference runtimes, reads credentials or
contacts a server. Generated records are pending evidence, never test results.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
import platform
import plistlib
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "presencekit",
    "fastapi",
    "uvicorn",
    "opencv-python-headless",
    "numpy",
    "psutil",
    "ai-edge-litert",
    "puresound-streaming",
    "onnxruntime",
)
MODEL_FILES = (
    "face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx",
    "movenet_multipose_lightning_fp16.tflite",
    "dpcrn_curriculum_v2.onnx",
    "dpcrn_curriculum_v2.json",
)
THRESHOLDS = (
    "reid_threshold",
    "reid_uncertain_low",
    "reid_min_face",
    "reid_face_wait",
    "reid_timeout",
    "reid_cooldown",
    "reid_cap_per_hour",
    "reid_cap_total",
    "reid_cost_per_call",
)
CASES = (
    ("device.quiet_120s", ["A"], "same_session", 120),
    ("device.quiet_120s_2", ["A", "B"], "same_session", 120),
    ("device.quiet_120s_3", ["A", "B", "C"], "same_session", 120),
    ("device.turned_head_3s", ["A"], "same_session", 3),
    ("device.bystander_not_extend", ["A", "B"], "ended", 20),
    ("device.background_speech_not_extend", ["A"], "ended", 20),
    ("device.occlusion_3s", ["A"], "same_session", 3),
    ("device.last_leave_20s", ["A"], "ended", 20),
    ("device.remaining_1", ["A"], "same_session", 40),
    ("device.remaining_2", ["A", "B"], "same_session", 40),
    ("device.remaining_3", ["A", "B", "C"], "same_session", 40),
    ("device.sensor_interruption", ["A"], "same_session", 25),
    ("device.same_page_reconnect", ["A"], "new_session", 10),
    ("device.refresh_clears", ["A"], "cleared", 10),
    ("device.server_restart_clears", ["A"], "cleared", 10),
    ("jev.same_person", ["A"], "resume", 10),
    ("jev.different_person", ["B"], "unassociated", 10),
    ("jev.member_overlap", ["B", "C"], "resume", 10),
    ("jev.ambiguous_overlap", ["A", "B"], "unassociated", 10),
    ("jev.unknown_timeout", ["A"], "unassociated", 10),
    ("jev.service_error", ["A"], "unassociated", 10),
    ("jev.revoke", ["A"], "cleared", 10),
    ("jev.ttl_600s", ["A"], "unassociated", 600),
    ("jev.max_3_history", ["A", "B", "C"], "history_bounded", 600),
    ("device.normal_load_30min", ["A"], "performance_gate", 1800),
)
# Values must be numeric or null, with no raw face/audio/identifier payload fields.
METRIC_BOUNDS = {
    "elapsed_seconds": [0, 86400],
    "decision_seconds": [0, 86400],
    "false_ended_count": [0, 10000],
    "wrong_handoff_count": [0, 10000],
    "history_conversations": [0, 10000],
    "rtf_p95": [0, 100],
    "audio_dropped": [0, 1000000000],
    "uncaught_errors": [0, 1000000000],
    "queue_start": [0, 1000000],
    "queue_end": [0, 1000000],
    "rss_growth_mib": [-1000000, 1000000],
    "valid_sample_count": [0, 1000000000],
    "sample_coverage": [0, 1],
    "queue_peak": [0, 1000000],
    "send_audio_dropped": [0, 1000000000],
    "jev_calls": [0, 1000000],
    "jev_input_tokens": [0, 1000000000],
    "jev_latency_p95_ms": [0, 86400000],
    "true_match_count": [0, 1000000],
    "false_match_count": [0, 1000000],
    "true_nonmatch_count": [0, 1000000],
    "false_nonmatch_count": [0, 1000000],
}


def profile() -> dict:
    """Read only literal Config defaults; never execute its environment factory."""
    tree = ast.parse((ROOT / "presencekit/config.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Config")
    defaults = {
        n.target.id: ast.literal_eval(n.value)
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in THRESHOLDS
    }
    return {
        **defaults,
        "tracker_mode": "kalman",
        "presence_model": "hmm",
        "intent_model": "score",
        "priority_mode": "vision",
        "join_seconds": 1.5,
        "grace_seconds": 10,
        "face_roi": [0.08, 0.02, 0.92, 0.98],
        "pose_roi": [0, 0, 1, 1],
        "pose_enabled": True,
        "fps": 2,
        "adaptive_fps": True,
        "fps_high": 2,
        "fps_low": 0.5,
        "audio_pipeline": "puresound",
        "asr_provider": "none",
        "reid_enabled": False,
        "reid_consent": False,
        "reid_show_crops": False,
        "reid_ttl": 600,
        "reid_max_refs": 3,
        "reid_decision_timeout": 10,
    }


def chrome_version(paths: tuple[Path, ...] | None = None) -> dict:
    paths = (
        paths
        if paths is not None
        else (
            Path("/Applications/Google Chrome.app/Contents/Info.plist"),
            Path.home() / "Applications/Google Chrome.app/Contents/Info.plist",
        )
    )
    for path in paths:
        try:
            with path.open("rb") as stream:
                value = plistlib.load(stream).get("CFBundleShortVersionString")
            if isinstance(value, str) and len(value) <= 64 and all(c in "0123456789." for c in value):
                return {"status": "installed_metadata_only", "version": value}
        except (OSError, ValueError, plistlib.InvalidFileException):
            continue
    return {"status": "unverified", "version": None}


def preflight() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": 1,
        "mode": "offline_preparation",
        "status": "pending_real",
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "cpu": {"architecture": platform.machine(), "logical_processors": os.cpu_count()},
        "python": platform.python_version(),
        "packages": packages,
        "chrome": chrome_version(),
        "models_default_directory_only": {name: (ROOT / "models" / name).is_file() for name in MODEL_FILES},
        "manual_checks": {
            name: "unverified"
            for name in (
                "camera_permission",
                "microphone_permission",
                "running_server_profile",
                "running_server_jev_credential",
                "effective_model_directory",
                "device_model",
                "chrome_runtime_version",
            )
        },
        "limitations": [
            "Metadata existence is not runtime readiness.",
            "No environment variables, credentials, network or devices were inspected.",
        ],
    }


def results_template() -> dict:
    return {
        "schema_version": 1,
        "status": "pending_real",
        "run_label": None,
        "manual_environment": {"device_model": None, "chrome_version": None, "jev_response_model": None},
        "confirmations": {
            name: False
            for name in (
                "operator_consent",
                "participant_consent",
                "provider_terms",
                "cost_budget",
            )
        },
        "schema": {
            "anonymous_label_pattern": "^[A-Z0-9_-]{1,32}$",
            "device_model_allowed": ["M4", "other", "unverified"],
            "chrome_version_pattern": "^[0-9.]{1,64}$",
            "jev_response_model_pattern": "^jev-[0-9]+(?:\\.[0-9]+){1,3}$",
            "ground_truth_allowed": [
                "same_session",
                "new_session",
                "ended",
                "resume",
                "unassociated",
                "cleared",
                "history_bounded",
                "performance_gate",
            ],
            "statuses": ["pending_real", "not_run", "passed", "failed"],
            "metric_bounds": METRIC_BOUNDS,
            "rules": [
                "No names, face crops, raw audio, secrets or free-text notes.",
                "Labels are operator-defined anonymous run/session labels, never production IDs.",
                "Only an operator may assess real evidence; this tool never promotes a status.",
                "Mock passes do not qualify as real device/Jev passes.",
            ],
        },
        "cases": [
            {
                "id": name,
                "status": "pending_real",
                "participants": people,
                "session_label": None,
                "ground_truth": None,
                "expected": expected,
                "prerequisites": (
                    ["approved_jev_phase", "nonempty_reference_history_verified"]
                    if name in {"device.refresh_clears", "device.server_restart_clears"}
                    else []
                ),
                "scenario_seconds": duration,
                "metrics": dict.fromkeys(METRIC_BOUNDS),
                "operator_verified": False,
            }
            for name, people, expected, duration in CASES
        ],
    }


README = """# Real acceptance preparation — pending_real

此資料夾只有離線準備資料，所有案例尚未執行，沒有任何通過判定。
完整操作流程：docs/zh-TW/real-acceptance.md。

1. 人工核對 preflight.json：模型存在與 Chrome 安裝 metadata 不代表 runtime 可用。
2. 在 UI Import 匯入 profile.json。這是固定 PoC 驗收 profile，並非 recommended preset。
   re-ID / consent / crops 預設關閉；攝影機、麥克風及服務均未啟動。
3. 依 runbook 完成裝置權限、服務配置、匿名 A/B/C ground truth、參與者同意、
   Jev 服務條款及費用預算確認後，才手動啟用 Jev。請勿在此資料夾記錄任何 key。
4. 依 manual-results.json 記錄匿名 labels 與 numeric metrics；每例必須人工對照 ground truth。
   status 初始全部 pending_real；不得用 mock 結果代替真人驗收。
5. 30 分鐘測試：RTF p95 ≤0.5、audio dropped=0、uncaught errors=0、queue 不累積；
   warmup 後末 10 分鐘平均 RSS 減首 10 分鐘平均 RSS ≤30 MiB。

請勿記錄姓名、實際 session/conversation IDs、臉部圖像、原聲或自由文字個資。
工具拒絕覆寫既有目錄；新一輪請指定新的 --output-dir，保留原始驗收紀錄。
"""


def prepare(output_dir: Path) -> Path:
    """Validate/build first, then publish a new directory without replacing evidence."""
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("output directory already exists; choose a new run directory")
    payloads = {"profile.json": profile(), "preflight.json": preflight(), "manual-results.json": results_template()}
    encoded = {
        name: json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        for name, value in payloads.items()
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".real-preparation-", dir=output_dir.parent))
    claimed = False
    try:
        for name, content in encoded.items():
            (temporary / name).write_text(content, encoding="utf-8")
        (temporary / "README.md").write_text(README, encoding="utf-8")
        # mkdir is an atomic no-clobber claim; rename on POSIX could replace an empty existing directory.
        output_dir.mkdir()
        claimed = True
        for source in temporary.iterdir():
            source.rename(output_dir / source.name)
    except Exception:
        if claimed:
            shutil.rmtree(output_dir)
        raise
    finally:
        shutil.rmtree(temporary)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/acceptance/real-preparation"))
    args = parser.parse_args(argv)
    try:
        prepare(args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print("Offline preparation created. All real cases remain pending_real; no devices or Jev contacted.")
    return 0

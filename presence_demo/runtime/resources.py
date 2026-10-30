"""Process-wide shared resources: the vision pipeline (one detector shared by all connections, guarded by a
lock), the optional pose detector, the audio thread pool and distance calibration."""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time

from .. import diag
from ..body import calibration_factor
from ..optimizer import FrameProcessor
from ..tracker import PresenceTracker
from ..vision import VisionPipeline, decode_jpeg, load_pose_detector

TOKEN = os.environ.get("PRESENCE_TOKEN", "")  # empty = no auth (local use)
VISION_LOCK_TIMEOUT = float(os.environ.get("PRESENCE_VISION_LOCK_TIMEOUT", "10"))
_pipeline: VisionPipeline | None = None
_pose_cache: dict = {}
_PIPE_LOCK = threading.Lock()


def pose_detector():
    """Load the pose detector once per process (downloads MoveNet MultiPose Lightning on first use)."""
    if "det" not in _pose_cache:
        t = time.perf_counter()
        try:
            _pose_cache["det"] = load_pose_detector()
        except Exception as e:  # noqa: BLE001
            _pose_cache["det"] = None
            diag.event("pose_load_failed", "error", error=f"{type(e).__name__}: {e}")
        det = _pose_cache["det"]
        diag.event(
            "module_init",
            module="pose",
            name=det.name if det else "unavailable",
            ms=round((time.perf_counter() - t) * 1000, 1),
        )
    return _pose_cache["det"]


def calibrate(tr: PresenceTracker, meters) -> list[str]:
    """Set distance_calib so the closest visible person's raw distance maps to ``meters``."""
    try:
        known = float(meters)
    except (TypeError, ValueError):
        return ["calibration: invalid distance"]
    cands = [t for t in tr.tracks.values() if t.visible and t.dist_raw]
    if not cands or known <= 0:
        return ["calibration: no visible person with a distance estimate"]
    t = min(cands, key=lambda x: x.dist_raw)
    warns = tr.cfg.update({"distance_calib": calibration_factor(t.dist_raw, known)})
    for x in tr.tracks.values():
        x.distance = None  # restart EMA with the new factor
    return warns + [f"calibrated: factor {tr.cfg.distance_calib:.3f} (track #{t.id})"]


# One shared VisionPipeline (OpenCV DNN models) serves every WebSocket connection. cv2.FaceDetectorYN keeps its
# input size as mutable state and is not thread-safe: two connections detecting at once corrupted results (and
# could crash the process). All vision inference is therefore serialized through this lock.
VISION_LOCK = threading.Lock()
# Audio isolation gets its own single worker thread so it never waits behind vision jobs in the default pool.
AUDIO_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio")


class VisionBusy(RuntimeError):
    """The shared vision lock could not be acquired in time (another inference is hung or very slow)."""


def _vision_job(proc: FrameProcessor, data: bytes, cfg, stable: bool, conn: str | None = None):
    img, dec_ms = decode_jpeg(data)  # in memory only
    if img is None:
        return None
    t0 = time.perf_counter()
    if not VISION_LOCK.acquire(timeout=VISION_LOCK_TIMEOUT):
        diag.event("vision_lock_timeout", "error", conn, waited_s=VISION_LOCK_TIMEOUT)
        diag.log.error("stack dump (vision_lock_timeout)\n%s", diag.dump_stacks())
        raise VisionBusy(f"vision lock not acquired within {VISION_LOCK_TIMEOUT} s")
    try:
        wait_ms = (time.perf_counter() - t0) * 1000
        dets, tm = proc.process(img, cfg, stable)
    finally:
        VISION_LOCK.release()
    tm["lock_wait"] = wait_ms
    tm.update(decode=dec_ms, w=img.shape[1], h=img.shape[0])
    return dets, tm


def pipeline() -> VisionPipeline:
    global _pipeline
    with _PIPE_LOCK:  # two connections opening at once must not build two pipelines
        if _pipeline is None:
            t = time.perf_counter()
            _pipeline = VisionPipeline()
            diag.event(
                "module_init", module="vision", name=_pipeline.name, ms=round((time.perf_counter() - t) * 1000, 1)
            )
    return _pipeline

import cv2
import numpy as np
import pytest

from presencekit.optimizer import FrameProcessor, crop_box, map_raw_to_frame, moved, recommend_fps
from presencekit.tracker import Config


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class RectDetector:
    """Finds the white rectangle in the (possibly cropped/scaled) image; records input sizes."""

    name = "rect"

    def __init__(self):
        self.calls = []

    def detect(self, img):
        self.calls.append(img.shape[:2])
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pts = cv2.findNonZero((g > 200).astype(np.uint8))
        if pts is None:
            return []
        x, y, w, h = cv2.boundingRect(pts)
        row = np.zeros(15, np.float32)
        row[:4] = (x, y, w, h)
        row[4:14] = [x, y] * 5
        return [(row, 0.9)]


class CountEmbedder:
    name = "count"

    def __init__(self):
        self.calls = 0

    def embed(self, img, raw):
        self.calls += 1
        return np.ones(8, np.float32)


class Pipe:
    def __init__(self):
        self.detector, self.embedder = RectDetector(), CountEmbedder()


def frame(x=300, y=200, w=60, h=80, noise=0):
    img = np.full((480, 640, 3), 40, np.uint8)
    if w:
        img[y : y + h, x : x + w] = 255
    if noise:
        img[:50, :50] = noise
    return img


@pytest.fixture
def setup():
    clk = Clock()
    pipe = Pipe()
    return FrameProcessor(pipe, clock=clk), pipe, clk


def test_crop_box_and_mapping():
    assert crop_box((0.2, 0.1, 0.8, 0.9), 0.1, 640, 480) == (64, 0, 576, 480)
    raw = np.zeros(15, np.float32)
    raw[:4] = (10, 20, 30, 40)
    raw[4:6] = (15, 25)
    m = map_raw_to_frame(raw, 0.5, 64, 0)
    assert list(m[:4]) == [84, 40, 60, 80] and list(m[4:6]) == [94, 50]


def test_crop_downscale_boxes_map_back(setup):
    proc, pipe, clk = setup
    cfg = Config(roi_crop=True, max_width=200, motion_gate=False)
    dets, tm = proc.process(frame(300, 200, 60, 80), cfg)
    assert pipe.detector.calls[-1][1] <= 200  # detector saw a downscaled crop
    x0, y0, x1, y1 = dets[0].bbox
    assert abs(x0 * 640 - 300) < 4 and abs(y0 * 480 - 200) < 4
    assert abs(x1 * 640 - 360) < 4 and abs(y1 * 480 - 280) < 4
    cfg2 = Config(roi_crop=False, max_width=1920, motion_gate=False)
    dets2, _ = proc.process(frame(300, 200, 60, 80), cfg2)
    assert pipe.detector.calls[-1] == (480, 640)
    assert np.allclose(dets2[0].bbox, (300 / 640, 200 / 480, 360 / 640, 280 / 480), atol=1e-3)


def test_motion_gate_skips_static_and_forces_periodic(setup):
    proc, pipe, clk = setup
    cfg = Config(motion_gate=True, force_detect_seconds=3.0)
    proc.process(frame(), cfg)
    clk.t += 1
    dets, tm = proc.process(frame(), cfg)
    assert tm["skipped"] and len(pipe.detector.calls) == 1 and len(dets) == 1
    clk.t += 1
    proc.process(frame(), cfg, stable=False)  # unstable state -> always detect
    assert len(pipe.detector.calls) == 2
    clk.t += 3.5
    _, tm = proc.process(frame(), cfg)  # forced detection after force_detect_seconds
    assert not tm["skipped"] and len(pipe.detector.calls) == 3
    clk.t += 1
    _, tm = proc.process(frame(x=100), cfg)  # motion -> detect
    assert not tm["skipped"]
    assert proc.counts["skipped"] == 1


def test_motion_gate_off(setup):
    proc, pipe, clk = setup
    cfg = Config(motion_gate=False)
    for _ in range(3):
        clk.t += 1
        proc.process(frame(), cfg)
    assert len(pipe.detector.calls) == 3 and proc.counts["skipped"] == 0


def test_embedding_throttle(setup):
    proc, pipe, clk = setup
    cfg = Config(motion_gate=False, embed_throttle=True, embed_interval=3.0, embed_move_thresh=0.15)
    proc.process(frame(), cfg)
    clk.t += 1
    proc.process(frame(x=302), cfg)  # tiny move -> reuse
    clk.t += 1
    proc.process(frame(), cfg)
    assert pipe.embedder.calls == 1 and proc.counts["embed_reused"] == 2
    clk.t += 0.5
    proc.process(frame(x=330), cfg)  # moved half a face width -> recompute
    assert pipe.embedder.calls == 2
    clk.t += 3.1
    proc.process(frame(x=330), cfg)  # interval elapsed -> recompute
    assert pipe.embedder.calls == 3


def test_embedding_throttle_off(setup):
    proc, pipe, clk = setup
    cfg = Config(motion_gate=False, embed_throttle=False)
    for _ in range(3):
        clk.t += 1
        proc.process(frame(), cfg)
    assert pipe.embedder.calls == 3


def test_moved():
    a = (0.4, 0.4, 0.5, 0.5)
    assert not moved(a, (0.405, 0.4, 0.505, 0.5), 0.15)
    assert moved(a, (0.43, 0.4, 0.53, 0.5), 0.15)
    assert moved(a, (0.4, 0.4, 0.55, 0.55), 0.15)


def _snap(state, tracks=()):
    return {"state": state, "tracks": list(tracks)}


def test_adaptive_fps():
    cfg = Config(fps_low=0.5, fps_high=2.0)
    p = {"id": 1, "participant": True, "visible": True}
    assert recommend_fps(_snap("ACTIVE", [p]), cfg, False) == 0.5
    assert recommend_fps(_snap("IDLE", [p]), cfg, False) == 0.5
    assert recommend_fps(_snap("MAYBE_AWAY", [p]), cfg, False) == 2.0
    assert recommend_fps(_snap("ACTIVE", [p]), cfg, True) == 2.0  # tracks changing
    cand = {"id": 2, "participant": False, "visible": True}
    assert recommend_fps(_snap("ACTIVE", [p, cand]), cfg, False) == 2.0
    gone = {"id": 1, "participant": True, "visible": False}
    assert recommend_fps(_snap("IDLE", [gone]), cfg, False) == 2.0
    cfg.adaptive_fps = False
    assert recommend_fps(_snap("MAYBE_AWAY"), cfg, True) == 0.0


def test_bool_config_validation():
    c = Config()
    c.update({"motion_gate": "false", "roi_crop": 0, "adaptive_fps": "on"})
    assert c.motion_gate is False and c.roi_crop is False and c.adaptive_fps is True
    w = c.update({"embed_throttle": "maybe", "fps_low": 4, "fps_high": 1})
    assert c.embed_throttle is True and w and c.fps_low == 1


class FakePose:
    name = "fake-pose"

    def __init__(self):
        self.shapes = []

    def detect(self, img):
        self.shapes.append(img.shape[:2])
        h, w = img.shape[:2]
        kp = {
            "nose": (w / 2, h * 0.2, 0.9),
            "l_shoulder": (w * 0.6, h * 0.4, 0.9),
            "r_shoulder": (w * 0.4, h * 0.4, 0.9),
        }
        return [((w * 0.25, h * 0.1, w * 0.75, h * 0.9), 0.8, kp)]


def test_pose_runs_on_pose_roi_and_maps_back():
    clk = Clock()
    fake = FakePose()
    proc = FrameProcessor(Pipe(), clock=clk, pose_loader=lambda: fake)
    cfg = Config(motion_gate=False, pose_enabled=True, pose_roi=(0.5, 0.0, 1.0, 1.0), crop_margin=0.0)
    dets, tm = proc.process(frame(w=0), cfg)  # no face, one body
    assert fake.shapes[-1] == (480, 320) and proc.counts["pose_calls"] == 1 and "pose" in tm
    (d,) = dets
    assert d.face_bbox is None and d.body is not None and d.embedding is None
    x0, y0, x1, y1 = d.body.bbox
    assert abs(x0 - (0.5 + 0.25 * 0.5)) < 1e-3 and abs(x1 - (0.5 + 0.75 * 0.5)) < 1e-3
    assert abs(d.body.keypoints["nose"][0] - 0.75) < 1e-3
    cfg.pose_enabled = False
    proc.process(frame(), cfg)
    assert proc.counts["pose_calls"] == 1


class MatchedPose:
    """Deterministic full-frame pose whose head matches frame()'s default face."""

    name = "matched-pose"

    def __init__(self, empty=False, extra_body=False):
        self.calls = 0
        self.empty = empty
        self.extra_body = extra_body

    def detect(self, img):
        self.calls += 1
        if self.empty:
            return []
        bodies = [((240, 180, 420, 470), 0.9, {"nose": (330, 240, 0.9)})]
        if self.extra_body:
            bodies.append(((20, 20, 150, 470), 0.9, {"nose": (80, 80, 0.9)}))
        return bodies


def pose_setup(**kwargs):
    clk = Clock()
    fake = MatchedPose(**kwargs)
    pipe = Pipe()
    return FrameProcessor(pipe, clock=clk, pose_loader=lambda: fake), fake, clk


def test_pose_cadence_independent_of_face_motion_gate():
    proc, fake, clk = pose_setup()
    cfg = Config(pose_enabled=True, pose_interval=1.0, force_detect_seconds=10.0)
    proc.process(frame(), cfg)
    clk.t += 0.25
    dets, tm = proc.process(frame(), cfg)
    assert tm["skipped"] and tm["pose_reused"] and tm["pose"] == 0
    assert len(dets) == 1 and dets[0].body is not None
    clk.t += 0.75
    _, tm = proc.process(frame(), cfg)
    assert tm["skipped"] and not tm["pose_reused"]
    assert fake.calls == 2 and proc.counts["detect_calls"] == 1
    assert proc.counts["pose_reused"] == 1


@pytest.mark.parametrize("empty", [False, True])
def test_pose_cadence_with_full_face_detection_and_empty_pose_cache(empty):
    proc, fake, clk = pose_setup(empty=empty)
    cfg = Config(pose_enabled=True, motion_gate=False, pose_interval=1.0)
    proc.process(frame(), cfg)
    clk.t += 0.5
    dets, tm = proc.process(frame(), cfg)
    assert not tm["skipped"] and tm["pose_reused"] and fake.calls == 1
    assert (dets[0].body is None) == empty
    clk.t += 0.5
    proc.process(frame(), cfg)
    assert fake.calls == 2


def test_pose_unstable_and_zero_interval_run_every_frame():
    proc, fake, clk = pose_setup()
    cfg = Config(pose_enabled=True, motion_gate=True)
    for _ in range(3):
        proc.process(frame(), cfg, stable=False)
        clk.t += 0.1
    assert fake.calls == 3
    cfg.pose_interval = 0
    cfg.motion_gate = False
    for _ in range(3):
        proc.process(frame(), cfg)
        clk.t += 0.1
    assert fake.calls == 6  # no additional pose throttle on fully processed frames


def test_pose_face_disappearance_motion_and_new_face_force_update():
    proc, fake, clk = pose_setup()
    cfg = Config(pose_enabled=True, motion_gate=False)
    proc.process(frame(), cfg)
    for img in (frame(x=400), frame(w=0), frame(w=0), frame()):
        clk.t += 0.01
        proc.process(img, cfg)
    assert fake.calls == 5


def test_pose_body_only_never_reused_even_alongside_stable_face():
    proc, fake, clk = pose_setup(extra_body=True)
    cfg = Config(pose_enabled=True)
    first, _ = proc.process(frame(), cfg)
    assert len(first) == 2
    fake.extra_body = False
    clk.t += 0.1
    dets, tm = proc.process(frame(), cfg)
    assert not tm["pose_reused"] and fake.calls == 2
    assert len(dets) == 1  # departed unmatched body cannot extend presence
    clk.t += 0.1
    dets, tm = proc.process(frame(), cfg)
    assert tm["pose_reused"] and len(dets) == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("pose_roi", (0.1, 0.1, 0.9, 0.9)),
        ("face_roi", (0.1, 0.1, 0.9, 0.9)),
        ("pose_conf", 0.5),
        ("pose_interval", 1.0),
        ("crop_margin", 0.2),
        ("roi_crop", False),
    ],
)
def test_pose_configuration_invalidates_cache(key, value):
    proc, fake, clk = pose_setup()
    cfg = Config(pose_enabled=True)
    proc.process(frame(), cfg)
    clk.t += 0.1
    setattr(cfg, key, value)
    _, tm = proc.process(frame(), cfg)
    assert fake.calls == 2 and not tm["skipped"] and not tm["pose_reused"]


def test_pose_dimensions_reset_and_toggle_invalidate_cache():
    proc, fake, clk = pose_setup()
    cfg = Config(pose_enabled=True)
    proc.process(frame(), cfg)
    clk.t += 0.1
    proc.process(cv2.resize(frame(), (800, 600)), cfg)
    assert fake.calls == 2
    cfg.pose_enabled = False
    dets, _ = proc.process(frame(), cfg)
    assert all(d.body is None for d in dets)
    cfg.pose_enabled = True
    proc.process(frame(), cfg)
    assert fake.calls == 3
    proc.reset()
    proc.process(frame(), cfg)
    assert fake.calls == 4 and proc.counts["pose_calls"] == 1 and proc.counts["pose_reused"] == 0


def test_pose_face_identity_change_does_not_reuse_body():
    proc, fake, clk = pose_setup()
    cfg = Config(pose_enabled=True, motion_gate=False, embed_throttle=False)
    proc.process(frame(), cfg)
    proc.pipe.embedder.embed = lambda img, raw: -np.ones(8, np.float32)
    clk.t += 0.1
    _, tm = proc.process(frame(), cfg)
    assert not tm["pose_reused"] and fake.calls == 2


def test_pose_default_static_cadence_does_not_increase_work():
    proc, fake, clk = pose_setup()
    cfg = Config(pose_enabled=True)
    for _ in range(21):
        proc.process(frame(), cfg)
        clk.t += 0.25
    assert fake.calls == proc.counts["detect_calls"] == 3  # t=0,2,4 as before throttling


def test_pose_minor_background_motion_with_stationary_face_reuses_pose():
    proc, fake, clk = pose_setup()
    cfg = Config(pose_enabled=True, motion_gate=False)
    proc.process(frame(), cfg)
    clk.t += 0.25
    _, tm = proc.process(frame(noise=50), cfg)
    assert tm["diff"] < cfg.motion_threshold
    assert not tm["skipped"] and tm["pose_reused"] and fake.calls == 1


def test_pose_small_moves_accumulate_against_pose_anchor():
    proc, fake, clk = pose_setup()
    cfg = Config(pose_enabled=True, motion_gate=False, max_width=1920, roi_crop=False)
    proc.process(frame(), cfg)
    clk.t += 0.1
    _, tm = proc.process(frame(x=305), cfg)
    assert tm["pose_reused"]
    clk.t += 0.1
    _, tm = proc.process(frame(x=310), cfg)  # >15% of face width from pose anchor, not previous frame
    assert not tm["pose_reused"] and fake.calls == 2


def test_pose_new_face_forces_update_and_bodies_follow_face_order():
    proc, fake, clk = pose_setup()
    cfg = Config(pose_enabled=True, motion_gate=False, max_width=1920, roi_crop=False)
    initial, _ = proc.process(frame(), cfg)
    assert initial[0].body is not None
    original_detect = proc.pipe.detector.detect

    def two_faces(img):
        rows = original_detect(img)
        extra = rows[0][0].copy()
        extra[0] = 30
        return [*rows, (extra, 0.9)]

    proc.pipe.detector.detect = two_faces
    clk.t += 0.1
    dets, tm = proc.process(frame(), cfg)
    assert not tm["pose_reused"] and fake.calls == 2 and len(dets) == 2
    assert dets[0].body is not None and dets[1].body is None

    def reversed_faces(img):
        return list(reversed(two_faces(img)))

    proc.pipe.detector.detect = reversed_faces
    clk.t += 0.1
    dets, tm = proc.process(frame(), cfg)
    assert tm["pose_reused"] and fake.calls == 2
    assert dets[0].bbox[0] < dets[1].bbox[0]
    assert dets[0].body is None and dets[1].body is not None


@pytest.mark.parametrize("pose_enabled", [False, True])
def test_gated_frames_reuse_appearance_but_pose_refresh_recomputes(monkeypatch, pose_enabled):
    calls = []

    def describe(img, box):
        calls.append(box)
        return np.array([len(calls)], np.float32)

    monkeypatch.setattr("presencekit.optimizer.descriptor", describe)
    proc, fake, clk = pose_setup()
    cfg = Config(pose_enabled=pose_enabled, tracker_mode="kalman", force_detect_seconds=10.0)
    first, _ = proc.process(frame(), cfg)
    assert len(calls) == 1
    clk.t += 0.25
    dets, tm = proc.process(frame(), cfg)
    assert tm["skipped"] and len(calls) == 1
    assert dets[0].appearance is first[0].appearance
    clk.t += 2
    proc.process(frame(), cfg)
    assert len(calls) == (2 if pose_enabled else 1)
    cfg.crop_margin += 0.01
    proc.process(frame(), cfg)
    assert len(calls) == (3 if pose_enabled else 2)


def test_gated_appearance_matches_face_bbox_after_cache_reordering(monkeypatch):
    calls = []

    def describe(img, box):
        calls.append(box)
        return np.array([len(calls)], np.float32)

    monkeypatch.setattr("presencekit.optimizer.descriptor", describe)
    proc, fake, clk = pose_setup()
    cfg = Config(tracker_mode="kalman", roi_crop=False, max_width=1920)
    original_detect = proc.pipe.detector.detect

    def two_faces(img):
        rows = original_detect(img)
        extra = rows[0][0].copy()
        extra[0] = 30
        return [*rows, (extra, 0.9)]

    proc.pipe.detector.detect = two_faces
    initial, _ = proc.process(frame(), cfg)
    expected = {d.face_bbox: d.appearance for d in initial}
    proc.last_dets.reverse()
    clk.t += 0.1
    dets, tm = proc.process(frame(), cfg)
    assert tm["skipped"] and len(calls) == 2
    assert all(d.appearance is expected[d.face_bbox] for d in dets)


def test_switch_to_kalman_computes_appearance(monkeypatch):
    calls = []
    monkeypatch.setattr("presencekit.optimizer.descriptor", lambda img, box: calls.append(box))
    proc, fake, clk = pose_setup()
    cfg = Config(tracker_mode="legacy")
    proc.process(frame(), cfg)
    cfg.tracker_mode = "kalman"
    clk.t += 0.1
    proc.process(frame(), cfg)
    assert len(calls) == 1


def test_strong_motion_new_body_only_is_detected_immediately():
    proc, fake, clk = pose_setup()
    cfg = Config(pose_enabled=True)
    initial, _ = proc.process(frame(), cfg)
    assert len(initial) == 1
    fake.extra_body = True
    changed = frame()
    changed[np.all(changed == 40, axis=2)] = 100
    clk.t += 0.25
    dets, tm = proc.process(changed, cfg)
    assert tm["diff"] >= cfg.motion_threshold and not tm["skipped"]
    assert not tm["pose_reused"] and fake.calls == 2
    assert len(dets) == 2 and sum(d.face_bbox is None for d in dets) == 1


def test_reset_during_process_is_deferred_and_never_blocks(setup):
    import threading

    proc, pipe, clk = setup
    counts = proc.counts
    proc.process(frame(), Config(motion_gate=False))
    inside, release = threading.Event(), threading.Event()
    detect = pipe.detector.detect

    def slow_detect(img):
        inside.set()
        release.wait(5)
        return detect(img)

    pipe.detector.detect = slow_detect
    th = threading.Thread(target=proc.process, args=(frame(), Config(motion_gate=False)))
    th.start()
    assert inside.wait(5)
    proc.reset()  # from another thread (the event loop) while the worker is mid-frame: must not wait for it
    release.set()
    th.join(5)
    # the in-flight frame did not refill the caches after the reset, and stats still see the same counts dict
    assert proc.last_faces == [] and proc.cache == [] and proc.last_thumb is None
    assert proc.counts is counts and proc.counts["frames"] == 0

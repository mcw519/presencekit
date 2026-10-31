"""Layer 4 tracking: Kalman filter, Hungarian assignment, appearance descriptor, kalman tracker mode."""

import itertools

import numpy as np
import pytest

from presencekit.appearance import descriptor, torso_box
from presencekit.body import Body
from presencekit.kalman import KalmanBox, appearance_similarity, hungarian
from presencekit.tracker import Config, Detection, PresenceTracker


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def box(cx, cy=0.4, w=0.08, h=0.11):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ---------------------------------------------------------------- Kalman
def test_kalman_learns_velocity_and_predicts_with_variable_dt():
    kf = KalmanBox(box(0.3), 0.0)
    t, x = 0.0, 0.3
    for dt in (0.5, 1.0, 2.0, 0.5, 0.5, 1.0, 2.0, 0.5):  # 0.5-2 fps
        t += dt
        x += 0.05 * dt  # 0.05 / s
        kf.update(box(x), t)
    assert kf.velocity[0] == pytest.approx(0.05, abs=0.01)
    kf.predict_to(t + 2.0)
    cx = (kf.bbox()[0] + kf.bbox()[2]) / 2
    assert cx == pytest.approx(x + 0.1, abs=0.02)


def test_kalman_uncertainty_grows_when_coasting_and_gates():
    kf = KalmanBox(box(0.5), 0.0)
    for i in range(1, 5):
        kf.update(box(0.5), float(i))
    near, far = kf.mahalanobis(box(0.51)), kf.mahalanobis(box(0.8))
    assert near < 9.21 < far
    p0 = kf.P[0, 0]
    kf.predict_to(7.0)
    assert kf.P[0, 0] > p0


# ---------------------------------------------------------------- Hungarian
def brute(C):
    n, m = C.shape
    best = None
    if n <= m:
        for perm in itertools.permutations(range(m), n):
            v = sum(C[i, perm[i]] for i in range(n))
            best = v if best is None or v < best else best
    else:
        return brute(C.T)
    return best


@pytest.mark.parametrize("shape", [(3, 3), (2, 4), (4, 2), (5, 5), (1, 3)])
def test_hungarian_matches_brute_force(shape):
    rng = np.random.default_rng(sum(shape))
    for _ in range(20):
        C = rng.random(shape)
        pairs = hungarian(C)
        assert len(pairs) == min(shape)
        assert len({r for r, _ in pairs}) == len(pairs) == len({c for _, c in pairs})
        assert sum(C[r, c] for r, c in pairs) == pytest.approx(brute(C))


def test_hungarian_beats_greedy():
    C = np.array([[0.1, 0.2], [0.15, 0.9]])  # greedy takes (0,0) then (1,1)=1.0; optimum 0.35
    assert hungarian(C) == [(0, 1), (1, 0)]


# ---------------------------------------------------------------- appearance
def test_appearance_descriptor_in_memory_and_similarity():
    img = np.zeros((240, 320, 3), np.uint8)
    img[:, :160] = (0, 0, 200)  # red shirt (BGR)
    img[:, 160:] = (200, 0, 0)  # blue shirt
    red = descriptor(img, (0.1, 0.3, 0.4, 0.8))
    red2 = descriptor(img, (0.05, 0.4, 0.45, 0.9))
    blue = descriptor(img, (0.6, 0.3, 0.9, 0.8))
    assert red.shape == (48,) and red.sum() == pytest.approx(1.0)
    assert appearance_similarity(red, red2) > 0.95 > 0.1 > appearance_similarity(red, blue)
    assert descriptor(img, (0.5, 0.5, 0.505, 0.505)) is None  # too small
    assert appearance_similarity(None, red) is None


def test_torso_box_from_keypoints_body_and_face():
    b = Body((0.3, 0.1, 0.7, 0.95), 0.9, {"l_shoulder": (0.6, 0.3, 0.9), "r_shoulder": (0.4, 0.3, 0.9)})
    b.keypoints.update(l_hip=(0.58, 0.6, 0.8), r_hip=(0.42, 0.6, 0.8))
    x0, y0, x1, y1 = torso_box(None, b)
    assert 0.4 < x0 < x1 < 0.6 and y0 == pytest.approx(0.3) and y1 == pytest.approx(0.6)
    fx0, fy0, fx1, fy1 = torso_box((0.45, 0.2, 0.55, 0.3), None)
    assert fy0 > 0.3 and fx0 < 0.45 and fx1 > 0.55
    assert torso_box(None, None) is None


# ---------------------------------------------------------------- tracker (kalman mode)
def emb(seed):
    v = np.random.default_rng(seed).normal(size=64)
    return v / np.linalg.norm(v)


def app(seed):
    return np.random.default_rng(100 + seed).dirichlet(np.full(48, 0.3))


def body_at(cx, face=True):
    kp = {"l_shoulder": (cx + 0.08, 0.5, 0.9), "r_shoulder": (cx - 0.08, 0.5, 0.9)}
    if not face:
        kp = {"l_shoulder": (cx - 0.01, 0.5, 0.9), "r_shoulder": (cx + 0.01, 0.5, 0.9)}
    return Body((cx - 0.13, 0.33, cx + 0.13, 0.95), 0.8, kp)


def det(pid, cx, face=True):
    b = box(cx)
    if face:
        return Detection(b, 0.9, emb(pid), face_bbox=b, body=body_at(cx), appearance=app(pid))
    return Detection(b, 0.75, None, face_bbox=None, body=body_at(cx, False), appearance=app(pid))


def mk(**kw):
    clk = Clock()
    return PresenceTracker(Config(tracker_mode="kalman", pose_enabled=True, **kw), clock=clk), clk


def ids(tr):
    return {round((t.bbox[0] + t.bbox[2]) / 2, 2): t.id for t in tr.tracks.values() if t.visible_kind == "detected"}


def test_coasting_predicted_track_counts_as_present_then_fades():
    tr, clk = mk()
    for _ in range(4):
        clk.t += 0.5
        tr.update([det(1, 0.5)])
    assert tr.present
    clk.t += 0.5
    s = tr.update([])
    t = s["tracks"][0]
    assert t["visible_kind"] == "predicted" and t["vis_conf"] == pytest.approx(0.75) and t["visible"]
    assert tr.present and not tr.empty
    for _ in range(3):
        clk.t += 0.5
        s = tr.update([])
    t = s["tracks"][0]
    assert t["visible_kind"] == "predicted" and not t["visible"]  # vis_conf 0.0..0.25 < 0.4
    for _ in range(3):
        clk.t += 0.5
        s = tr.update([])
    assert s["tracks"][0]["visible_kind"] == "none" and tr.empty
    # empty clock starts when coasting ended, not at the first missing frame
    assert tr.empty_since == pytest.approx(3.5)


def test_id_persists_through_occlusion():
    tr, clk = mk()
    x = 0.4
    for _ in range(6):
        clk.t += 0.5
        x += 0.01
        tr.update([det(1, x)])
    tid = next(iter(tr.tracks))
    for _ in range(4):  # 2 s occlusion while still walking
        clk.t += 0.5
        x += 0.01
        tr.update([])
    clk.t += 0.5
    x += 0.01
    tr.update([det(1, x)])
    assert list(tr.tracks) == [tid] and tr.tracks[tid].visible_kind == "detected"
    assert tr.tracks[tid].kf.velocity[0] == pytest.approx(0.02, abs=0.01)


def test_id_persists_through_side_turn_without_face():
    tr, clk = mk()
    for _ in range(4):
        clk.t += 1
        tr.update([det(1, 0.45), det(2, 0.7)])
    before = ids(tr)
    for k in range(6):  # person 1 turns sideways and drifts; no face, body + appearance only
        clk.t += 1
        tr.update([det(1, 0.45 + 0.01 * k, face=False), det(2, 0.7)])
    clk.t += 1
    tr.update([det(1, 0.51), det(2, 0.7)])
    after = ids(tr)
    assert len(tr.tracks) == 2 and before[0.45] == after[0.51] and before[0.7] == after[0.7]


def test_swap_places_without_faces_keeps_ids_by_appearance():
    def swap(mode):
        clk = Clock()
        tr = PresenceTracker(Config(tracker_mode=mode, pose_enabled=True), clock=clk)
        for _ in range(4):
            clk.t += 0.5
            tr.update([det(1, 0.4), det(2, 0.62)])
        id1 = ids(tr)[0.4]
        for k in range(1, 7):  # swap over 3 s, both side-on
            clk.t += 0.5
            a, b = 0.4 + 0.22 * k / 6, 0.62 - 0.22 * k / 6
            tr.update([det(1, a, face=False), det(2, b, face=False)])
        clk.t += 0.5
        tr.update([det(1, 0.62), det(2, 0.4)])
        return ids(tr).get(0.62) == id1

    assert swap("kalman")


def test_appearance_template_is_ema():
    tr, clk = mk(appearance_ema=0.5)
    clk.t += 1
    tr.update([det(1, 0.5)])
    a0 = tr.tracks[1].appearance.copy()
    clk.t += 1
    d = det(1, 0.5)
    d.appearance = app(7)
    tr.update([d])
    exp = 0.5 * a0 + 0.5 * app(7)
    assert np.allclose(tr.tracks[1].appearance, exp / exp.sum())


def test_snapshot_tracking_fields_and_legacy_default():
    assert Config().tracker_mode == "legacy"
    tr, clk = mk()
    for _ in range(3):
        clk.t += 1
        s = tr.update([det(1, 0.5)])
    t = s["tracks"][0]
    assert t["visible_kind"] == "detected" and t["vis_conf"] == 1.0
    assert t["dwell_seconds"] == pytest.approx(2.0) and len(t["velocity"]) == 2
    clk2 = Clock()
    lg = PresenceTracker(Config(), clock=clk2)
    clk2.t += 1
    lg.update([det(1, 0.5)])
    clk2.t += 1
    s = lg.update([])
    assert s["tracks"][0]["visible_kind"] == "none" and s["tracks"][0]["velocity"] is None

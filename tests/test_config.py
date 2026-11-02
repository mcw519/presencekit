from dataclasses import fields

from presencekit.tracker import RANGES, Config


def test_ranges_cover_all_fields():
    # npu_delegate_path is server-side only (env PRESENCE_NPU_DELEGATE): never settable over WebSocket
    assert {f.name for f in fields(Config)} - {"npu_delegate_path"} == set(RANGES)
    c = Config()
    assert c.update({"npu_delegate_path": "/tmp/evil.so"}) and c.npu_delegate_path != "/tmp/evil.so"


def test_clamp_and_types():
    c = Config()
    w = c.update({"min_conf": 5, "present_frames": 3.7, "absent_frames": -2, "empty_timeout": 0})
    assert c.min_conf == 1.0 and c.present_frames == 4 and c.absent_frames == 1 and c.empty_timeout == 1.0
    assert isinstance(c.present_frames, int)
    assert any("min_conf" in x for x in w)


def test_invalid_and_unknown_ignored():
    c = Config()
    w = c.update({"cos_match": "abc", "foo": 1, "base_timeout": float("nan")})
    assert c.cos_match == Config().cos_match and c.base_timeout == Config().base_timeout
    assert len(w) == 3


def test_roi_sorted_clamped_and_min_size():
    c = Config()
    c.update({"roi": [0.9, 1.5, 0.1, -0.2]})
    assert c.roi == (0.1, 0.0, 0.9, 1.0)
    w = c.update({"roi": [0.5, 0.5, 0.52, 0.9]})
    assert c.roi == (0.1, 0.0, 0.9, 1.0) and w
    c.update({"roi": [0.1, 0.2]})
    assert c.roi == (0.1, 0.0, 0.9, 1.0)


def test_cos_weak_not_above_cos_match():
    c = Config()
    c.update({"cos_match": 0.3, "cos_weak": 0.8})
    assert c.cos_weak == 0.3


def test_pose_interval_validation():
    c = Config()
    assert c.pose_interval == 2.0
    assert c.update({"pose_interval": 4}) and c.pose_interval == 2.0
    assert c.update({"pose_interval": -1}) and c.pose_interval == 0.0
    assert c.update({"pose_interval": float("nan")}) and c.pose_interval == 0.0

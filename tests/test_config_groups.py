"""v0.13 grouped Config view: complete, disjoint, and a pure view over the flat keys (which stay authoritative)."""

from __future__ import annotations

from dataclasses import asdict, fields

import pytest

from presencekit.config import GROUPS, RANGES, Config, ConfigGroup, group_of


def test_every_field_is_in_exactly_one_group():
    grouped = [k for keys in GROUPS.values() for k in keys]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {f.name for f in fields(Config)}
    assert set(RANGES) <= set(grouped)


def test_group_view_reads_and_writes_the_flat_fields():
    c = Config()
    g = c.group("reid")
    assert isinstance(g, ConfigGroup) and g.reid_threshold == c.reid_threshold
    g.reid_threshold = 0.8
    assert c.reid_threshold == 0.8
    with pytest.raises(AttributeError):
        _ = g.min_conf
    with pytest.raises(AttributeError):
        g.min_conf = 0.1
    with pytest.raises(KeyError):
        c.group("server")


def test_group_update_uses_the_flat_validation():
    c = Config()
    w = c.group("core").update({"min_conf": 5, "reid_threshold": 0.7})
    assert c.min_conf == 1.0 and c.reid_threshold == Config().reid_threshold
    assert w == ["unknown key reid_threshold", "min_conf clamped to 1.0"]


def test_flat_access_and_serialisation_are_unchanged():
    c = Config()
    flat = asdict(c)
    assert "core" not in flat and len(flat) == len(fields(Config))
    nested = c.as_groups()
    assert {k: v for d in nested.values() for k, v in d.items()} == flat
    assert group_of("roi") == "core" and group_of("asr_feed") == "asr" and group_of("nope") is None
    assert c.update({"roi": (0.1, 0.1, 0.9, 0.9)}) == [] and c.group("core").face_roi == (0.1, 0.1, 0.9, 0.9)

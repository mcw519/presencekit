"""Wire protocol checks are independent of camera devices and inference models."""

import base64
import struct

import pytest

from presencekit.web.protocol import binary_frame, legacy_frame

JPEG = b"\xff\xd8jpeg\xff\xd9"


def test_video_protocols_normalize_identically():
    expected = {"data": JPEG, "t": 125.5, "speech": True}
    assert binary_frame(struct.pack("<BdB", 4, 125.5, 1) + JPEG) == expected
    assert legacy_frame({"data": base64.b64encode(JPEG).decode(), "t": 125.5, "speech": True}) == expected
    assert legacy_frame({"data": "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()})["t"] is None


@pytest.mark.parametrize(
    "packet",
    [
        b"",
        b"\x04",
        b"\x04" * 9,
        struct.pack("<BdB", 4, float("nan"), 0) + JPEG,
        struct.pack("<BdB", 4, float("inf"), 0) + JPEG,
        struct.pack("<BdB", 4, -1, 0) + JPEG,
        struct.pack("<BdB", 4, 1, 2) + JPEG,
        struct.pack("<BdB", 1, 1, 0) + JPEG,
        struct.pack("<BdB", 4, 1, 0) + b"not jpeg",
    ],
)
def test_reject_invalid_binary_video(packet):
    with pytest.raises(ValueError):
        binary_frame(packet)


@pytest.mark.parametrize(
    "message",
    [
        {},
        {"data": 4},
        {"data": "%%%"},
        {"data": base64.b64encode(JPEG).decode(), "t": True},
        {"data": base64.b64encode(JPEG).decode(), "t": float("nan")},
    ],
)
def test_reject_invalid_legacy_video(message):
    with pytest.raises(ValueError):
        legacy_frame(message)

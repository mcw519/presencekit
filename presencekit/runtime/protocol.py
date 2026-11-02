"""Client <-> server message protocol (transport-neutral).

Binary packets (first byte = tag): 0x01 audio PCM16 in, 0x02 clean audio out, 0x03 recording tracks out,
0x04 video (JPEG). In-memory video framing only; 0x01/02/03 remain reserved for audio paths.
Text packets are JSON objects with a ``type`` from :data:`CLIENT_MESSAGE_TYPES`.
"""

from __future__ import annotations

import base64
import binascii
import math
import struct
from typing import Literal

AUDIO_IN_TAG, AUDIO_OUT_TAG, RECORDING_TAG, VIDEO_TAG = 1, 2, 3, 4
VIDEO_HEADER = struct.Struct("<BdB")  # tag, client performance.now() milliseconds, speech

# JSON message types a client may send (v0.12 wire protocol, unchanged in v0.13):
#   frame      legacy JSON video frame {data: base64 JPEG, t?, speech?}
#   ping       {t, rtt?} -> pong
#   vad        browser VAD {speech}
#   config     {config: {key: value}} -> state with config_ack + warnings
#   calibrate  {meters} distance calibration on the closest visible person
#   client     browser-side report {what, ...} (visibility / auto-pause / media ended / reconnect)
#   reid_wipe  wipe this page's re-ID references
#   record     {on} client-side recording of aligned audio tracks (the server stores nothing)
#   start / stop / reset   session control
ClientMessageType = Literal[
    "frame", "ping", "vad", "config", "calibrate", "client", "reid_wipe", "record", "start", "stop", "reset"
]
CLIENT_MESSAGE_TYPES: tuple[str, ...] = (
    "frame",
    "ping",
    "vad",
    "config",
    "calibrate",
    "client",
    "reid_wipe",
    "record",
    "start",
    "stop",
    "reset",
)


def _jpeg(data: bytes) -> bytes:
    if len(data) < 4 or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError("invalid JPEG envelope")
    return data


def _timestamp(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid video timestamp")
    return value


def binary_frame(packet: bytes) -> dict:
    """Validate before admitting video into the latest-frame slot."""
    if len(packet) < VIDEO_HEADER.size + 4:
        raise ValueError("short video packet")
    tag, timestamp, speech = VIDEO_HEADER.unpack_from(packet)
    if tag != VIDEO_TAG or speech not in (0, 1):
        raise ValueError("invalid video header")
    return {"data": _jpeg(packet[VIDEO_HEADER.size :]), "t": _timestamp(timestamp), "speech": bool(speech)}


def legacy_frame(message: dict) -> dict:
    """Keep legacy JSON clients compatible, with the same worker representation."""
    data = message.get("data")
    if not isinstance(data, str):
        raise ValueError("missing JPEG")
    try:
        jpeg = base64.b64decode(data.split(",")[-1], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64 JPEG") from exc
    timestamp = message.get("t")
    if timestamp is not None:
        timestamp = _timestamp(timestamp)
    return {"data": _jpeg(jpeg), "t": timestamp, "speech": bool(message.get("speech"))}

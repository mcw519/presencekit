"""v0.12 location of the message protocol; it now lives in :mod:`presence_demo.runtime.protocol`."""

from ..runtime.protocol import (  # noqa: F401  (re-exported)
    AUDIO_IN_TAG,
    AUDIO_OUT_TAG,
    CLIENT_MESSAGE_TYPES,
    RECORDING_TAG,
    VIDEO_HEADER,
    VIDEO_TAG,
    ClientMessageType,
    _jpeg,
    _timestamp,
    binary_frame,
    legacy_frame,
)

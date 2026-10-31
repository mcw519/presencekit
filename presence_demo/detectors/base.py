"""Detector protocols of the v0.13 detectors layer (formalized from v0.12 ``presence_demo.vision``)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FaceDetector(Protocol):
    """Faces in pixel coordinates: ``[(raw_row, conf)]`` where ``raw_row`` is the YuNet layout
    (x, y, w, h, 5 landmark pairs, score). Also exported as ``presence_demo.vision.Detector``.

    Optional attributes read with ``getattr``: ``landmarks`` (True when the rows carry the 5 landmarks an SFace
    embedder aligns on) and ``accelerator`` (shown in diagnostics)."""

    name: str

    def detect(self, img: np.ndarray) -> list[tuple[np.ndarray, float]]: ...  # (raw row, conf)


class Embedder(Protocol):
    """Identity embedding of the face described by ``raw`` (a :class:`FaceDetector` row)."""

    name: str

    def embed(self, img: np.ndarray, raw: np.ndarray) -> np.ndarray: ...


class PoseDetector(Protocol):
    """Returns bodies in pixel coordinates of the given image: (bbox_xyxy, conf, {name: (x, y, c)})."""

    name: str

    def detect(self, img: np.ndarray) -> list[tuple[tuple, float, dict]]: ...

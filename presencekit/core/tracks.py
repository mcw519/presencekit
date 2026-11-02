"""Per-person values of the core layer: :class:`Detection` (one person in a frame) and :class:`Track`."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .. import presence_model as pm
from ..body import Body
from ..kalman import KalmanBox


@dataclass
class Detection:
    """One person in a frame. ``bbox`` is the track box (face box, or head box estimated from the body)."""

    bbox: tuple  # normalized x0,y0,x1,y1
    conf: float
    embedding: np.ndarray | None = None
    face_bbox: tuple | None = None
    body: Body | None = None
    appearance: np.ndarray | None = None  # torso HSV histogram (kalman mode)
    frontal: float | None = None  # frontal-face score from YuNet landmarks (1 = facing the camera)

    def __post_init__(self):
        if self.face_bbox is None and self.body is None:
            self.face_bbox = self.bbox  # plain face detection (legacy constructor)


@dataclass
class Track:
    id: int
    bbox: tuple
    embedding: np.ndarray | None
    conf: float
    first_seen: float
    last_seen: float
    in_roi_since: float | None
    participant: bool = False
    visible: bool = True
    similarity: float = 0.0  # similarity of last match
    has_face: bool = True
    body: Body | None = None
    facing: str | None = None  # facing | side | away (pose only)
    facing_score: float = 0.0
    distance: float | None = None  # meters (EMA, calibrated)
    dist_raw: float | None = None  # meters with calib = 1 (for calibration)
    trend: str = "steady"  # approaching | receding | steady
    far: bool = False  # beyond leave_distance
    size_hist: deque = field(default_factory=lambda: deque(maxlen=12))
    visible_kind: str = "detected"  # detected | predicted | none
    vis_conf: float = 1.0  # 1 when detected; decays while coasting (kalman)
    appearance: np.ndarray | None = None
    kf: KalmanBox | None = None
    logodds: float = 0.0  # hmm: log P(present)/P(left)
    frontal: float | None = None
    intent: float | None = None  # score: interaction intent in [0, 1] (EMA)
    intent_cues: dict = field(default_factory=dict)
    intent_t: float | None = None
    intent_since: float | None = None  # since when intent >= join_threshold
    low_intent_since: float | None = None  # participant: since when intent < drop_threshold
    vx: float = 0.0  # lateral speed (EMA, frame widths / s)
    size_rate: float = 0.0  # relative size change per second (EMA, + = approaching)
    _prev: tuple | None = None  # (t, cx, size) of the last detection

    @property
    def p(self) -> float:
        return pm.sigmoid(self.logodds)


def asdict_shallow(t: Track) -> dict:
    return {f: getattr(t, f) for f in t.__dataclass_fields__}

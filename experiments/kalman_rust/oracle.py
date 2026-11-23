"""Primitive replay interface used by rust-anything and the experimental tracker."""

from types import SimpleNamespace

import numpy as np

from presencekit.body import Body
from presencekit.core.tracking import TrackingMixin
from presencekit.kalman import MEAS_STD, KalmanBox

# One filter row: x[8], four position/velocity covariance blocks[4 x 4], timestamp.
_AXIS = np.arange(4)
_ROWS = np.stack((_AXIS, _AXIS, _AXIS + 4, _AXIS + 4), axis=1)
_COLS = np.stack((_AXIS, _AXIS + 4, _AXIS, _AXIS + 4), axis=1)
_GATE_KEYS = ("gate_mahalanobis", "iou_match", "match_cos", "cos_weak", "appearance_min_sim")


def _pack_state(kf):
    return [*kf.x.tolist(), *kf.P[_ROWS, _COLS].ravel().tolist(), float(kf.t)]


def _unpack_state(row):
    filt = KalmanBox.__new__(KalmanBox)
    filt.x = np.array(row[:8])
    filt.P = np.zeros((8, 8))
    filt.P[_ROWS, _COLS] = np.asarray(row[8:24]).reshape(4, 4)
    filt.R = np.diag(np.square(MEAS_STD))
    filt.t = row[24]
    return filt


def _pack_observation(obs):
    return [
        list(obs.bbox),
        list(obs.body.bbox) if obs.body is not None else [],
        obs.embedding.tolist() if obs.embedding is not None else [],
        obs.appearance.ravel().tolist() if obs.appearance is not None else [],
    ]


def _unpack_observation(row):
    bbox, body, embedding, appearance = row
    return SimpleNamespace(
        bbox=tuple(bbox),
        body=Body(tuple(body), 1.0) if body else None,
        embedding=np.array(embedding, dtype=np.float32) if embedding else None,
        appearance=np.array(appearance) if appearance else None,
    )


def associate(
    now: float,
    gates: list[float],
    states: list[list[float]],
    tracks: list[list[list[float]]],
    detections: list[list[list[float]]],
) -> list[list[float]]:
    """Run Python association on boxed inputs, then return filter rows and matches."""
    if len(gates) != 5 or len(states) != len(tracks) or any(len(row) != 25 for row in states):
        raise ValueError("invalid Kalman state or gates")
    if any(len(row) != 4 or len(row[0]) != 4 or len(row[1]) not in (0, 4) for row in tracks + detections):
        raise ValueError("invalid observation")
    cfg = SimpleNamespace(**dict(zip(_GATE_KEYS, gates, strict=True)))
    kernel = TrackingMixin()
    kernel.cfg = cfg
    live = [_unpack_observation(row) for row in tracks]
    for track, row in zip(live, states, strict=True):
        track.kf = _unpack_state(row)
    pred, matches = kernel._associate_kalman([_unpack_observation(row) for row in detections], live, now)
    return [
        *([*_pack_state(track.kf), *box] for track, box in zip(live, pred, strict=True)),
        *([float(i), float(j), cs] for i, j, cs in matches),
    ]

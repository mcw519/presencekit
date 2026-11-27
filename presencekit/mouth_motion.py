"""YuNet-aligned local mouth motion from consecutive camera frames."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

FACE_SIZE = 128
FRESH_S = 0.085
MAX_HEAD_SHIFT = 0.04


def align_face(frame, row):
    eyes = sorted(row[4:8].reshape(2, 2), key=lambda p: p[0])
    left, right = eyes
    axis = right - left
    if np.linalg.norm(axis) < 12:
        return None
    third = left + np.array([-axis[1], axis[0]])
    source = np.float32([left, right, third])
    target = np.float32([[40, 39], [88, 39], [40, 87]])
    transform = cv2.getAffineTransform(source, target)
    patch = cv2.warpAffine(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), transform, (FACE_SIZE, FACE_SIZE))
    corners = cv2.transform(row[10:14].reshape(1, 2, 2), transform)[0]
    return patch, corners


def mouth_motion(previous, current):
    prev_patch, prev_corners = previous
    patch, corners = current
    midpoint = (corners.mean(axis=0) + prev_corners.mean(axis=0)) / 2
    width = (np.linalg.norm(corners[0] - corners[1]) + np.linalg.norm(prev_corners[0] - prev_corners[1])) / 2
    if width < 12 or midpoint[1] < 62:
        return None
    yy, xx = np.ogrid[:FACE_SIZE, :FACE_SIZE]
    x, y = abs(xx - midpoint[0]) / width, abs(yy - midpoint[1]) / width
    lip = (x <= 0.49) & (y <= 0.26)
    cheeks = (x >= 0.54) & (x <= 0.88) & (y <= 0.26)
    upper = lip & (yy < midpoint[1])
    lower = lip & (yy >= midpoint[1])
    if not upper.any() or not lower.any() or not cheeks.any():
        return None
    flow = cv2.calcOpticalFlowFarneback(prev_patch, patch, None, 0.5, 2, 13, 2, 5, 1.1, 0)
    shift = np.median(flow[cheeks], axis=0)
    if float(np.linalg.norm(shift)) / width > MAX_HEAD_SHIFT:
        return 0.0
    rel = flow - shift
    motion = np.linalg.norm(rel, axis=2)
    # Compare local motion against cheeks to suppress common landmark jitter/head movement,
    # and require vertical upper-vs-lower lip divergence so silent head turns/smiles do not fire.
    lip_motion = np.percentile(motion[lip], 80)
    cheek_motion = np.percentile(motion[cheeks], 80)
    orig = max(0.0, float(lip_motion - cheek_motion) / width)
    div_vy = abs(float(np.median(rel[lower, 1]) - np.median(rel[upper, 1]))) / width
    return min(orig, div_vy)


MOUTH_THRESHOLD = 0.004587638191878796


@dataclass
class MotionSample:
    bbox: tuple
    patch: tuple
    scores: tuple[float, ...] = ()


def _box_stable(old_box: tuple, new_box: tuple) -> bool:
    ow = old_box[2] - old_box[0]
    nw = new_box[2] - new_box[0]
    if ow <= 0 or nw <= 0:
        return False
    scale = nw / ow
    if not (0.75 <= scale <= 1.33):
        return False
    dx = ((new_box[0] + new_box[2]) - (old_box[0] + old_box[2])) / 2
    dy = ((new_box[1] + new_box[3]) - (old_box[1] + old_box[3])) / 2
    return math.hypot(dx, dy) <= ow * 0.25


class MouthMotion:
    """Per-stream short history. Missing, ambiguous or sparse frames break the sequence."""

    def __init__(self):
        self.previous: list[MotionSample] = []
        self.time: float | None = None
        self.interval_ms: float | None = None

    def step(self, samples, frame_time, overlap):
        gap = None if self.time is None or frame_time is None else frame_time - self.time
        self.interval_ms = gap * 1000 if gap is not None and gap > 0 else None
        previous = self.previous if gap is not None and 0 < gap <= FRESH_S else []
        matches = [[old for old in previous if overlap(new.bbox, old.bbox) > 0.5] for new in samples]
        values = []
        for new, candidates in zip(samples, matches, strict=True):
            if len(candidates) == 1:
                old = candidates[0]
                if sum(any(candidate is old for candidate in group) for group in matches) == 1 and _box_stable(
                    old.bbox, new.bbox
                ):
                    score = mouth_motion(old.patch, new.patch)
                    if score is not None:
                        new.scores = (*old.scores[-2:], score)
            values.append(float(np.median(new.scores)) if len(new.scores) == 3 else None)
        self.previous, self.time = samples, frame_time
        return values

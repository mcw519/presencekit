"""YuNet-aligned local mouth motion from consecutive camera frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FACE_SIZE = 128
FRESH_S = 0.085


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
    if not lip.any() or not cheeks.any():
        return None
    flow = cv2.calcOpticalFlowFarneback(prev_patch, patch, None, 0.5, 2, 13, 2, 5, 1.1, 0)
    shift = np.median(flow[cheeks], axis=0)
    motion = np.linalg.norm(flow - shift, axis=2)
    # Compare local motion against cheeks to suppress common landmark jitter/head movement.
    lip_motion = np.percentile(motion[lip], 80)
    cheek_motion = np.percentile(motion[cheeks], 80)
    return float(max(0, lip_motion - cheek_motion) / width)


MOUTH_THRESHOLD = 0.004587638191878796


@dataclass
class MotionSample:
    bbox: tuple
    patch: tuple
    scores: tuple[float, ...] = ()


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
                if sum(any(candidate is old for candidate in group) for group in matches) == 1:
                    score = mouth_motion(old.patch, new.patch)
                    if score is not None:
                        new.scores = (*old.scores[-2:], score)
            values.append(float(np.median(new.scores)) if len(new.scores) == 3 else None)
        self.previous, self.time = samples, frame_time
        return values

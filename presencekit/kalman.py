"""Constant-velocity Kalman filter for boxes, Hungarian assignment and appearance similarity.

Pure numpy (no OpenCV, no SciPy). Boxes are normalized (x0, y0, x1, y1); the filter state is
(cx, cy, w, h, vcx, vcy, vw, vh) with velocities in normalized units per second.
"""

from __future__ import annotations

import numpy as np

# process noise: acceleration std (normalized units / s^2) for center and size
ACC_CENTER = 0.5
ACC_SIZE = 0.1
MEAS_STD = (0.01, 0.01, 0.015, 0.015)


def box_to_z(b) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], dtype=float)


def z_to_box(z) -> tuple:
    cx, cy, w, h = (float(v) for v in z[:4])
    w, h = max(w, 1e-4), max(h, 1e-4)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class KalmanBox:
    """Constant-velocity Kalman filter on (cx, cy, w, h). ``dt`` comes from real timestamps."""

    H = np.hstack([np.eye(4), np.zeros((4, 4))])

    def __init__(self, bbox, t: float):
        self.x = np.zeros(8)
        self.x[:4] = box_to_z(bbox)
        w, h = self.x[2], self.x[3]
        self.P = np.diag([(0.05 * w) ** 2, (0.05 * h) ** 2, (0.05 * w) ** 2, (0.05 * h) ** 2, 0.25, 0.25, 0.01, 0.01])
        self.R = np.diag(np.square(MEAS_STD))
        self.t = t

    @staticmethod
    def _FQ(dt: float):
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        q = np.array([ACC_CENTER, ACC_CENTER, ACC_SIZE, ACC_SIZE]) ** 2
        Q = np.zeros((8, 8))
        Q[:4, :4] = np.diag(q * dt**4 / 4)
        Q[:4, 4:] = Q[4:, :4] = np.diag(q * dt**3 / 2)
        Q[4:, 4:] = np.diag(q * dt**2)
        return F, Q

    def predict_to(self, t: float) -> None:
        dt = max(0.0, t - self.t)
        if dt > 0:
            F, Q = self._FQ(dt)
            self.x = F @ self.x
            self.x[2:4] = np.maximum(self.x[2:4], 1e-3)
            self.P = F @ self.P @ F.T + Q
            self.t = t

    def innovation(self, bbox):
        y = box_to_z(bbox) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def mahalanobis(self, bbox) -> float:
        """Squared Mahalanobis distance of the box center (2 dof) to the prediction."""
        y, S = self.innovation(bbox)
        return float(y[:2] @ np.linalg.solve(S[:2, :2], y[:2]))

    def update(self, bbox, t: float | None = None) -> None:
        if t is not None:
            self.predict_to(t)
        y, S = self.innovation(bbox)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def bbox(self) -> tuple:
        return z_to_box(self.x)

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[4]), float(self.x[5])


def hungarian(cost) -> list[tuple[int, int]]:
    """Minimum-cost assignment for a rectangular cost matrix (rows, cols). Returns (row, col) pairs.

    O(n^2 m) shortest augmenting path with potentials (no SciPy needed). Every row of the smaller
    dimension is assigned.
    """
    C = np.asarray(cost, dtype=float)
    if C.size == 0:
        return []
    transposed = C.shape[0] > C.shape[1]
    if transposed:
        C = C.T
    n, m = C.shape
    INF = float("inf")
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)  # p[j] = row assigned to column j (1-based), 0 = free
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)


def appearance_similarity(a, b) -> float | None:
    """Bhattacharyya coefficient of two L1-normalized histograms (1 = identical, 0 = disjoint)."""
    if a is None or b is None:
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return None
    return float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))

"""Face detectors: YuNet on ONNX Runtime (default), YuNet on OpenCV DNN and the Haar cascade fallback.

Every detector returns YuNet rows ``(x, y, w, h, 5 landmark pairs, score)`` in pixels of the given image.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..inference import OrtOptions, load_model


def nms_int_boxes(rows: np.ndarray, score_thr: float, nms_thr: float, top_k: int) -> np.ndarray:
    """Greedy NMS exactly as ``cv::dnn::NMSBoxes`` on ``Rect2i`` (boxes truncated to int, scores > threshold,
    stable descending sort, ``top_k`` candidates, keep while IoU <= ``nms_thr``)."""
    scores = rows[:, 14]
    order = [i for i in np.argsort(-scores, kind="stable") if scores[i] > score_thr]
    if top_k > 0:
        order = order[:top_k]
    boxes = rows[:, :4].astype(np.int32).astype(np.int64)  # C++ int(float): truncation toward zero
    x0, y0, w, h = boxes.T
    area = w * h
    keep: list[int] = []
    for i in order:
        if keep:
            k = np.asarray(keep)
            iw = np.minimum(x0[i] + w[i], x0[k] + w[k]) - np.maximum(x0[i], x0[k])
            ih = np.minimum(y0[i] + h[i], y0[k] + h[k]) - np.maximum(y0[i], y0[k])
            inter = np.where((iw > 0) & (ih > 0), iw * ih, 0)
            total = area[i] + area[k]
            # cv::jaccardDistance: two empty boxes count as identical (overlap 1)
            overlap = np.where(total > 0, inter / np.maximum(total - inter, 1), 1.0)
            if (overlap.astype(np.float32) > np.float32(nms_thr)).any():
                continue
        keep.append(int(i))
    return rows[keep]


class OrtYuNetDetector:
    """YuNet on ONNX Runtime with OpenCV ``FaceDetectorYN``'s pre/post-processing, so the rows match
    :class:`YuNetDetector` (zero-pad to a multiple of 32, BGR float input, ``sqrt(cls*obj)`` score, anchor-free
    decode per stride, NMS on integer boxes). Needs the dynamic-input export (``CATALOG["yunet"]``).

    Stateless per call, so it is safe to share between threads.
    """

    name = "YuNet"
    landmarks = True
    STRIDES = (8, 16, 32)

    def __init__(self, path, score: float = 0.5, nms: float = 0.3, top_k: int = 50, options: OrtOptions | None = None):
        self.model = load_model(path, options)
        self.score_thr, self.nms_thr, self.top_k = score, nms, top_k
        self._input = self.model.input_names[0]
        self._outputs = [f"{k}_{s}" for k in ("cls", "obj", "bbox", "kps") for s in self.STRIDES]

    @property
    def accelerator(self) -> str:
        return self.model.accelerator

    def detect(self, img):
        h, w = img.shape[:2]
        ph, pw = -(-h // 32) * 32, -(-w // 32) * 32
        x = np.zeros((1, 3, ph, pw), np.float32)
        x[0, :, :h, :w] = img.transpose(2, 0, 1)
        out = self.model.run({self._input: x}, self._outputs)
        n = len(self.STRIDES)
        rows = []
        for i, stride in enumerate(self.STRIDES):
            cls = np.clip(out[i][0, :, 0], 0.0, 1.0)
            obj = np.clip(out[i + n][0, :, 0], 0.0, 1.0)
            score = np.sqrt(cls * obj)
            idx = np.nonzero(score >= self.score_thr)[0]
            if not idx.size:
                continue
            cols = pw // stride
            r, c = (idx // cols).astype(np.float32), (idx % cols).astype(np.float32)
            bb, kp = out[i + 2 * n][0, idx], out[i + 3 * n][0, idx]
            bw, bh = np.exp(bb[:, 2]) * stride, np.exp(bb[:, 3]) * stride
            f = np.empty((idx.size, 15), np.float32)
            f[:, 0] = (c + bb[:, 0]) * stride - bw / 2
            f[:, 1] = (r + bb[:, 1]) * stride - bh / 2
            f[:, 2], f[:, 3] = bw, bh
            f[:, 4:14:2] = (kp[:, 0::2] + c[:, None]) * stride
            f[:, 5:14:2] = (kp[:, 1::2] + r[:, None]) * stride
            f[:, 14] = score[idx]
            rows.append(f)
        if not rows:
            return []
        faces = np.concatenate(rows)
        if len(faces) > 1:
            faces = nms_int_boxes(faces, self.score_thr, self.nms_thr, self.top_k)
        return [(f, float(f[14])) for f in faces]


class YuNetDetector:
    """YuNet on OpenCV DNN (fixed-size opencv_zoo export). Not thread-safe: ``setInputSize`` mutates the net."""

    name = "YuNet"
    landmarks = True
    accelerator = "cpu (OpenCV DNN)"

    def __init__(self, path, score=0.5):
        self.det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score, 0.3, 50)

    def detect(self, img):
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        return [] if faces is None else [(f, float(f[14])) for f in faces]


class HaarDetector:
    name = "Haar(fallback)"
    landmarks = False
    accelerator = "cpu (OpenCV)"

    def __init__(self):
        self.det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects, _, weights = self.det.detectMultiScale3(g, 1.1, 5, minSize=(40, 40), outputRejectLevels=True)
        out = []
        for (x, y, w, h), wt in zip(rects, np.ravel(weights) if len(rects) else [], strict=False):
            row = np.zeros(15, np.float32)
            row[:4] = (x, y, w, h)
            out.append((row, float(min(1.0, max(0.0, wt / 5.0)))))  # rough confidence
        return out

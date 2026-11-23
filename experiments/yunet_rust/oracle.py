"""Replay the production NumPy decode and NMS against the raw YuNet output heads."""

import cv2
import numpy as np

from presencekit.detectors.face import OrtYuNetDetector, nms_int_boxes

STRIDES = OrtYuNetDetector.STRIDES
LANES = (1, 1, 4, 10)
HEAD_WIDTHS = tuple(width for width in LANES for _ in STRIDES)


def _decode(pad_width, heads, score_thr):
    rows = []
    count = len(STRIDES)
    for i, stride in enumerate(STRIDES):
        cls = np.clip(heads[i].reshape(-1), 0.0, 1.0)
        obj = np.clip(heads[i + count].reshape(-1), 0.0, 1.0)
        score = np.sqrt(cls * obj)
        idx = np.nonzero(score >= score_thr)[0]
        if not idx.size:
            continue
        cols = pad_width // stride
        r, c = (idx // cols).astype(np.float32), (idx % cols).astype(np.float32)
        bb = heads[i + 2 * count].reshape(-1, 4)[idx]
        kp = heads[i + 3 * count].reshape(-1, 10)[idx]
        bw, bh = np.exp(bb[:, 2]) * stride, np.exp(bb[:, 3]) * stride
        f = np.empty((idx.size, 15), np.float32)
        f[:, 0] = (c + bb[:, 0]) * stride - bw / 2
        f[:, 1] = (r + bb[:, 1]) * stride - bh / 2
        f[:, 2], f[:, 3] = bw, bh
        f[:, 4:14:2] = (kp[:, 0::2] + c[:, None]) * stride
        f[:, 5:14:2] = (kp[:, 1::2] + r[:, None]) * stride
        f[:, 14] = score[idx]
        rows.append(f)
    return np.concatenate(rows) if rows else np.empty((0, 15), np.float32)


def _opencv_nms(rows, score_thr, nms_thr, top_k):
    keep = cv2.dnn.NMSBoxes(
        rows[:, :4].astype(np.int32).tolist(), rows[:, 14].tolist(), score_thr, nms_thr, top_k=max(top_k, 0)
    )
    return rows[np.asarray(keep, dtype=np.intp).ravel()]


def _postprocess(pad_width, heads, score_thr, nms_thr, top_k, *, nms=nms_int_boxes):
    rows = _decode(pad_width, heads, score_thr)
    return nms(rows, score_thr, nms_thr, top_k) if len(rows) > 1 else rows


def postprocess(
    pad_width: int, heads: list[list[float]], score_thr: float, nms_thr: float, top_k: int
) -> list[list[float]]:
    maps = [np.asarray(head, dtype=np.float32).reshape(-1, lane) for head, lane in zip(heads, HEAD_WIDTHS, strict=True)]
    return _postprocess(pad_width, maps, score_thr, nms_thr, top_k).tolist()

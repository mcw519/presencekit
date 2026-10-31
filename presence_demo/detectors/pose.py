"""Pose detector: MoveNet MultiPose Lightning (Apache-2.0) on ONNX Runtime (default) or LiteRT (fallback).

Both runners share the letterbox and the decoding, so they return the same bodies for the same model output.
The ONNX file is a local export of the TFLite model (``scripts/export_movenet_onnx.py``).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..inference import OrtOptions, load_model

COCO_KP = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder"]
COCO_KP_IDX = {**{n: i for i, n in enumerate(COCO_KP)}, "l_hip": 11, "r_hip": 12}


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """BGR image -> RGB uint8 canvas whose long side is ``size`` and both sides multiples of 32 (zero padding at
    the right/bottom), plus the scale applied."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    pw, ph = -(-nw // 32) * 32, -(-nh // 32) * 32
    canvas = np.zeros((ph, pw, 3), np.uint8)
    canvas[:nh, :nw] = cv2.cvtColor(cv2.resize(img, (nw, nh)), cv2.COLOR_BGR2RGB)
    return canvas, s


def decode(out: np.ndarray, canvas_hw: tuple[int, int], s: float, img_hw: tuple[int, int], conf: float) -> list:
    """MoveNet MultiPose output ``[6, 56]`` (17 x (y, x, score) normalized to the padded canvas, then ymin, xmin,
    ymax, xmax, score) -> ``[(bbox_xyxy, score, {name: (x, y, c)})]`` in pixels of the original image."""
    (ph, pw), (h, w) = canvas_hw, img_hw
    fx, fy = pw / s, ph / s  # normalized padded coords -> original pixels
    res = []
    for r in out:
        if r[55] < conf:
            continue
        ymin, xmin, ymax, xmax = r[51:55]
        bbox = (float(xmin * fx), float(ymin * fy), float(min(xmax * fx, w)), float(min(ymax * fy, h)))
        kps = r[:51].reshape(17, 3)
        kp = {n: (float(kps[j, 1] * fx), float(kps[j, 0] * fy), float(kps[j, 2])) for n, j in COCO_KP_IDX.items()}
        res.append((bbox, float(r[55]), kp))
    return res


class OrtMoveNetDetector:
    """MoveNet.MultiPose Lightning (up to 6 people, COCO-17 keypoints) on ONNX Runtime. Stateless per call."""

    name = "MoveNet MultiPose Lightning"

    def __init__(self, path, conf: float = 0.3, size: int = 256, options: OrtOptions | None = None):
        self.model = load_model(path, options)
        self.conf, self.size = conf, size
        self._input = self.model.input_names[0]

    @property
    def accelerator(self) -> str:
        return self.model.accelerator

    def detect(self, img):
        canvas, s = letterbox(img, self.size)
        out = self.model.run({self._input: canvas[None]})[0][0]
        return decode(out, canvas.shape[:2], s, img.shape[:2], self.conf)


class MoveNetMultiPoseDetector:
    """The same model as TFLite on LiteRT (``ai-edge-litert``, the ``litert`` extra). Not thread-safe (the
    interpreter is resized per input shape)."""

    name = "MoveNet MultiPose Lightning"
    accelerator = "cpu (LiteRT XNNPACK)"

    def __init__(self, path, conf: float = 0.3, size: int = 256, threads: int = 4):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(str(path), num_threads=threads)
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        self.conf, self.size, self._shape = conf, size, None

    def detect(self, img):
        canvas, s = letterbox(img, self.size)
        if self._shape != canvas.shape[:2]:
            self.it.resize_tensor_input(self.inp, [1, *canvas.shape])
            self.it.allocate_tensors()
            self._shape = canvas.shape[:2]
        self.it.set_tensor(self.inp, canvas[None])
        self.it.invoke()
        return decode(self.it.get_tensor(self.out)[0], canvas.shape[:2], s, img.shape[:2], self.conf)

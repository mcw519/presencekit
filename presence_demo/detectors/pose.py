"""Pose detector: MoveNet MultiPose Lightning (Apache-2.0) on CPU via LiteRT."""

from __future__ import annotations

import cv2
import numpy as np

COCO_KP = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder"]
COCO_KP_IDX = {**{n: i for i, n in enumerate(COCO_KP)}, "l_hip": 11, "r_hip": 12}


class MoveNetMultiPoseDetector:
    """MoveNet.MultiPose Lightning (up to 6 people, COCO-17 keypoints) on CPU via LiteRT.

    Input: RGB uint8, letterboxed so the long side is `size` (multiple of 32). Output [1, 6, 56]:
    17 x (y, x, score) normalized to the padded input, then (ymin, xmin, ymax, xmax, score)."""

    name = "MoveNet MultiPose Lightning"

    def __init__(self, path, conf: float = 0.3, size: int = 256, threads: int = 4):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(str(path), num_threads=threads)
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        self.conf, self.size, self._shape = conf, size, None

    def detect(self, img):
        h, w = img.shape[:2]
        s = self.size / max(h, w)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        pw, ph = -(-nw // 32) * 32, -(-nh // 32) * 32
        if self._shape != (ph, pw):
            self.it.resize_tensor_input(self.inp, [1, ph, pw, 3])
            self.it.allocate_tensors()
            self._shape = (ph, pw)
        canvas = np.zeros((ph, pw, 3), np.uint8)
        canvas[:nh, :nw] = cv2.cvtColor(cv2.resize(img, (nw, nh)), cv2.COLOR_BGR2RGB)
        self.it.set_tensor(self.inp, canvas[None])
        self.it.invoke()
        out = self.it.get_tensor(self.out)[0]
        fx, fy = pw / s, ph / s  # normalized padded coords -> original pixels
        res = []
        for r in out:
            if r[55] < self.conf:
                continue
            ymin, xmin, ymax, xmax = r[51:55]
            bbox = (float(xmin * fx), float(ymin * fy), float(min(xmax * fx, w)), float(min(ymax * fy, h)))
            kps = r[:51].reshape(17, 3)
            kp = {n: (float(kps[j, 1] * fx), float(kps[j, 0] * fy), float(kps[j, 2])) for n, j in COCO_KP_IDX.items()}
            res.append((bbox, float(r[55]), kp))
        return res

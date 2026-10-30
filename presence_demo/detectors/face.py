"""Face detectors: OpenCV YuNet (default) and the Haar cascade fallback bundled with OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


class YuNetDetector:
    name = "YuNet"

    def __init__(self, path, score=0.5):
        self.det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score, 0.3, 50)

    def detect(self, img):
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        return [] if faces is None else [(f, float(f[14])) for f in faces]


class HaarDetector:
    name = "Haar(fallback)"

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

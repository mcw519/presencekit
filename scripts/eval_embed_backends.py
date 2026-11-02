"""Accuracy of the SFace embedding backends vs the fp32 reference (in memory only).

Faces: CC0 portraits from the Cleveland Museum of Art Open Access API (scripts/faces_cc0.py), a set
*disjoint* from the INT8 calibration set (the first 30 'portrait' faces are skipped), plus the CC0 test
face. Same-identity pairs are (original, augmentation) — brightness, contrast, blur,
rotation/scale, JPEG q35, noise;
different-identity pairs are all pairs of distinct portraits. Nothing is written to disk.

    uv run python scripts/eval_embed_backends.py [--n 40]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from faces_cc0 import augment, fetch_aligned_faces  # noqa: E402

from presencekit.embed_backends import OpenCVSFace, TFLiteSFace  # noqa: E402
from presencekit.tracker import Config  # noqa: E402
from presencekit.vision import MODEL_DIR, SFACE, SFACE_INT8, YUNET  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    fp32 = MODEL_DIR / SFACE[0]
    calib = fetch_aligned_faces(MODEL_DIR / YUNET[0], fp32, n=30)  # same call as the export script
    seen = {i for i, _, _ in calib}
    faces = []
    for q in ("portrait", "man", "woman", "girl", "boy"):
        for f in fetch_aligned_faces(MODEL_DIR / YUNET[0], fp32, n=a.n, query=q):
            if f[0] not in seen and len(faces) < a.n:
                seen.add(f[0])
                faces.append(f)
    img = cv2.imread(str(ROOT / "tests" / "data" / "face.jpg"))
    det = cv2.FaceDetectorYN.create(str(MODEL_DIR / YUNET[0]), "", img.shape[1::-1], 0.8)
    rec = cv2.FaceRecognizerSF.create(str(fp32), "")
    faces.append(("test", "face.jpg", rec.alignCrop(img, det.detect(img)[1][0])))
    rng = np.random.default_rng(0)
    groups = [[f, *augment(f, rng)] for _, _, f in faces]
    backends = {"fp32 (ONNX, OpenCV)": OpenCVSFace(fp32)}
    backends["int8 (ONNX, OpenCV)"] = OpenCVSFace(MODEL_DIR / SFACE_INT8[0], "int8")
    for name, f in (
        ("npu int8 (TFLite)", "face_recognition_sface_2021dec_int8.tflite"),
        ("npu fp16 (TFLite)", "face_recognition_sface_2021dec_fp16.tflite"),
    ):
        if (MODEL_DIR / f).exists():
            backends[name] = TFLiteSFace(MODEL_DIR / f)
    emb = {k: [[b.feature_aligned(c) for c in g] for g in groups] for k, b in backends.items()}
    thr = Config().cos_match
    ref = emb["fp32 (ONNX, OpenCV)"]
    same = [(i, 0, i, j) for i in range(len(groups)) for j in range(1, len(groups[i]))]
    diff = [(i, 0, k, 0) for i, k in itertools.combinations(range(len(groups)), 2)]
    print(
        f"{len(faces)} 張 CC0 人像（和校正資料不重疊）+ 每張 6 種增強；同一人 {len(same)} 對、不同人 {len(diff)} 對；門檻 cos_match={thr}\n"  # noqa: E501
    )
    print(
        "| 後端 | 與 fp32 cos 平均 | 最小 | 同一人 cos 平均 | 不同人 cos 最大 | 同一人通過率 | 不同人誤配率 | 與 fp32 判定一致 |"  # noqa: E501
    )
    print("|---|---|---|---|---|---|---|---|")
    for k, e in emb.items():
        cs = [float(e[i][j] @ ref[i][j]) for i in range(len(groups)) for j in range(len(groups[i]))]
        s = [float(e[i][a_] @ e[k2][b_]) for i, a_, k2, b_ in same]
        d = [float(e[i][a_] @ e[k2][b_]) for i, a_, k2, b_ in diff]
        rs = [float(ref[i][a_] @ ref[k2][b_]) for i, a_, k2, b_ in same + diff]
        agree = np.mean([(x >= thr) == (y >= thr) for x, y in zip(s + d, rs, strict=True)])
        print(
            f"| {k} | {np.mean(cs):.4f} | {np.min(cs):.4f} | {np.mean(s):.3f} | {np.max(d):.3f} | "
            f"{np.mean(np.array(s) >= thr):.1%} | {np.mean(np.array(d) >= thr):.2%} | {agree:.2%} |"
        )
    # best threshold per backend (max of TPR - FPR on these pairs)
    print()
    for k, e in emb.items():
        s = np.array([float(e[i][a_] @ e[k2][b_]) for i, a_, k2, b_ in same])
        d = np.array([float(e[i][a_] @ e[k2][b_]) for i, a_, k2, b_ in diff])
        ts = np.arange(0.2, 0.7, 0.01)
        j = [np.mean(s >= t) - np.mean(d >= t) for t in ts]
        print(f"{k}: 最佳門檻（TPR−FPR 最大）≈ {ts[int(np.argmax(j))]:.2f}")


if __name__ == "__main__":
    main()

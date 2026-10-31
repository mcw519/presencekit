"""Model catalogue and local model store.

Every model file the SDK can use is described once in :data:`CATALOG` (file name, download URL, sha256, license,
origin). :class:`ModelStore` decides where the files live, downloads the downloadable ones on first use and
verifies checksums. Files that cannot be downloaded (produced by an export script, or copied from a repository)
name the command that produces them.

Store root: ``ModelStore(root)``, else ``$PRESENCE_MODEL_DIR``, else ``models/`` of a source checkout, else the
per-user cache (``$XDG_CACHE_HOME/presencekit/models``, default ``~/.cache/presencekit/models``). Model files
are the only files this module writes.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tarfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)
MIN_BYTES = 1000  # smaller files are error pages or git-LFS pointers, not models

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
# libfacedetection.train (BSD-3-Clause), pinned commit; yunet_s_640_640.onnx there is byte-identical to opencv_zoo's
# face_detection_yunet_2023mar.onnx, and yunet_s_dynamic.onnx is the same network exported with a dynamic input size.
LFD = "https://github.com/ShiqiYu/libfacedetection.train/raw/2b5e61dee21441ef4676505f835ab63ee53bae1e/onnx/"
KAGGLE_MOVENET = (
    "https://www.kaggle.com/api/v1/models/google/movenet/tfLite/multipose-lightning-tflite-float16/1/download"
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str | None = None  # None: not downloadable, see ``produce``
    sha256: str | None = None
    license: str = ""
    source: str = ""
    produce: str = ""  # command that creates or copies a non-downloadable file
    archive_suffix: str | None = None  # the download is a tar archive: keep its single member ending with this


CATALOG: dict[str, ModelSpec] = {
    s.key: s
    for s in (
        ModelSpec(
            "yunet",
            "yunet_s_dynamic.onnx",
            LFD + "yunet_s_dynamic.onnx",
            "daede24002cc8590f457998cc5e6c9e8b82f45f6bd969506a8913cd4ed10a2d3",
            "BSD-3-Clause",
            "libfacedetection.train (YuNet-s, dynamic input; same weights as opencv_zoo 2023mar)",
        ),
        ModelSpec(
            "yunet-opencv",
            "face_detection_yunet_2023mar.onnx",
            ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
            "MIT",
            "opencv_zoo (fixed 640x640 input; used by the OpenCV DNN fallback only)",
        ),
        ModelSpec(
            "sface",
            "face_recognition_sface_2021dec.onnx",
            ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
            "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
            "Apache-2.0",
            "opencv_zoo",
        ),
        ModelSpec(
            "sface-int8",
            "face_recognition_sface_2021dec_int8.onnx",
            ZOO + "face_recognition_sface/face_recognition_sface_2021dec_int8.onnx",
            "2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a",
            "Apache-2.0",
            "opencv_zoo (official INT8 quantization)",
        ),
        ModelSpec(
            "sface-tflite-int8",
            "face_recognition_sface_2021dec_int8.tflite",
            license="Apache-2.0 (derived from SFace)",
            source="local export",
            produce="scripts/export_sface_tflite.py",
        ),
        ModelSpec(
            "sface-tflite-fp16",
            "face_recognition_sface_2021dec_fp16.tflite",
            license="Apache-2.0 (derived from SFace)",
            source="local export",
            produce="scripts/export_sface_tflite.py",
        ),
        ModelSpec(
            "movenet",
            "movenet_multipose_lightning.onnx",
            license="Apache-2.0 (derived from MoveNet)",
            source="local export of the MoveNet MultiPose Lightning TFLite model",
            produce="scripts/export_movenet_onnx.py",
        ),
        ModelSpec(
            "movenet-tflite",
            "movenet_multipose_lightning_fp16.tflite",
            KAGGLE_MOVENET,
            "d4489f89e6bd6777a8b9a1a16189832131f84ff90d82fae729e670b84d7948dd",
            license="Apache-2.0",
            source="Google MoveNet (Kaggle Models)",
            archive_suffix=".tflite",
        ),
        ModelSpec(
            "puresound",
            "dpcrn_curriculum_v2.onnx",
            sha256="9a79e9eaeab9a6e288fec7216adb1c87a05b557948e8aa014f5f6096e1f375e4",
            license="Apache-2.0",
            source="PureSound (github.com/mcw519/puresound)",
            produce="scripts/fetch_puresound_model.py --clone",
        ),
        ModelSpec(
            "puresound-manifest",
            "dpcrn_curriculum_v2.json",
            sha256="92de972e9dc9be407e48b51f1b8c8351493a99ee942f8c9b662ac5ab48b3aaa9",
            license="Apache-2.0",
            source="PureSound (github.com/mcw519/puresound)",
            produce="scripts/fetch_puresound_model.py --clone",
        ),
    )
}


def default_root() -> Path:
    """``$PRESENCE_MODEL_DIR``, else ``models/`` of a source checkout, else the per-user cache directory."""
    env = os.environ.get("PRESENCE_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").exists() and (checkout / "presencekit").is_dir():
        return checkout / "models"
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    root = Path(cache) / "presencekit" / "models"
    legacy = Path(cache) / "presence-demo" / "models"
    return legacy if not root.exists() and legacy.is_dir() else root


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelStore:
    """Locates, downloads and verifies model files under one root directory."""

    def __init__(self, root: str | Path | None = None, *, allow_download: bool = True):
        self.root = Path(root).expanduser() if root else default_root()
        self.allow_download = allow_download
        self._verified: set[Path] = set()
        self._lock = threading.Lock()

    @staticmethod
    def spec(model: str | ModelSpec) -> ModelSpec:
        return model if isinstance(model, ModelSpec) else CATALOG[model]

    def path(self, model: str | ModelSpec) -> Path:
        return self.root / self.spec(model).filename

    def local(self, model: str | ModelSpec) -> Path | None:
        """The file when present (and, with a known sha256, verified once); never touches the network."""
        spec = self.spec(model)
        p = self.root / spec.filename
        if not p.exists() or p.stat().st_size < MIN_BYTES:
            return None
        if spec.sha256 and p not in self._verified:
            if sha256_file(p) != spec.sha256:
                log.warning("checksum mismatch for %s", p)
                return None
            self._verified.add(p)
        return p

    def ensure(self, model: str | ModelSpec, *, allow_download: bool | None = None) -> Path | None:
        """The local file, downloading it first when allowed and possible. None when unavailable."""
        spec = self.spec(model)
        with self._lock:
            p = self.local(spec)
            if p is not None:
                return p
            if not spec.url or not (self.allow_download if allow_download is None else allow_download):
                return None
            return self._download(spec)

    def _download(self, spec: ModelSpec) -> Path | None:
        dst = self.root / spec.filename
        tmp = dst.with_suffix(dst.suffix + ".part")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            log.info("downloading %s", spec.url)
            with urllib.request.urlopen(spec.url, timeout=60) as r:
                data = r.read()
            if spec.archive_suffix:
                with tarfile.open(fileobj=io.BytesIO(data)) as tf:
                    member = next(m for m in tf.getmembers() if m.name.endswith(spec.archive_suffix))
                    data = tf.extractfile(member).read()
            if spec.sha256 and hashlib.sha256(data).hexdigest() != spec.sha256:
                raise ValueError(f"sha256 mismatch for {spec.filename}")
            tmp.write_bytes(data)
            tmp.replace(dst)
            self._verified.add(dst)
            return dst
        except Exception as e:  # noqa: BLE001 - offline, moved URL, bad archive...
            log.warning("model download failed (%s): %s", spec.filename, e)
            tmp.unlink(missing_ok=True)
            return None

    def status(self) -> list[dict]:
        """One row per catalogue entry: present / downloadable / how to produce it (no hashing, no network)."""
        return [
            {
                "key": s.key,
                "file": s.filename,
                "present": (self.root / s.filename).exists(),
                "downloadable": bool(s.url),
                "produce": s.produce,
                "license": s.license,
            }
            for s in CATALOG.values()
        ]


_default: ModelStore | None = None


def default_store() -> ModelStore:
    """The process-wide store at :func:`default_root` (created on first use)."""
    global _default
    if _default is None or _default.root != default_root():
        _default = ModelStore()
    return _default

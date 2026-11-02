"""Copy the PureSound voice-isolation model (dpcrn_curriculum_v2: per-frame ONNX + manifest) into models/.

The model lives in the PureSound repository (https://github.com/mcw519/puresound, public, Apache-2.0),
under ``egs/voice_isolate/pretrained_ckpt/streaming/``. presencekit never installs PureSound at runtime;
this script only copies the two files and verifies their sha256 (models/ is gitignored, nothing is committed).

    # from an existing clone
    uv run python scripts/fetch_puresound_model.py --from /path/to/puresound
    # or let the script shallow-clone the public repository
    uv run python scripts/fetch_puresound_model.py --clone
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presencekit.audio import PURESOUND_FILES, PURESOUND_SUBDIR  # noqa: E402
from presencekit.inference import default_root  # noqa: E402

REPO = "https://github.com/mcw519/puresound.git"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy_from(src_root: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name, want in PURESOUND_FILES.items():
        src = src_root / PURESOUND_SUBDIR / name
        if not src.exists():
            raise SystemExit(f"missing {src}")
        got = sha256(src)
        if got != want:
            raise SystemExit(f"sha256 mismatch for {name}: {got} != {want}")
        tmp = dst / (name + ".part")
        shutil.copyfile(src, tmp)
        if sha256(tmp) != want:
            tmp.unlink()
            raise SystemExit(f"copy of {name} corrupted")
        tmp.replace(dst / name)
        print(f"ok {name} sha256={want[:12]}…")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from", dest="src", type=Path, help="path to a PureSound clone")
    g.add_argument("--clone", action="store_true", help="shallow-clone the public PureSound repo to a temp dir")
    ap.add_argument(
        "--dest", type=Path, default=default_root(), help="model store (default: models/ or $PRESENCE_MODEL_DIR)"
    )
    a = ap.parse_args()
    if a.src:
        copy_from(a.src, a.dest)
        return
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, d], check=True)
        copy_from(Path(d), a.dest)


if __name__ == "__main__":
    main()

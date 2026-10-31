"""Run offline synthetic/mock PoC acceptance; see --help for suite selection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presencekit.acceptance import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

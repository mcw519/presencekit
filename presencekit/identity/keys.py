"""Memory-only API key for the cloud identity matcher (env ``TYPESAFE_API_KEY`` or set from the UI).

PRIVACY: the key is never logged, echoed or written anywhere. ``key_status()`` exposes only whether it is set.
"""

from __future__ import annotations

import os

_key: dict = {"value": os.environ.get("TYPESAFE_API_KEY", "") or "", "source": "env"}
if not _key["value"]:
    _key["source"] = None


def set_key(value: str | None) -> dict:
    v = (value or "").strip()
    _key["value"], _key["source"] = v, ("ui" if v else None)
    return key_status()


def key_status() -> dict:
    return {"set": bool(_key["value"]), "source": _key["source"]}

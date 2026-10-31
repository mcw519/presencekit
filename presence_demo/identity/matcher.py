"""Identity matcher protocol and the shipped cloud backend (Typesafe Jev)."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .keys import _key

JEV_URL = os.environ.get("REID_JEV_URL") or "https://api.typesafe.ai/v1/systemone"  # override: tests / proxies
QUESTION = (
    "Are the person in `reference` and the person in `query` the same individual? "
    "Judge identity from facial features, not clothing, pose, or background."
)


@runtime_checkable
class IdentityMatcher(Protocol):
    """``compare(reference_jpeg, query_jpeg) -> (score 0..1, meta)``; raises :class:`ReidError` on failure.

    ``meta`` may carry ``model`` and ``in_tok`` (input tokens) for accounting."""

    name: str

    async def compare(self, ref: bytes, query: bytes, *, model: str, timeout: float) -> tuple[float, dict]: ...


class ReidError(RuntimeError):
    def __init__(self, kind: str, msg: str = ""):
        super().__init__(msg or kind)
        self.kind = kind


class CloudReid:
    """Backend interface: ``compare(reference_jpeg, query_jpeg) -> (score 0..1, meta)``. Raises ReidError."""

    name = "base"

    async def compare(self, ref: bytes, query: bytes, *, model: str, timeout: float) -> tuple[float, dict]:
        raise NotImplementedError


class JevClient(CloudReid):
    """Typesafe Jev images API (``noul`` question over two image assets). Call shape follows the LFW benchmark."""

    name = "jev"

    def __init__(self, url: str | None = None, key_fn: Callable[[], str] = lambda: _key["value"]):
        self.url = url or JEV_URL
        self.key_fn = key_fn

    def body(self, ref: bytes, query: bytes, model: str) -> dict:
        r1, r2 = str(uuid.uuid4()), str(uuid.uuid4())
        enc = lambda b: base64.b64encode(b).decode()  # noqa: E731
        return {
            "model": model,
            "state": {"reference": r1, "query": r2},
            "assets": {
                r1: {"type": "image", "format": "jpg", "data_base64": enc(ref)},
                r2: {"type": "image", "format": "jpg", "data_base64": enc(query)},
            },
            "questions": {"same": {"type": "noul", "instructions": QUESTION}},
        }

    def _post(self, data: bytes, key: str, timeout: float):
        req = urllib.request.Request(
            self.url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "TypeSafe-Danger-Zone-Images": "true",  # opt-in for image assets (meaning unverified; docs/reid.md)
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https URL / test override)
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise ReidError(f"http_{e.code}") from None
        except (TimeoutError, OSError) as e:
            raise ReidError(
                "timeout" if "timed out" in str(e).lower() or isinstance(e, TimeoutError) else "network"
            ) from None
        except ValueError:
            raise ReidError("bad_response") from None

    async def compare(self, ref: bytes, query: bytes, *, model: str, timeout: float) -> tuple[float, dict]:
        key = self.key_fn()
        if not key:
            raise ReidError("no_key")
        data = json.dumps(self.body(ref, query, model)).encode()
        j = await asyncio.to_thread(self._post, data, key, timeout)
        try:
            score = float(((j.get("answers") or {}).get("same") or {})["noul"])
        except (KeyError, TypeError, ValueError):
            raise ReidError("bad_response") from None
        return max(0.0, min(1.0, score)), {
            "model": j.get("model"),
            "in_tok": (j.get("usage") or {}).get("input_tokens"),
        }


# v0.13 layer name of the shipped matcher
JevMatcher = JevClient

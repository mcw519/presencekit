"""Identity matcher protocol and the shipped cloud backend (Typesafe Jev)."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence
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
    """Typesafe Jev images API, with one ``noul`` question per reference face."""

    name = "jev"

    def __init__(self, url: str | None = None, key_fn: Callable[[], str] = lambda: _key["value"]):
        self.url = url or JEV_URL
        self.key_fn = key_fn

    def body(self, ref: bytes, query: bytes, model: str) -> dict:
        return self._body(query, {"same": ("reference", ref)}, model)

    @staticmethod
    def _body(query: bytes, refs: dict[str, tuple[str, bytes]], model: str) -> dict:
        images = {"query": query, **dict(refs.values())}
        state = {name: str(uuid.uuid4()) for name in images}
        return {
            "model": model,
            "state": state,
            "assets": {
                state[name]: {"type": "image", "format": "jpg", "data_base64": base64.b64encode(jpeg).decode()}
                for name, jpeg in images.items()
            },
            "questions": {
                key: {"type": "noul", "instructions": QUESTION.replace("`reference`", f"`{name}`")}
                for key, (name, _) in refs.items()
            },
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
        scores, meta = await self._compare(self.body(ref, query, model), timeout)
        return scores[0], meta

    async def compare_batch(
        self, refs: Sequence[bytes], query: bytes, *, model: str, timeout: float
    ) -> tuple[list[float], dict]:
        """Compare one query to all references in one request, preserving reference order."""
        if not refs:
            return [], {}
        assets = {f"same_{i}": (f"reference_{i}", ref) for i, ref in enumerate(refs)}
        return await self._compare(self._body(query, assets, model), timeout)

    async def _compare(self, body: dict, timeout: float) -> tuple[list[float], dict]:
        key = self.key_fn()
        if not key:
            raise ReidError("no_key")
        data = json.dumps(body).encode()
        j = await asyncio.to_thread(self._post, data, key, timeout)
        scores = []
        try:
            for question in body["questions"]:
                value = j["answers"][question]["noul"]
                if isinstance(value, bool):
                    raise ValueError
                score = float(value)
                if not 0 <= score <= 1:  # Also rejects NaN and infinity.
                    raise ValueError
                scores.append(score)
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ReidError("bad_response") from None
        usage = j.get("usage")
        tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        return scores, {"model": j.get("model"), "in_tok": tokens if type(tokens) is int and tokens >= 0 else None}


# v0.13 layer name of the shipped matcher
JevMatcher = JevClient

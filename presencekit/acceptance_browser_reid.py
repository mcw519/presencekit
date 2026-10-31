"""Offline Chrome re-ID control-flow acceptance, with an in-process mock client.

Mock scores verify orchestration only; they say nothing about Jev identity quality.
"""

from __future__ import annotations

import asyncio
import math
import time

from . import acceptance_browser as browser

METHOD = {
    **browser.METHOD,
    "profile": "rules/vision synthetic face; re-ID consent toggled via production config; crops display off",
    "reid": "production observe/crop/compare/handoff with child-only injected CloudReid mock",
    "limitations": "No real Jev, physical devices, identity accuracy or RTF performance gate",
}
KEYS = browser.NUMERIC_KEYS | {
    "handoffs",
    "resumed",
    "stale",
    "provisionalDifferent",
    "uiCurrent",
    "uiResumed",
    "uiCleared",
    "refs",
    "calling",
}


def _case(name, ok, expected, actual=None):
    return {
        "id": "browser-reid." + name,
        "suite": "browser-reid",
        "status": "passed" if ok else "failed",
        "expected": expected,
        "actual": actual,
        "details": {"method": METHOD},
    }


def _numbers(value):
    return {k: v for k, v in value.items() if k in KEYS and isinstance(v, (int, float)) and math.isfinite(v)}


def _media_progress(before, after):
    return (
        after["session"] == before["session"]
        and after["errors"] == after["ended"] == 0
        and all(
            after[key] > before[key]
            for key in ("currentFramesProcessed", "currentPcmSent", "currentVideoSent", "currentAudioReceived")
        )
    )


async def _snapshot(cdp):
    return _numbers(await cdp.evaluate("acceptanceReid.snapshot()"))


async def _wait(cdp, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = await _snapshot(cdp)
        if predicate(value):
            return value
        await asyncio.sleep(0.15)
    raise TimeoutError("Browser re-ID contract timed out")


async def _control(cdp, action):
    # Control route is registered only in this isolated child, never production.
    return await cdp.evaluate("fetch('/_acceptance/reid/" + action + "',{method:'POST'}).then(r=>r.json())")


async def _wait_control(cdp, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = await _control(cdp, "status")
        if predicate(value):
            return value
        await asyncio.sleep(0.05)
    raise TimeoutError("Mock completion contract timed out")


def _late_result_discarded(before, after):
    return (
        after["session"] == before["session"]
        and after["handoffs"] == before["handoffs"]
        and after["resumed"] == before["resumed"]
        and after["uiCleared"] == 1
        and after["stale"] == 0
    )


async def _exercise(cdp, port, seconds):
    cases = []
    await cdp.evaluate("acceptanceReid.configure(false);acceptanceFixture.begin()")
    await _wait(cdp, lambda s: s["session"] > 0 and s["audioReceived"] > 0 and s["framesProcessed"] > 0)
    await asyncio.sleep(seconds)
    disabled = await _control(cdp, "status")
    cases.append(
        _case(
            "consent_required",
            disabled["calls"] == disabled["members"] == 0,
            "No comparison or retained face before consent",
            disabled,
        )
    )
    await cdp.evaluate("acceptanceReid.consent(true)")
    await _wait(cdp, lambda s: s["handoffs"] > 0)
    deadline = time.monotonic() + 8
    while True:
        retained = await _control(cdp, "status")
        if retained["members"] > 0:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Reference crop unavailable")
        await asyncio.sleep(0.2)
    first = await _snapshot(cdp)
    await cdp.evaluate("acceptanceFixture.reconnect()")
    pending = await _wait(cdp, lambda s: s["session"] > first["session"] and s["calling"] == 1)
    cases.append(
        _case(
            "provisional_conversation",
            pending["provisionalDifferent"] == 1 and pending["resumed"] == 0,
            "New session receives a distinct provisional conversation before handoff",
            pending,
        )
    )
    await _control(cdp, "release")
    resumed = await _wait(cdp, lambda s: s["resumed"] == 1)
    cases.append(
        _case(
            "same_page_handoff",
            resumed["stale"] == 0
            and resumed["sameCapability"] == 1
            and resumed["uiCurrent"] == 1
            and resumed["uiResumed"] == 1,
            "Unique mock match resumes original conversation; UI matches current session",
            resumed,
        )
    )
    completed = await _control(cdp, "hold")
    await cdp.evaluate("acceptanceFixture.reconnect()")
    pending = await _wait(cdp, lambda s: s["session"] > resumed["session"] and s["calling"] == 1)
    confirmed = await _control(cdp, "status")
    if confirmed["calls"] <= completed["calls"]:
        raise RuntimeError("Pending mock comparison was not observed")
    await asyncio.sleep(3)
    during_call = await _snapshot(cdp)
    cases.append(
        _case(
            "nonblocking_comparison",
            _media_progress(pending, during_call) and during_call["calling"] == 1,
            "JPEG and PCM continue during a deliberately pending comparison",
            during_call,
        )
    )
    await cdp.evaluate("acceptanceReid.consent(false)")
    revoked = await _wait_control(cdp, lambda s: s["members"] == 0 and s["cancelled"] > confirmed["cancelled"])
    cases.append(
        _case(
            "consent_revocation",
            revoked["members"] == 0 and revoked["cancelled"] > 0,
            "Revocation clears all references and cancels pending comparison",
            revoked,
        )
    )
    await _control(cdp, "release")
    await _wait_control(cdp, lambda s: s["completed_after_cancel"] > revoked["completed_after_cancel"])
    await asyncio.sleep(seconds)
    after = await _snapshot(cdp)
    cases.append(
        _case(
            "late_result_discarded",
            _late_result_discarded(during_call, after),
            "Delayed cancelled mock response cannot handoff or leave stale UI",
            after,
        )
    )
    cases.append(
        _case(
            "nonblocking_media",
            _media_progress(pending, after),
            "JPEG and PCM continue during re-ID/revocation without false ENDED",
            after,
        )
    )
    await cdp.evaluate("acceptanceFixture.finish()")
    stopped = await _wait(cdp, lambda s: s["productionContextLive"] == 0)
    cases.append(
        _case(
            "cleanup",
            all(stopped[k] == 0 for k in ("running", "tracksLive", "sourceContextsLive", "frameTimerLive"))
            and stopped["errors"] == 0,
            "Synthetic sources and production capture cleaned without errors",
            stopped,
        )
    )
    health = await asyncio.to_thread(browser.pipeline._health, port)
    cases.append(_case("server_errors", health["errors"] == 0, {"errors": 0}, {"errors": health["errors"]}))
    return cases


def run_browser_reid(seeds=20, seconds=10):
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 3 <= seconds <= 3600:
        raise ValueError("seconds must be finite and in 3..3600")
    missing = (
        "Local Google Chrome is unavailable; no installation attempted."
        if not browser.CHROME.is_file()
        else browser.pipeline._preflight("twofaces")
    )
    if missing:
        case = _case("availability", False, "Local Chrome and offline models")
        case.update(status="not_run", actual={"reason": missing})
        return [case]
    try:
        return asyncio.run(
            browser._isolated(
                seconds,
                server_module=__name__,
                extra_fixture=browser.ROOT / "scripts/acceptance_browser_reid_fixture.js",
                exercise=_exercise,
            )
        )
    except Exception as exc:
        return [
            _case(
                "execution",
                False,
                "All re-ID browser steps complete within deadlines",
                {"error_type": type(exc).__name__},
            )
        ]


def _serve(fd):
    from . import reid
    from .server import app
    from .web.continuity import PAGES

    class Mock(reid.CloudReid):
        def __init__(self):
            self.gate = asyncio.Event()
            self.calls = self.cancelled = self.completed = self.completed_after_cancel = 0

        async def compare(self, ref, query, *, model, timeout):
            self.calls += 1
            cancelled = False
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                cancelled = True
                # Deliberately cancellation-resistant: production generation guard must discard it.
                await self.gate.wait()
            self.completed += 1
            self.completed_after_cancel += int(cancelled)
            return 0.99, {}

    mock = Mock()
    original = reid.CrossSessionReid

    class Injected(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, client=mock, **kwargs)

    reid.CrossSessionReid = Injected

    @app.post("/_acceptance/reid/{action}")
    async def control(action: str):
        if action == "release":
            mock.gate.set()
        elif action == "hold":
            mock.gate.clear()
        return {
            "calls": mock.calls,
            "cancelled": mock.cancelled,
            "completed": mock.completed,
            "completed_after_cancel": mock.completed_after_cancel,
            "members": sum(len(p.store.refs) + len(p.store.active) for p in PAGES.pages.values())
            + sum(len(d.members) for p in PAGES.pages.values() for d in p.store.drivers),
        }

    browser._serve(fd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-fd", type=int, required=True)
    _serve(parser.parse_args().serve_fd)

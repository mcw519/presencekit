"""Contract checks without browser, media or cloud access."""

import json
from pathlib import Path

import pytest

from presencekit import acceptance_browser_reid as adapter


@pytest.mark.parametrize("seconds", [None, "3", 2, 3601, float("nan"), float("inf")])
def test_duration_validation(seconds):
    with pytest.raises(ValueError):
        adapter.run_browser_reid(seconds=seconds)


def test_missing_chrome_and_models_are_not_run(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter.browser, "CHROME", tmp_path / "missing")
    assert adapter.run_browser_reid()[0]["status"] == "not_run"
    monkeypatch.setattr(adapter.browser, "CHROME", Path(__file__))
    monkeypatch.setattr(adapter.browser.pipeline, "_preflight", lambda scene: "Missing offline models")
    assert adapter.run_browser_reid()[0]["status"] == "not_run"


def test_shared_lifecycle_hooks_and_case_contract(monkeypatch):
    monkeypatch.setattr(adapter.browser, "CHROME", Path(__file__))
    monkeypatch.setattr(adapter.browser.pipeline, "_preflight", lambda scene: None)

    async def isolated(seconds, **kwargs):
        assert seconds == 3
        assert kwargs["server_module"] == "presencekit.acceptance_browser_reid"
        assert kwargs["exercise"] is adapter._exercise
        assert kwargs["extra_fixture"].is_file()
        return [adapter._case("test", True, "contract")]

    monkeypatch.setattr(adapter.browser, "_isolated", isolated)
    case = adapter.run_browser_reid(seconds=3)[0]
    assert case["id"] == "browser-reid.test"
    assert case["suite"] == "browser-reid"
    assert case["status"] == "passed"


def test_exception_output_contains_only_type(monkeypatch):
    monkeypatch.setattr(adapter.browser, "CHROME", Path(__file__))
    monkeypatch.setattr(adapter.browser.pipeline, "_preflight", lambda scene: None)

    async def failed(*args, **kwargs):
        raise RuntimeError("private CID capability media key")

    monkeypatch.setattr(adapter.browser, "_isolated", failed)
    result = adapter.run_browser_reid(seconds=3)
    assert result[0]["status"] == "failed"
    assert result[0]["actual"] == {"error_type": "RuntimeError"}
    assert "private" not in json.dumps(result)


def test_observable_result_drops_identity_media_and_nonfinite_values():
    assert adapter._numbers(
        {
            "resumed": 1,
            "refs": 0,
            "calling": float("nan"),
            "conversation_id": "x",
            "image": b"x",
            "page_token": "secret",
        }
    ) == {"resumed": 1, "refs": 0}


def test_fixture_does_not_clear_production_ui_or_persist_identity():
    source = (adapter.browser.ROOT / "scripts/acceptance_browser_reid_fixture.js").read_text()
    assert "REID={}" not in source and "REID.handoff=" not in source
    assert "localStorage" not in source and "MediaRecorder" not in source
    assert "reid_show_crops:false" in source


def test_media_gate_uses_fresh_current_socket_not_historical_maximum():
    before = dict(
        session=3,
        errors=0,
        ended=0,
        framesProcessed=40,
        currentFramesProcessed=2,
        currentPcmSent=30,
        currentVideoSent=2,
        currentAudioReceived=20,
    )
    after = dict(before, currentFramesProcessed=4, currentPcmSent=40, currentVideoSent=4, currentAudioReceived=30)
    assert adapter._media_progress(before, after)
    assert not adapter._media_progress(before, dict(after, currentFramesProcessed=2))
    assert not adapter._media_progress(before, dict(after, session=4))


def test_late_gate_rejects_new_decision_wrong_session_and_stale_ui():
    before = dict(session=3, handoffs=2, resumed=1)
    after = dict(before, uiCleared=1, stale=0)
    assert adapter._late_result_discarded(before, after)
    for change in ({"session": 4}, {"handoffs": 3}, {"resumed": 2}, {"uiCleared": 0}, {"stale": 1}):
        assert not adapter._late_result_discarded(before, dict(after, **change))


def test_mock_wait_requires_observed_completion(monkeypatch):
    import asyncio

    observations = iter([{"completed_after_cancel": 0}, {"completed_after_cancel": 1}])

    async def control(cdp, action):
        return next(observations)

    monkeypatch.setattr(adapter, "_control", control)
    assert asyncio.run(adapter._wait_control(None, lambda s: s["completed_after_cancel"] > 0)) == {
        "completed_after_cancel": 1
    }
    with pytest.raises(TimeoutError):
        asyncio.run(adapter._wait_control(None, lambda s: True, timeout=0))


@pytest.mark.parametrize("bad", ["", "wrong_provisional", "wrong_old_ref", "wrong_session", "bad_then_good"])
def test_real_js_fixture_validates_entire_handoff_contract(bad):
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = r"""
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
class Socket {
  constructor(){this.handlers=[]}
  addEventListener(kind, fn){this.handlers.push(fn)}
  emit(m){this.handlers.forEach(fn=>fn({data:JSON.stringify(m)}))}
}
const c={window:{WebSocket:Socket},acceptanceFixture:{snapshot:()=>({session:2})},REID:{},last:{session:2}};
vm.createContext(c);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),c);
const s=new c.window.WebSocket();
s.emit({type:'state',session:1});s.emit({type:'handoff',session:1,decision:'new',conversation_id:'old'});
s.emit({type:'state',session:2});s.emit({type:'reid',session:2,phase:'calling',conversation_id:'provisional'});
const good={type:'handoff',session:2,decision:'resume',conversation_id:'old',
  previous_conversation_id:'provisional',resumed_from_session:1};
const bad=process.argv[2];let result={...good};
if(bad==='wrong_provisional'||bad==='bad_then_good')result.previous_conversation_id='WRONG-PROVISIONAL';
if(bad==='wrong_old_ref')result.resumed_from_session=999;
if(bad==='wrong_session')result.session=1;
s.emit(result);if(bad==='bad_then_good')s.emit(good);
assert.equal(c.window.acceptanceReid.snapshot().stale,bad?1:0);
assert.equal(c.window.acceptanceReid.snapshot().provisionalDifferent,1);
"""
    subprocess.run(
        [node, "-e", script, str(adapter.browser.ROOT / "scripts/acceptance_browser_reid_fixture.js"), bad],
        check=True,
        capture_output=True,
        text=True,
    )

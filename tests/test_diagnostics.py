"""Diagnostic events (session_open, track lifecycle, face misses, client reports, host_suspend) and a client
disconnect being a normal close."""

from pathlib import Path

from sources import server_source, ui_source

from presencekit import diag

ROOT = Path(__file__).parents[1] / "presencekit"
HTML = ui_source()
CLIENT_ONLY = {"fps", "vad_k", "vad_min", "vad_hang"}  # browser-side send rate / VAD
SRV = server_source()


def test_diagnostic_events_present():
    for ev in (
        '"session_open"',
        '"track_new"',
        '"track_lost"',
        '"track_recovered"',
        '"track_dropped"',
        '"face_miss_summary"',
        '"asr_skip"',
        '"host_suspend"',
        '"client"',
        '"sender_closed"',
    ):
        assert ev in SRV, ev
    assert 'reason=tr_ev[-1].get("reason")' in SRV
    for what in ("'visibility'", "'autopause'", "'media_ended'", "'reconnect'", "'stale_socket'", "'client_suspend'"):
        assert what in HTML, what
    assert "performance.now()-lastRx>8000" in HTML  # half-open socket is closed and reconnected


def test_client_report_and_normal_close_over_ws():
    from fastapi.testclient import TestClient

    from presencekit import server

    q = f"?token={server.TOKEN}" if server.TOKEN else ""
    with TestClient(server.app).websocket_connect("/ws" + q) as ws:
        ws.send_json({"type": "client", "what": "visibility", "hidden": True, "level": "x", "big": "y" * 500})
        ws.send_json({"type": "ping", "t": 1})
        for _ in range(20):
            if ws.receive_json().get("type") == "pong":
                break
    ev = [e for e in diag.recent(200) if e["event"] == "client"]
    assert ev and ev[-1]["what"] == "visibility" and ev[-1]["level"] == "info" and len(ev[-1]["big"]) <= 64
    crashes = [e for e in diag.recent(50) if e["event"] == "task_crash" and e.get("task") == "sender"]
    assert not crashes

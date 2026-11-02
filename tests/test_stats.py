from presencekit.stats import ConnStats, _pct, system_stats


class Clk:
    t = 100.0

    def __call__(self):
        return self.t


def test_percentile():
    assert _pct([], 0.95) == 0.0
    assert _pct(list(range(1, 101)), 0.95) == 95


def test_conn_stats_fps_latency_bandwidth():
    c = Clk()
    s = ConnStats(clock=c)
    for i in range(10):
        c.t += 0.5
        s.on_received(1000)
        s.on_processed({"decode": 1, "detect": 10 + i, "embed": 5, "tracker": 0.1, "total": 20, "w": 640, "h": 480})
    s.dropped = 2
    snap = s.snapshot()
    assert snap["fps"] == 2.0 and snap["bandwidth_Bps"] == 2000
    assert snap["latency_ms"]["detect"]["avg"] == 14.5 and snap["latency_ms"]["detect"]["p95"] == 19
    assert snap["frames"] == {"received": 10, "processed": 10, "dropped": 2}
    assert snap["frame_size"] == [640, 480] and snap["rtt_ms"] is None


def test_system_stats_shape():
    st = system_stats()
    assert st["proc_rss"] > 0 and st["mem_total"] > 0 and isinstance(st["per_core"], list)


def test_audio_recent_weighting_eviction_and_capacity():
    from presencekit.stats import AudioStats

    clock = Clk()
    stats = AudioStats(clock=clock, limit=3)
    assert stats.snapshot()["rtf_recent"] is None
    stats.record(100, 0.01, 0.02, 0.03)
    stats.record(101, 0.01, 0.1, 0.04)
    clock.t = 101
    snap = stats.snapshot()
    assert snap["rtf_recent"] == 0.167  # weighted by duration, not mean([0.5, 0.1])
    assert snap["proc_p95_ms"] == 10
    assert snap["queue_wait_ms_avg"] == 35 and snap["queue_wait_ms_p95"] == 40
    clock.t = 105.1
    assert stats.snapshot()["rtf_recent"] == 0.1
    clock.t = 107
    assert stats.snapshot()["proc_p95_ms"] is None
    for _ in range(10):
        stats.record(107, 0.01, 0.02, 0)
    assert len(stats.samples) == 3


def test_added_vision_timings_are_bounded_and_preserved():
    stats = ConnStats(window=2)
    for n in (1, 2, 3):
        stats.on_processed({k: n for k in ("queue_wait", "executor_wait", "lock_wait", "appearance")})
    for key in ("queue_wait", "executor_wait", "lock_wait", "appearance"):
        assert stats.snapshot()["latency_ms"][key] == {"avg": 2.5, "p95": 3.0, "last": 3.0}

"""Statistical acceptance contracts, using numeric fake measurements only."""

import pytest

from presencekit import acceptance_pipeline as pipeline


def samples(seconds=10, rtf=0.3, queue=0, dropped=0, rss_growth_mib=0):
    return [
        {
            "t": second,
            "rtf_recent": rtf,
            "queue": queue,
            "audio_dropped": dropped,
            "rss_bytes": (300 + (rss_growth_mib if second > seconds - 600 else 0)) * (1 << 20),
            "audio_age_seconds": 0.1,
        }
        for second in range(1, seconds + 1)
    ]


def indexed(rows):
    return {row["id"].split(".", 1)[1]: row for row in rows}


def test_short_pipeline_passes_numeric_gates_and_retains_only_metrics():
    cases = pipeline.summarize_samples(samples(), suite="pipeline", seconds=10)
    assert all(case["status"] == "passed" for case in cases)
    assert cases[-1]["actual"]["samples"] == samples()
    assert all(case["details"]["qualification"] for case in cases)
    assert all("ASR none; re-ID off" in case["details"]["method"]["cloud"] for case in cases)


def test_short_soak_cannot_pass_30_minute_or_memory_gates():
    result = indexed(pipeline.summarize_samples(samples(), suite="soak", seconds=10))
    assert result["rtf"]["status"] == "passed"
    assert result["rss_growth"]["status"] == "not_run"
    assert result["30_minute_qualification"]["status"] == "not_run"
    assert not result["30_minute_qualification"]["details"]["qualification"]


@pytest.mark.parametrize("growth,expected", [(30, "passed"), (30.01, "failed")])
def test_real_30_minute_rss_boundary(growth, expected):
    result = indexed(pipeline.summarize_samples(samples(1800, rss_growth_mib=growth), suite="soak", seconds=1800))
    assert result["30_minute_qualification"]["status"] == "passed"
    assert result["rss_growth"]["status"] == expected
    assert result["rss_growth"]["actual"]["growth_mib"] == pytest.approx(growth)


@pytest.mark.parametrize("rtf,expected", [(0.5, "passed"), (0.501, "failed")])
def test_rtf_p95_boundary(rtf, expected):
    result = indexed(pipeline.summarize_samples(samples(rtf=rtf), suite="pipeline", seconds=10))
    assert result["rtf"]["status"] == expected


def test_drops_are_measured_relative_to_warmup_but_never_ignored():
    rows = samples(dropped=2)
    result = indexed(pipeline.summarize_samples(rows, suite="pipeline", seconds=10, dropped_baseline=2))
    assert result["audio_dropped"]["status"] == "passed"
    rows[-1]["audio_dropped"] = 3
    result = indexed(pipeline.summarize_samples(rows, suite="pipeline", seconds=10, dropped_baseline=2))
    assert result["audio_dropped"]["status"] == "failed"


def test_sender_audio_drop_also_fails_gate():
    rows = samples()
    rows[-1]["send_dropped"] = 1
    result = indexed(pipeline.summarize_samples(rows, suite="pipeline", seconds=10))
    assert result["audio_dropped"]["status"] == "failed"
    assert result["audio_dropped"]["actual"]["send_dropped"] == 1


def test_sample_export_never_copies_media_or_arbitrary_fields():
    rows = samples()
    rows[0]["media"] = "must-not-be-exported"
    result = indexed(pipeline.summarize_samples(rows, suite="pipeline", seconds=10))
    assert "media" not in result["measurements"]["actual"]["samples"][0]


def test_queue_growth_and_capacity_fail_even_without_drops():
    rows = samples()
    for row in rows[-3:]:
        row["queue"] = 2
    result = indexed(pipeline.summarize_samples(rows, suite="pipeline", seconds=10))
    assert result["queue"]["status"] == "failed"
    rows = samples()
    rows[4]["queue"] = 25
    result = indexed(pipeline.summarize_samples(rows, suite="pipeline", seconds=10))
    assert result["queue"]["status"] == "failed"


def test_any_error_fails_gate():
    result = indexed(pipeline.summarize_samples(samples(), suite="pipeline", seconds=10, error_count=1))
    assert result["errors"]["status"] == "failed"


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf")])
def test_missing_or_invalid_metrics_cannot_pass(bad_value):
    rows = samples()
    rows[0]["rtf_recent"] = bad_value
    result = indexed(pipeline.summarize_samples(rows, suite="pipeline", seconds=10))
    assert result["measurement_coverage"]["status"] == "failed"
    assert result["performance_gates"]["status"] == "not_run"


def test_stale_audio_and_incomplete_sampling_cannot_establish_stability():
    rows = samples(1800)
    for row in rows[-100:]:
        row["audio_age_seconds"] = 2
    result = indexed(pipeline.summarize_samples(rows, suite="soak", seconds=1800))
    assert result["measurement_coverage"]["status"] == "failed"
    assert result["performance_gates"]["status"] == "not_run"


def test_missing_models_are_not_run_without_starting_server(monkeypatch):
    monkeypatch.setattr(pipeline, "_preflight", lambda scene: "Local model missing; no download attempted.")

    def forbidden(*args, **kwargs):
        pytest.fail("No server or network should be started with unavailable prerequisites")

    monkeypatch.setattr(pipeline, "_isolated_run", forbidden)
    cases = pipeline.run_pipeline()
    assert cases[0]["status"] == "not_run"
    assert "no download" in cases[0]["details"]["reason"]


def test_child_environment_excludes_cloud_credentials(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    monkeypatch.setenv("PRESENCE_TOKEN", "private")
    monkeypatch.setenv("PRESENCE_LOG_FILE", "/must-not-write")
    env = pipeline._child_environment()
    assert "TYPESAFE_API_KEY" not in env
    assert "PRESENCE_TOKEN" not in env
    assert "PRESENCE_LOG_FILE" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.parametrize("seconds", [0, 2, 3601, float("nan"), float("inf")])
def test_invalid_duration_is_rejected_before_model_checks(seconds, monkeypatch):
    def forbidden(scene):
        pytest.fail("Invalid input must fail before accessing model paths")

    monkeypatch.setattr(pipeline, "_preflight", forbidden)
    with pytest.raises(ValueError, match="seconds"):
        pipeline.run_soak(seconds=seconds)

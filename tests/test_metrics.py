import itertools

import pytest

from benchmark import metrics


def test_percentile_returns_an_observed_value():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    assert metrics.percentile(values, 50) == 5.0
    assert metrics.percentile(values, 95) == 10.0
    assert metrics.percentile(values, 100) == 10.0


def test_percentile_ignores_input_order():
    assert metrics.percentile([9.0, 1.0, 5.0], 50) == 5.0


def test_percentile_of_single_sample():
    assert metrics.percentile([4.2], 95) == 4.2


@pytest.mark.parametrize("pct", [0, -1, 101])
def test_percentile_rejects_out_of_range_percentages(pct):
    with pytest.raises(metrics.MetricsError):
        metrics.percentile([1.0], pct)


def test_percentile_rejects_empty_sample():
    with pytest.raises(metrics.MetricsError, match="empty sample"):
        metrics.percentile([], 50)


def test_measure_records_one_latency_per_successful_iteration():
    calls: list[int] = []

    measurement = metrics.measure("noop", calls.append, iterations=5, warmup=2)

    assert calls == [0, 1, 0, 1, 2, 3, 4]  # warm-up runs first, then the measured pass
    assert measurement.attempted == 5
    assert measurement.successes == 5
    assert measurement.failure_count == 0
    assert measurement.warmup_iterations == 2


def test_measure_counts_failures_without_dropping_them():
    def flaky(iteration: int) -> None:
        if iteration % 2 == 0:
            raise RuntimeError("boom")

    measurement = metrics.measure("flaky", flaky, iterations=4)

    assert measurement.successes == 2
    assert measurement.failure_count == 2
    assert measurement.failure_samples[0].iteration == 0
    assert measurement.failure_samples[0].error == "RuntimeError: boom"


def test_measure_caps_stored_failure_samples_but_not_the_count():
    def always_fails(_: int) -> None:
        raise ValueError("nope")

    iterations = metrics.MAX_FAILURE_SAMPLES + 5
    measurement = metrics.measure("failing", always_fails, iterations=iterations)

    assert measurement.failure_count == iterations
    assert len(measurement.failure_samples) == metrics.MAX_FAILURE_SAMPLES
    assert measurement.to_dict()["failures"] == iterations


def test_measure_survives_failing_warmup_iterations():
    calls = itertools.count()

    def fails_on_the_first_two_calls(_: int) -> None:
        if next(calls) < 2:
            raise RuntimeError("cold")

    measurement = metrics.measure("warmup", fails_on_the_first_two_calls, iterations=3, warmup=2)

    assert measurement.successes == 3
    assert measurement.failure_count == 0


def test_measure_rejects_zero_iterations():
    with pytest.raises(metrics.MetricsError):
        metrics.measure("noop", lambda _: None, iterations=0)


def test_measurement_dict_omits_percentiles_when_everything_failed():
    def always_fails(_: int) -> None:
        raise RuntimeError("boom")

    summary = metrics.measure("failing", always_fails, iterations=3).to_dict()

    assert summary["successes"] == 0
    assert summary["failures"] == 3
    assert "p50_ms" not in summary


def test_measurement_dict_reports_percentiles_and_throughput():
    measurement = metrics.Measurement(
        name="lookup",
        attempted=4,
        warmup_iterations=1,
        latencies_ms=(1.0, 2.0, 3.0, 4.0),
        wall_time_s=2.0,
        concurrency=10,
    )

    summary = measurement.to_dict()

    assert summary["workload"] == "lookup"
    assert summary["p50_ms"] == 2.0
    assert summary["p95_ms"] == 4.0
    assert summary["mean_ms"] == 2.5
    assert summary["ops_per_second"] == 2.0  # 4 successful operations in 2 seconds


def test_stopwatch_measures_elapsed_time():
    with metrics.stopwatch() as watch:
        sum(range(10_000))

    assert watch.elapsed_s > 0


def test_throughput_is_zero_when_untimed():
    assert metrics.throughput(100, 0) == 0.0
    assert metrics.throughput(100, 2.0) == 50.0


def test_describe_error_is_single_line_and_bounded():
    text = metrics.describe_error(RuntimeError("a\nb" + "x" * 500))

    assert "\n" not in text
    assert len(text) <= metrics.MAX_ERROR_CHARS

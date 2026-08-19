import dataclasses

import pytest

from benchmark import metrics, mixed
from benchmark.adapters.base import AdapterError
from benchmark.dataset import Dataset
from tests.fakes import SETTINGS, FakeAdapter

DATA = Dataset(node_ids=(1, 2, 3, 4, 5), edges=((1, 2), (2, 3), (3, 4), (4, 5)))
SWEEP_SETTINGS = dataclasses.replace(
    SETTINGS, measured_iterations=10, warmup_iterations=2, concurrency_levels=(1, 3)
)


def test_operations_honour_the_configured_read_write_ratio():
    operations = mixed.build_operations(DATA, SWEEP_SETTINGS, client_index=0, count=10)

    assert len(operations) == 10
    assert sum(1 for operation in operations if operation.is_write) == 1  # 10% of 10


def test_ratio_is_exact_at_a_different_split():
    settings = dataclasses.replace(SWEEP_SETTINGS, read_ratio=0.5)

    operations = mixed.build_operations(DATA, settings, client_index=0, count=20)

    assert sum(1 for operation in operations if operation.is_write) == 10


def test_operations_are_deterministic_for_a_client():
    first = mixed.build_operations(DATA, SWEEP_SETTINGS, client_index=2, count=10)
    second = mixed.build_operations(DATA, SWEEP_SETTINGS, client_index=2, count=10)

    assert first == second


def test_clients_do_not_all_run_the_same_sequence():
    first = mixed.build_operations(DATA, SWEEP_SETTINGS, client_index=0, count=10)
    second = mixed.build_operations(DATA, SWEEP_SETTINGS, client_index=1, count=10)

    assert first != second


def test_reads_start_from_nodes_that_have_outgoing_edges():
    operations = mixed.build_operations(DATA, SWEEP_SETTINGS, client_index=0, count=10)

    sources = {source for source, _ in DATA.edges}
    assert all(operation.first in sources for operation in operations if not operation.is_write)


def test_sweep_runs_one_measurement_per_concurrency_level():
    adapters_used: list[FakeAdapter] = []

    def factory() -> FakeAdapter:
        adapter = FakeAdapter()
        adapters_used.append(adapter)
        return adapter

    measurements = mixed.run_sweep(SWEEP_SETTINGS, DATA, factory)

    assert [measurement.concurrency for measurement in measurements] == [1, 3]
    assert len(adapters_used) == 4  # one client at level 1, three at level 3


def test_each_client_gets_its_own_connection():
    adapters_used: list[FakeAdapter] = []

    def factory() -> FakeAdapter:
        adapter = FakeAdapter()
        adapters_used.append(adapter)
        return adapter

    mixed.run_level(3, SWEEP_SETTINGS, DATA, factory)

    assert len(adapters_used) == 3
    for adapter in adapters_used:
        assert adapter.calls[0] == "connect"
        assert adapter.calls[-1] == "close"
        assert adapter.connected is False


def test_level_reports_reads_and_writes_actually_performed():
    adapters_used: list[FakeAdapter] = []

    def factory() -> FakeAdapter:
        adapter = FakeAdapter()
        adapters_used.append(adapter)
        return adapter

    measurement = mixed.run_level(2, SWEEP_SETTINGS, DATA, factory)

    writes = sum(len(adapter.writes) for adapter in adapters_used)
    reads = sum(adapter.calls.count("mixed_read") for adapter in adapters_used)
    assert writes > 0
    assert reads > writes
    assert measurement.attempted == 2 * SWEEP_SETTINGS.measured_iterations
    assert measurement.successes == measurement.attempted


def test_throughput_is_total_operations_over_the_parallel_window():
    measurement = mixed.run_level(2, SWEEP_SETTINGS, DATA, FakeAdapter)

    expected = measurement.successes / measurement.wall_time_s
    assert measurement.ops_per_second == pytest.approx(expected)


def test_a_refused_level_is_recorded_rather_than_discarding_the_run():
    def broken_factory() -> FakeAdapter:
        adapter = FakeAdapter()
        adapter.fail_on = {"connect"}
        return adapter

    measurement = mixed.run_level(2, SWEEP_SETTINGS, DATA, broken_factory)

    assert measurement.concurrency == 2
    assert measurement.successes == 0
    assert measurement.failure_count == 2 * SWEEP_SETTINGS.measured_iterations
    assert measurement.ops_per_second == 0.0
    assert "refused" in measurement.notes[0]
    assert "p50_ms" not in measurement.to_dict()


def test_a_sweep_continues_past_a_refused_level():
    calls: list[int] = []

    def factory() -> FakeAdapter:
        adapter = FakeAdapter()
        # Refuse only when the third client of the level-3 run appears.
        calls.append(1)
        if len(calls) > 1:
            adapter.fail_on = {"connect"}
        return adapter

    measurements = mixed.run_sweep(SWEEP_SETTINGS, DATA, factory)

    assert [measurement.concurrency for measurement in measurements] == [1, 3]
    assert measurements[0].successes > 0


def test_combine_totals_attempts_and_failures_without_summing_client_time():
    parts = [
        metrics.Measurement(
            name="client 0", attempted=5, warmup_iterations=2, latencies_ms=(1.0, 2.0), wall_time_s=9
        ),
        metrics.Measurement(
            name="client 1",
            attempted=5,
            warmup_iterations=2,
            latencies_ms=(3.0,),
            failure_count=4,
            wall_time_s=9,
        ),
    ]

    combined = metrics.combine("mixed", parts, concurrency=2, wall_time_s=3.0)

    assert combined.attempted == 10
    assert combined.successes == 3
    assert combined.failure_count == 4
    assert combined.wall_time_s == 3.0
    assert combined.ops_per_second == 1.0  # 3 successes over the 3 s window

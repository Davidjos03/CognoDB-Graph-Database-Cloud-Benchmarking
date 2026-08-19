import dataclasses

import pytest

from benchmark import adapters, runner
from benchmark.adapters.base import AdapterError
from benchmark.dataset import Dataset
from tests.fakes import SETTINGS, TARGET, FakeAdapter

DATA = Dataset(node_ids=(1, 2, 3, 4, 5), edges=((1, 2), (2, 3), (3, 4), (4, 5)))
DATASET_METADATA = {"name": "test", "node_count": 5, "relationship_count": 4}


@pytest.fixture
def fake_registry(monkeypatch):
    """Make the runner build our in-memory adapter instead of a real driver."""
    adapter = FakeAdapter(TARGET, SETTINGS)
    monkeypatch.setattr(adapters, "create", lambda target, settings: adapter)
    monkeypatch.setattr(runner.adapters, "create", lambda target, settings: adapter)
    return adapter


def run(**kwargs):
    return runner.run_platform(TARGET, SETTINGS, DATA, DATASET_METADATA, **kwargs)


def test_run_measures_every_workload(fake_registry):
    result = run()

    names = [workload["workload"] for workload in result.to_dict()["workloads"]]
    assert names == [
        "network baseline",
        "1-hop traversal",
        "2-hop traversal",
        "3-hop traversal",
        "point lookup",
        "filtered lookup",
        "aggregation",
    ]


def test_run_loads_before_measuring(fake_registry):
    run()

    calls = fake_registry.calls
    assert calls.index("load_relationships") < calls.index("point_lookup")


def test_run_sweeps_the_mixed_workload_and_cleans_up_its_writes(fake_registry):
    payload = run().to_dict()

    assert [level["concurrency"] for level in payload["mixed"]] == list(
        SETTINGS.concurrency_levels
    )
    assert all(level["workload"] == "mixed read/write" for level in payload["mixed"])
    assert "cleanup_writes" in fake_registry.calls
    assert fake_registry.writes == []  # nothing left behind in the graph


def test_run_records_ingest_and_footprint(fake_registry):
    payload = run().to_dict()

    assert payload["ingest"]["relationship_count"] == 4
    assert payload["ingest"]["relationships_per_second"] > 0
    assert payload["footprint"]["loaded_nodes"] == 5


def test_run_reports_the_configured_iteration_counts(fake_registry):
    payload = run().to_dict()

    for workload in payload["workloads"]:
        assert workload["attempted"] == SETTINGS.measured_iterations
        assert workload["warmup_iterations"] == SETTINGS.warmup_iterations
        assert workload["successes"] == SETTINGS.measured_iterations


def test_skipping_the_load_is_recorded_as_a_caveat(fake_registry):
    payload = run(load=False).to_dict()

    assert payload["ingest"] is None
    assert any("load phase skipped" in caveat for caveat in payload["caveats"])
    assert "reset" not in fake_registry.calls


def test_adapter_caveats_travel_into_the_result(fake_registry):
    fake_registry.drop_every_nth_node = 2

    payload = run().to_dict()

    assert any("nodes" in caveat for caveat in payload["caveats"])


def test_the_connection_is_always_closed(fake_registry):
    run()

    assert fake_registry.calls[0] == "connect"
    assert fake_registry.calls[-1] == "close"
    assert fake_registry.connected is False


def test_a_failing_workload_is_measured_as_failures_not_an_abort(fake_registry):
    fake_registry.fail_on = {"aggregation"}

    payload = run().to_dict()

    aggregation = payload["workloads"][-1]
    assert aggregation["workload"] == "aggregation"
    assert aggregation["successes"] == 0
    assert aggregation["failures"] == SETTINGS.measured_iterations
    assert "p50_ms" not in aggregation


def test_a_failing_connection_aborts_the_run(fake_registry):
    fake_registry.fail_on = {"connect"}

    with pytest.raises(AdapterError):
        run()


def test_iteration_counts_come_from_settings(fake_registry, monkeypatch):
    settings = dataclasses.replace(SETTINGS, measured_iterations=7, warmup_iterations=2)

    result = runner.run_platform(TARGET, settings, DATA, DATASET_METADATA)

    assert result.to_dict()["workloads"][0]["attempted"] == 7

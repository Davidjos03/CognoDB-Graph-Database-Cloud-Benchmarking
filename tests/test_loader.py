import pytest

from benchmark import loader
from benchmark.adapters.base import AdapterError
from benchmark.dataset import Dataset
from tests.fakes import FakeAdapter

DATA = Dataset(node_ids=(1, 2, 3, 4), edges=((1, 2), (2, 3), (3, 4)))


def test_load_runs_the_phases_in_a_fair_order():
    adapter = FakeAdapter()

    loader.load_into(adapter, DATA)

    assert adapter.calls[:4] == ["reset", "create_indexes", "load_nodes", "load_relationships"]


def test_load_indexes_before_loading_relationships():
    adapter = FakeAdapter()

    loader.load_into(adapter, DATA)

    assert adapter.calls.index("create_indexes") < adapter.calls.index("load_relationships")


def test_load_reports_counts_and_separate_phase_timings():
    adapter = FakeAdapter()

    report = loader.load_into(adapter, DATA)

    assert report.node_count == 4
    assert report.relationship_count == 3
    assert report.node_seconds > 0
    assert report.relationship_seconds > 0
    assert report.total_seconds == report.node_seconds + report.relationship_seconds


def test_load_writes_the_whole_graph():
    adapter = FakeAdapter()

    loader.load_into(adapter, DATA)

    assert adapter.nodes == {1: 1, 2: 2, 3: 3, 4: 4}
    assert adapter.edges == list(DATA.edges)


def test_ingest_rates_are_derived_from_the_measured_phases():
    report = loader.LoadReport(
        node_count=100, relationship_count=1000, node_seconds=2.0, relationship_seconds=4.0
    )

    assert report.nodes_per_second == 50.0
    assert report.relationships_per_second == 250.0
    assert report.total_seconds == 6.0


def test_a_platform_that_drops_rows_is_caveated_not_ignored():
    adapter = FakeAdapter()
    adapter.drop_every_nth_node = 2  # silently loses half the nodes

    report = loader.load_into(adapter, DATA)

    assert report.node_count == 2
    assert any("nodes" in caveat for caveat in adapter.caveats)


def test_load_surfaces_adapter_failures_instead_of_reporting_a_fast_load():
    adapter = FakeAdapter()
    adapter.fail_on = {"load_relationships"}

    with pytest.raises(AdapterError, match="load_relationships failed"):
        loader.load_into(adapter, DATA)

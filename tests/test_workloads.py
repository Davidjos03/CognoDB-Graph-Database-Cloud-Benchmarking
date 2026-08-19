import dataclasses

import pytest

from benchmark import dataset, workloads
from tests.fakes import SETTINGS, FakeAdapter

DATA = dataset.Dataset(node_ids=(1, 2, 3, 4, 5), edges=((1, 2), (2, 3), (3, 4), (4, 5)))


def build(settings=SETTINGS):
    adapter = FakeAdapter()
    return adapter, workloads.build_read_workloads(adapter, DATA, settings)


def test_every_required_read_workload_is_present():
    _, built = build()

    assert [workload.name for workload in built] == [
        "network baseline",
        "1-hop traversal",
        "2-hop traversal",
        "3-hop traversal",
        "point lookup",
        "filtered lookup",
        "aggregation",
    ]


def test_each_workload_calls_its_own_adapter_operation():
    adapter, built = build()

    for workload in built:
        workload.operation(0)

    assert adapter.calls == [
        "ping",
        "traversal:1",
        "traversal:2",
        "traversal:3",
        "point_lookup",
        "filtered_lookup",
        "aggregation",
    ]


def test_parameter_streams_are_independent():
    settings = dataclasses.replace(SETTINGS, measured_iterations=5, warmup_iterations=0)
    count = workloads.parameter_count(settings)

    start_nodes = dataset.sample_start_nodes(DATA, count, seed=settings.seed)
    lookup_nodes = dataset.sample_node_ids(DATA, count, seed=settings.seed + 1)

    assert start_nodes != lookup_nodes


def test_parameters_are_identical_across_runs_with_the_same_seed():
    first_adapter, first = build()
    second_adapter, second = build()

    for workload in first:
        workload.operation(1)
    for workload in second:
        workload.operation(1)

    assert first_adapter.calls == second_adapter.calls


def test_parameter_count_covers_the_longer_of_warmup_and_measured():
    assert workloads.parameter_count(dataclasses.replace(SETTINGS, measured_iterations=100, warmup_iterations=20)) == 100
    assert workloads.parameter_count(dataclasses.replace(SETTINGS, measured_iterations=5, warmup_iterations=50)) == 50


def test_every_iteration_of_the_warmup_and_measured_pass_has_a_parameter():
    settings = dataclasses.replace(SETTINGS, measured_iterations=4, warmup_iterations=6)
    adapter, built = build(settings)

    for workload in built:
        for iteration in range(max(settings.measured_iterations, settings.warmup_iterations)):
            workload.operation(iteration)  # must not raise IndexError


def test_group_ids_stay_inside_the_dataset_range():
    sampled = dataset.sample_group_ids(200, seed=1)

    assert len(sampled) == 200
    assert set(sampled) <= set(range(dataset.GROUP_COUNT))


def test_sampling_rejects_an_empty_request():
    with pytest.raises(dataset.DatasetError):
        dataset.sample_node_ids(DATA, 0, seed=1)

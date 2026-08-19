from pathlib import Path

import pytest

from benchmark import adapters
from benchmark.adapters import base, bolt
from benchmark.config import Settings, Target

SETTINGS = Settings(
    seed=42,
    warmup_iterations=1,
    measured_iterations=1,
    concurrency_levels=(1,),
    read_ratio=0.9,
    batch_size=2,
    query_timeout_s=5.0,
    data_dir=Path("data"),
    results_dir=Path("results"),
)


def target(name: str, kind: str) -> Target:
    return Target(
        name=name,
        kind=kind,
        uri="bolt+s://host.example.com",
        username="user",
        password="secret",
        database="neo4j",
        specs="entry tier",
    )


def test_registry_builds_a_bolt_adapter():
    adapter = adapters.create(target("cognodb", "bolt"), SETTINGS)

    assert isinstance(adapter, bolt.BoltAdapter)
    assert adapter.name == "cognodb"


def test_registry_reports_platforms_without_an_adapter():
    with pytest.raises(adapters.AdapterError, match="no adapter implemented"):
        adapters.create(target("falkordb", "falkordb"), SETTINGS)


def test_caveats_are_recorded_once_and_readable():
    adapter = adapters.create(target("cognodb", "bolt"), SETTINGS)

    adapter.add_caveat("index unsupported")
    adapter.add_caveat("index unsupported")

    assert adapter.caveats == ["index unsupported"]


def test_using_an_unconnected_adapter_is_an_adapter_error():
    adapter = adapters.create(target("cognodb", "bolt"), SETTINGS)

    with pytest.raises(adapters.AdapterError, match="not connected"):
        adapter.point_lookup(1)


def test_unsupported_traversal_depth_is_rejected():
    adapter = adapters.create(target("cognodb", "bolt"), SETTINGS)

    with pytest.raises(adapters.AdapterError, match="depth 4"):
        adapter.traversal(4, start_node=1)


@pytest.mark.parametrize(
    "query, placeholder",
    [
        (bolt.POINT_LOOKUP, "$node_id"),
        (bolt.FILTERED_LOOKUP, "$group_id"),
        (bolt.TRAVERSALS[1], "$node_id"),
        (bolt.TRAVERSALS[2], "$node_id"),
        (bolt.TRAVERSALS[3], "$node_id"),
        (bolt.MIXED_WRITE, "$source"),
        (bolt.LOAD_NODES, "$rows"),
        (bolt.LOAD_RELATIONSHIPS, "$rows"),
    ],
)
def test_queries_pass_values_as_parameters(query, placeholder):
    assert placeholder in query


def test_traversal_depth_is_expressed_as_that_many_hops():
    for depth in (1, 2, 3):
        assert bolt.TRAVERSALS[depth].count(f"[:{bolt.RELATIONSHIP_TYPE}]") == depth


def test_mixed_writes_use_a_separate_relationship_type():
    assert bolt.WRITE_RELATIONSHIP not in bolt.LOAD_RELATIONSHIPS
    assert bolt.WRITE_RELATIONSHIP in bolt.MIXED_WRITE
    assert bolt.WRITE_RELATIONSHIP in bolt.DELETE_WRITES


def test_indexes_cover_both_queried_properties():
    statements = " ".join(bolt.CREATE_INDEXES)

    assert "node_id" in statements
    assert "group_id" in statements


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ("group_id", ["group_id"]),  # CognoDB reports a single property string
        (["node_id"], ["node_id"]),  # Neo4j reports a list
        (["a", "b"], ["a", "b"]),
        (7, ["7"]),
    ],
)
def test_index_fields_are_read_from_either_server_shape(value, expected):
    assert bolt._as_list(value) == expected


def test_batched_splits_without_losing_items():
    items = [(index, index) for index in range(5)]

    assert list(base.batched(items, 2)) == [
        [(0, 0), (1, 1)],
        [(2, 2), (3, 3)],
        [(4, 4)],
    ]


def test_batched_of_nothing_yields_nothing():
    assert list(base.batched([], 3)) == []

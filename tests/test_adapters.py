from pathlib import Path

import pytest

from benchmark import adapters, config
from benchmark.adapters import arangodb, base, bolt, falkordb, memgraph
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


def test_registry_reports_a_kind_without_an_adapter():
    with pytest.raises(adapters.AdapterError, match="no adapter implemented"):
        adapters.create(target("neptune", "gremlin"), SETTINGS)


@pytest.mark.parametrize("platform, kind", sorted(config.PLATFORMS.items()))
def test_every_configured_platform_has_an_adapter(platform, kind):
    adapter = adapters.create(target(platform, kind), SETTINGS)

    assert adapter.name == platform


@pytest.mark.parametrize("adapter_class", sorted(adapters.ADAPTERS.values(), key=str))
def test_every_adapter_implements_the_whole_interface(adapter_class):
    missing = getattr(adapter_class, "__abstractmethods__", frozenset())

    assert not missing, f"{adapter_class.__name__} does not implement {sorted(missing)}"


def test_memgraph_uses_its_own_index_syntax():
    # The Neo4j 5 form is a syntax error on Memgraph, and vice versa.
    assert all("IF NOT EXISTS" not in statement for statement in memgraph.CREATE_INDEXES)
    assert all("IF NOT EXISTS" in statement for statement in bolt.CREATE_INDEXES)


def test_cypher_platforms_share_the_same_read_statements():
    # Same text, not merely the same intent, wherever the language allows it.
    assert falkordb.POINT_LOOKUP is bolt.POINT_LOOKUP
    assert falkordb.TRAVERSALS is bolt.TRAVERSALS
    assert falkordb.AGGREGATION is bolt.AGGREGATION


def test_arangodb_translations_are_parameterised():
    for query, placeholder in (
        (arangodb.TRAVERSAL, "@depth"),
        (arangodb.POINT_LOOKUP, "@node_id"),
        (arangodb.FILTERED_LOOKUP, "@group_id"),
        (arangodb.MIXED_WRITE, "@source"),
        (arangodb.LOAD_NODES, "@rows"),
    ):
        assert placeholder in query


def test_arangodb_writes_go_to_a_separate_edge_collection():
    assert arangodb.WRITE_EDGES != arangodb.EDGES
    assert arangodb.WRITE_EDGES in arangodb.MIXED_WRITE
    assert arangodb.EDGES not in arangodb.MIXED_WRITE


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

"""Exercise the FalkorDB and ArangoDB adapters against stub drivers.

Neither platform can be reached from the test suite, but the parts most likely
to be wrong are pure translation: which statement is sent, how parameters are
bound, and how a driver's row shape is read back. Stub drivers cover exactly
that, so a live run fails on the network rather than on a typo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.adapters import arangodb, falkordb
from benchmark.adapters.base import AdapterError
from benchmark.config import Settings, Target

SETTINGS = Settings(
    seed=42,
    warmup_iterations=1,
    measured_iterations=2,
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
        uri=f"stub://{name}",
        username="user",
        password="secret",
        database="benchmark",
        specs="capped 0.5 vCPU / 512 MB",
    )


class StubResult:
    def __init__(self, rows: list[list]) -> None:
        self.result_set = rows


class StubGraph:
    """Records every statement and answers with a queued row set."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []
        self.timeouts: list[int | None] = []
        self.answers: list[list[list]] = []
        self.deleted = 0

    def query(
        self, statement: str, params: dict | None = None, timeout: int | None = None
    ) -> StubResult:
        self.calls.append((statement, params))
        self.timeouts.append(timeout)
        rows = self.answers.pop(0) if self.answers else []
        return StubResult(rows)

    def delete(self) -> None:
        self.deleted += 1

    @property
    def statements(self) -> list[str]:
        return [statement for statement, _ in self.calls]


class StubConnection:
    def __init__(self) -> None:
        self.closed = False

    def info(self, section: str) -> dict:
        if section == "server":
            return {"redis_version": "7.2.4"}
        return {"used_memory_human": "37.4M"}

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def falkor(monkeypatch):
    """A connected FalkorDB adapter whose driver is a stub."""
    graph = StubGraph()
    connection = StubConnection()

    class StubClient:
        def __init__(self) -> None:
            self.connection = connection

        def select_graph(self, name: str) -> StubGraph:
            return graph

    monkeypatch.setattr(falkordb.FalkorDB, "from_url", staticmethod(lambda uri: StubClient()))
    adapter = falkordb.FalkorDBAdapter(target("falkordb", "falkordb"), SETTINGS)
    adapter.connect()
    graph.calls.clear()  # drop the connection ping
    graph.timeouts.clear()
    return adapter, graph


def test_falkordb_sends_the_shared_cypher_with_bound_parameters(falkor):
    adapter, graph = falkor
    graph.answers = [[[7]]]

    assert adapter.traversal(1, 42) == 7
    statement, params = graph.calls[0]
    assert statement == falkordb.TRAVERSALS[1]
    assert params == {"node_id": 42}


def test_falkordb_reads_positional_rows_rather_than_named_records(falkor):
    adapter, graph = falkor
    graph.answers = [[[137]]]

    assert adapter.filtered_lookup(3) == 137


def test_falkordb_treats_an_empty_row_set_as_zero(falkor):
    adapter, graph = falkor
    graph.answers = [[]]

    assert adapter.filtered_lookup(3) == 0


def test_falkordb_loads_in_batches_of_the_configured_size(falkor):
    adapter, graph = falkor

    loaded = adapter.load_relationships([(1, 2), (3, 4), (5, 6)])

    assert loaded == 3
    # batch_size is 2, so three edges are two round trips, not three.
    assert graph.statements == [falkordb.LOAD_RELATIONSHIPS] * 2
    assert graph.calls[0][1] == {"rows": [{"source": 1, "target": 2}, {"source": 3, "target": 4}]}


def test_falkordb_writes_only_the_benchmark_relationship_type(falkor):
    adapter, graph = falkor

    adapter.mixed_write(1, 2)

    assert "BENCH_WRITE" in graph.statements[0]
    assert "VOTED_FOR" not in graph.statements[0]


def test_falkordb_cleanup_deletes_until_nothing_is_left(falkor):
    adapter, graph = falkor
    graph.answers = [[[2]], [[1]], [[0]]]

    assert adapter.cleanup_writes() == 3


def test_falkordb_reset_drops_the_graph_instead_of_deleting_rows(falkor):
    adapter, graph = falkor

    adapter.reset_test_data()

    assert graph.deleted == 1


def test_falkordb_reports_a_real_memory_footprint(falkor):
    adapter, graph = falkor
    graph.answers = [[["User", "node_id"]], [[7115]], [[103689]]]

    footprint = adapter.footprint()

    assert footprint["stored_data_size"] == "37.4M"
    assert footprint["server"] == "FalkorDB/redis 7.2.4"
    assert footprint["indexes"] == ["User(node_id)"]
    assert footprint["loaded_nodes"] == 7115


def test_falkordb_applies_the_configured_timeout_in_milliseconds(falkor):
    adapter, graph = falkor

    adapter.ping()

    assert graph.timeouts == [int(SETTINGS.query_timeout_s * 1000)]


def test_falkordb_refuses_to_query_before_connecting():
    adapter = falkordb.FalkorDBAdapter(target("falkordb", "falkordb"), SETTINGS)

    with pytest.raises(AdapterError, match="not connected"):
        adapter.ping()


def test_falkordb_wraps_driver_errors_in_an_adapter_error(falkor):
    adapter, graph = falkor

    def explode(statement, params=None):
        raise RuntimeError("connection reset")

    graph.query = explode

    with pytest.raises(AdapterError, match="query failed"):
        adapter.ping()


class StubCursor(list):
    pass


class StubAql:
    def __init__(self, owner: "StubDatabase") -> None:
        self._owner = owner

    def execute(
        self, query: str, bind_vars: dict | None = None, max_runtime: float | None = None
    ) -> StubCursor:
        self._owner.calls.append((query, bind_vars))
        self._owner.runtimes.append(max_runtime)
        return StubCursor(self._owner.answers.pop(0) if self._owner.answers else [])


class StubCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.truncated = 0
        self.added_indexes: list[dict] = []
        self.documents: dict[str, dict] = {}

    def truncate(self) -> None:
        self.truncated += 1

    def add_index(self, definition: dict) -> None:
        self.added_indexes.append(definition)

    def indexes(self) -> list[dict]:
        return [{"fields": ["_key"]}, {"fields": ["group_id"]}]

    def statistics(self) -> dict:
        return {"documents_size": 4194304}

    def count(self) -> int:
        return len(self.documents)


class StubDatabase:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, dict | None]] = []
        self.runtimes: list[float | None] = []
        self.answers: list[list] = []
        self.collections: dict[str, StubCollection] = {}
        self.created_databases: list[str] = []
        self.aql = StubAql(self)

    def version(self) -> str:
        return "3.12.4"

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, edge: bool = False) -> StubCollection:
        self.collections[name] = StubCollection(name)
        return self.collections[name]

    def collection(self, name: str) -> StubCollection:
        return self.collections.setdefault(name, StubCollection(name))

    def has_database(self, name: str) -> bool:
        return name in self.created_databases

    def create_database(self, name: str) -> None:
        self.created_databases.append(name)

    @property
    def queries(self) -> list[str]:
        return [query for query, _ in self.calls]


@pytest.fixture
def arango(monkeypatch):
    """A connected ArangoDB adapter whose driver is a stub."""
    database = StubDatabase("benchmark")

    class StubArangoClient:
        def __init__(self, hosts: str) -> None:
            self.hosts = hosts
            self.closed = False

        def db(self, name: str, username: str, password: str, verify: bool) -> StubDatabase:
            return database

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(arangodb, "ArangoClient", StubArangoClient)
    adapter = arangodb.ArangoDBAdapter(target("arangodb", "arangodb"), SETTINGS)
    adapter.connect()
    database.calls.clear()
    return adapter, database


def test_arangodb_creates_its_database_and_collections_on_connect(arango):
    _, database = arango

    assert database.created_databases == ["benchmark"]
    assert set(database.collections) == {arangodb.NODES, arangodb.EDGES, arangodb.WRITE_EDGES}


def test_arangodb_records_the_translation_as_a_caveat(arango):
    adapter, _ = arango

    assert any("AQL translations" in caveat for caveat in adapter.caveats)


def test_arangodb_binds_a_document_handle_for_the_traversal_start(arango):
    adapter, database = arango
    database.answers = [[11]]

    assert adapter.traversal(2, 42) == 11
    query, bind_vars = database.calls[0]
    assert query == arangodb.TRAVERSAL
    assert bind_vars == {"depth": 2, "start": "users/42"}


def test_arangodb_rejects_an_unsupported_traversal_depth(arango):
    adapter, _ = arango

    with pytest.raises(AdapterError, match="unsupported traversal depth"):
        adapter.traversal(4, 42)


def test_arangodb_point_lookup_reports_a_miss_as_zero(arango):
    adapter, database = arango
    database.answers = [[None]]

    assert adapter.point_lookup(999) == 0


def test_arangodb_uses_node_id_as_the_document_key(arango):
    adapter, database = arango

    adapter.load_nodes([(3, 3), (28, 28)])

    _, bind_vars = database.calls[0]
    assert bind_vars["rows"][0] == {"_key": "3", "node_id": 3, "group_id": 3}


def test_arangodb_indexes_group_id_because_the_key_is_already_indexed(arango):
    adapter, database = arango

    adapter.create_indexes()

    assert database.collection(arangodb.NODES).added_indexes == [
        {"type": "persistent", "fields": ["group_id"], "unique": False}
    ]


def test_arangodb_cleanup_truncates_only_the_benchmark_edge_collection(arango):
    adapter, database = arango
    database.collection(arangodb.WRITE_EDGES).documents = {"1": {}, "2": {}}

    removed = adapter.cleanup_writes()

    assert removed == 2
    assert database.collection(arangodb.WRITE_EDGES).truncated == 1
    assert database.collection(arangodb.EDGES).truncated == 0


def test_arangodb_reports_a_real_stored_size(arango):
    adapter, database = arango
    database.answers = [[7115], [103689]]

    footprint = adapter.footprint()

    assert footprint["server"] == "ArangoDB/3.12.4"
    assert footprint["stored_data_size"] == "users: 4194304 bytes"
    assert footprint["loaded_relationships"] == 103689


def test_arangodb_caps_query_runtime_with_the_configured_timeout(arango):
    adapter, database = arango

    adapter.ping()

    assert database.runtimes == [SETTINGS.query_timeout_s]


def test_arangodb_refuses_to_query_before_connecting():
    adapter = arangodb.ArangoDBAdapter(target("arangodb", "arangodb"), SETTINGS)

    with pytest.raises(AdapterError, match="not connected"):
        adapter.ping()

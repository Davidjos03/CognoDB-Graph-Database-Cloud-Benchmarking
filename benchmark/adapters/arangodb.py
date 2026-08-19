"""ArangoDB: the same logical workloads expressed in AQL.

ArangoDB is not a Cypher database, so every statement here is a translation
rather than the same text. The translations are written to match the *logical*
operation as closely as the data model allows, and the differences that cannot
be removed are recorded as caveats so the comparison is not silently unfair:

* the graph is a document collection ``users`` plus an edge collection ``votes``;
  a Cypher label becomes a collection, and a relationship type becomes an edge
  collection.
* traversals use ``FOR v IN <depth>..<depth> OUTBOUND`` with
  ``uniqueVertices: 'global'``, which is the AQL equivalent of counting distinct
  endpoints at exactly that depth.
* ``node_id`` is also used as the document ``_key``, so a point lookup is a
  primary-key read. That is ArangoDB's natural fastest path, and it is called
  out in the README rather than presented as identical work to an index seek.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Iterable

from arango import ArangoClient

from benchmark.adapters.base import AdapterError, BaseGraphAdapter, batched
from benchmark.adapters.bolt import CONNECT_ATTEMPTS, CONNECT_BACKOFF_S
from benchmark.config import NOT_OBSERVABLE
from benchmark.metrics import describe_error

log = logging.getLogger(__name__)

NODES = "users"
EDGES = "votes"
WRITE_EDGES = "bench_writes"
SMOKE = "bench_smoke"

LOAD_NODES = f"""
FOR row IN @rows
  INSERT {{ _key: row._key, node_id: row.node_id, group_id: row.group_id }} INTO {NODES}
"""

LOAD_EDGES = f"""
FOR row IN @rows
  INSERT {{ _from: row._from, _to: row._to }} INTO {EDGES}
"""

TRAVERSAL = f"""
RETURN LENGTH(
  FOR v IN @depth..@depth OUTBOUND @start {EDGES}
    OPTIONS {{ uniqueVertices: 'global', bfs: true }}
    RETURN DISTINCT v._key
)
"""

POINT_LOOKUP = f"RETURN DOCUMENT(CONCAT('{NODES}/', @node_id))"

FILTERED_LOOKUP = f"""
RETURN LENGTH(FOR u IN {NODES} FILTER u.group_id == @group_id RETURN 1)
"""

AGGREGATION = f"""
FOR u IN {NODES}
  COLLECT group_id = u.group_id WITH COUNT INTO users
  SORT group_id
  RETURN {{ group_id, users }}
"""

MIXED_WRITE = f"""
INSERT {{ _from: CONCAT('{NODES}/', @source), _to: CONCAT('{NODES}/', @target) }} INTO {WRITE_EDGES}
"""

COUNT_NODES = f"RETURN LENGTH({NODES})"
COUNT_EDGES = f"RETURN LENGTH({EDGES})"


class ArangoDBAdapter(BaseGraphAdapter):
    """ArangoDB through python-arango, using AQL translations of each workload."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._client = None
        self._database = None
        self._server = NOT_OBSERVABLE

    def connect(self) -> None:
        last_error = ""
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            try:
                self._open()
            except Exception as exc:
                last_error = describe_error(exc)
                self.close()
                logger = log.warning if attempt == CONNECT_ATTEMPTS else log.debug
                logger(
                    "%s: connection attempt %d of %d failed: %s",
                    self.name,
                    attempt,
                    CONNECT_ATTEMPTS,
                    last_error,
                )
                if attempt < CONNECT_ATTEMPTS:
                    time.sleep(CONNECT_BACKOFF_S * attempt)
                continue

            if attempt > 1:
                self.add_caveat(f"connection succeeded only on attempt {attempt}: {last_error}")
            self.add_caveat(
                "workloads are AQL translations of the Cypher used elsewhere, not the same "
                "statements; point lookup reads by primary key"
            )
            log.info("%s: connected to %s, server %s", self.name, self.target.safe_uri, self._server)
            return

        raise AdapterError(
            f"{self.name}: cannot connect after {CONNECT_ATTEMPTS} attempts: {last_error}"
        )

    def _open(self) -> None:
        self._client = ArangoClient(hosts=self.target.uri)
        name = self.target.database or "_system"
        system = self._client.db(
            "_system", username=self.target.username, password=self.target.password, verify=True
        )
        if name != "_system" and not system.has_database(name):
            system.create_database(name)
        self._database = self._client.db(
            name, username=self.target.username, password=self.target.password, verify=True
        )
        version = self._database.version()
        self._server = f"ArangoDB/{version}"
        self._ensure_collections()

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
        except Exception as exc:
            log.debug("%s: error while closing: %s", self.name, describe_error(exc))
        self._client = None
        self._database = None

    def _ensure_collections(self) -> None:
        database = self._require_database()
        if not database.has_collection(NODES):
            database.create_collection(NODES)
        for name in (EDGES, WRITE_EDGES):
            if not database.has_collection(name):
                database.create_collection(name, edge=True)

    def smoke_test(self) -> None:
        database = self._require_database()
        if not database.has_collection(SMOKE):
            database.create_collection(SMOKE)
        collection = database.collection(SMOKE)

        marker = f"smoke-{uuid.uuid4()}"
        inserted = collection.insert({"marker": marker})
        found = collection.get(inserted["_key"])
        if not found or found["marker"] != marker:
            raise AdapterError(f"{self.name}: created document did not read back")
        collection.delete(inserted["_key"])
        if collection.get(inserted["_key"]) is not None:
            raise AdapterError(f"{self.name}: smoke test could not delete its own document")
        log.info("%s: smoke test passed (create, match, delete)", self.name)

    def ping(self) -> None:
        self._query("RETURN 1")

    def reset_test_data(self) -> None:
        database = self._require_database()
        for name in (NODES, EDGES, WRITE_EDGES):
            if database.has_collection(name):
                database.collection(name).truncate()

    def create_indexes(self) -> None:
        """``node_id`` is the primary key already; ``group_id`` needs an index."""
        collection = self._require_database().collection(NODES)
        try:
            collection.add_index({"type": "persistent", "fields": ["group_id"], "unique": False})
        except Exception as exc:
            self.add_caveat(f"index not created (group_id): {describe_error(exc)}")

    def load_nodes(self, nodes: Iterable[tuple[int, int]]) -> int:
        loaded = 0
        for batch in batched(nodes, self.settings.batch_size):
            rows = [
                {"_key": str(node_id), "node_id": node_id, "group_id": group_id}
                for node_id, group_id in batch
            ]
            self._query(LOAD_NODES, rows=rows)
            loaded += len(rows)
        return loaded

    def load_relationships(self, edges: Iterable[tuple[int, int]]) -> int:
        loaded = 0
        for batch in batched(edges, self.settings.batch_size):
            rows = [
                {"_from": f"{NODES}/{source}", "_to": f"{NODES}/{target}"}
                for source, target in batch
            ]
            self._query(LOAD_EDGES, rows=rows)
            loaded += len(rows)
        return loaded

    def traversal(self, depth: int, start_node: int) -> int:
        if depth not in (1, 2, 3):
            raise AdapterError(f"unsupported traversal depth {depth}")
        rows = self._query(TRAVERSAL, depth=depth, start=f"{NODES}/{start_node}")
        return int(rows[0]) if rows else 0

    def point_lookup(self, node_id: int) -> int:
        rows = self._query(POINT_LOOKUP, node_id=str(node_id))
        return 1 if rows and rows[0] else 0

    def filtered_lookup(self, group_id: int) -> int:
        rows = self._query(FILTERED_LOOKUP, group_id=group_id)
        return int(rows[0]) if rows else 0

    def aggregation(self) -> int:
        return len(self._query(AGGREGATION))

    def mixed_read(self, node_id: int) -> int:
        return self.traversal(1, node_id)

    def mixed_write(self, source: int, target: int) -> None:
        self._query(MIXED_WRITE, source=str(source), target=str(target))

    def cleanup_writes(self) -> int:
        database = self._require_database()
        if not database.has_collection(WRITE_EDGES):
            return 0
        collection = database.collection(WRITE_EDGES)
        removed = collection.count()
        collection.truncate()
        return removed

    def footprint(self) -> dict:
        observed: dict = {
            "configured_specs": self.target.specs,
            "server": self._server,
            "indexes": self._index_summary(),
            "stored_data_size": self._stored_size(),
        }
        for key, query in (("loaded_nodes", COUNT_NODES), ("loaded_relationships", COUNT_EDGES)):
            try:
                rows = self._query(query)
                observed[key] = int(rows[0]) if rows else NOT_OBSERVABLE
            except AdapterError:
                observed[key] = NOT_OBSERVABLE
        return observed

    def _index_summary(self) -> list[str] | str:
        try:
            indexes = self._require_database().collection(NODES).indexes()
        except Exception:
            return NOT_OBSERVABLE
        return [f"{NODES}({','.join(index.get('fields', []))})" for index in indexes]

    def _stored_size(self) -> str:
        """ArangoDB reports per-collection figures, which is a real footprint."""
        try:
            statistics = self._require_database().collection(NODES).statistics()
        except Exception:
            return NOT_OBSERVABLE
        size = statistics.get("documents_size") or statistics.get("file_size")
        return f"{NODES}: {size} bytes" if size else NOT_OBSERVABLE

    def _require_database(self):
        if self._database is None:
            raise AdapterError(f"{self.name}: not connected; call connect() first")
        return self._database

    def _query(self, query: str, **bind_variables: object) -> list:
        database = self._require_database()
        try:
            cursor = database.aql.execute(
                query,
                bind_vars=bind_variables or None,
                max_runtime=self.settings.query_timeout_s,
            )
            return list(cursor)
        except Exception as exc:
            raise AdapterError(f"{self.name}: query failed: {describe_error(exc)}") from exc

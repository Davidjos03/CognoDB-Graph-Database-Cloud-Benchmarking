"""Cypher over Bolt, using the official Neo4j driver.

CognoDB Cloud speaks the Bolt protocol and Cypher, so the same adapter drives
CognoDB and Neo4j with identical statements — no per-platform query rewriting,
which keeps the comparison honest.

Labels and relationship types cannot be parameterised in Cypher, so the
statements are built once from the dataset's model constants. Every value is
passed as a parameter.
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterable

from neo4j import GraphDatabase

from benchmark.adapters.base import AdapterError, BaseGraphAdapter, batched
from benchmark.config import NOT_OBSERVABLE
from benchmark.dataset import NODE_LABEL, RELATIONSHIP_TYPE
from benchmark.metrics import describe_error

log = logging.getLogger(__name__)

# Mixed-workload writes use their own relationship type, so the canonical graph
# is never modified and can be cleaned up in one statement.
WRITE_RELATIONSHIP = "BENCH_WRITE"
SMOKE_LABEL = "BenchSmoke"

CREATE_INDEXES = (
    f"CREATE INDEX user_node_id IF NOT EXISTS FOR (u:{NODE_LABEL}) ON (u.node_id)",
    f"CREATE INDEX user_group_id IF NOT EXISTS FOR (u:{NODE_LABEL}) ON (u.group_id)",
)

LOAD_NODES = f"""
UNWIND $rows AS row
CREATE (u:{NODE_LABEL} {{node_id: row.node_id, group_id: row.group_id}})
"""

LOAD_RELATIONSHIPS = f"""
UNWIND $rows AS row
MATCH (a:{NODE_LABEL} {{node_id: row.source}})
MATCH (b:{NODE_LABEL} {{node_id: row.target}})
CREATE (a)-[:{RELATIONSHIP_TYPE}]->(b)
"""

# "Reachable in exactly N hops", written out per depth rather than as a
# variable-length pattern, so every platform runs the same shape of work.
TRAVERSALS = {
    1: f"""
MATCH (u:{NODE_LABEL} {{node_id: $node_id}})-[:{RELATIONSHIP_TYPE}]->(v:{NODE_LABEL})
RETURN count(DISTINCT v) AS reached
""",
    2: f"""
MATCH (u:{NODE_LABEL} {{node_id: $node_id}})-[:{RELATIONSHIP_TYPE}]->()-[:{RELATIONSHIP_TYPE}]->(v:{NODE_LABEL})
RETURN count(DISTINCT v) AS reached
""",
    3: f"""
MATCH (u:{NODE_LABEL} {{node_id: $node_id}})-[:{RELATIONSHIP_TYPE}]->()-[:{RELATIONSHIP_TYPE}]->()-[:{RELATIONSHIP_TYPE}]->(v:{NODE_LABEL})
RETURN count(DISTINCT v) AS reached
""",
}

POINT_LOOKUP = f"""
MATCH (u:{NODE_LABEL} {{node_id: $node_id}})
RETURN u.node_id AS node_id, u.group_id AS group_id
"""

FILTERED_LOOKUP = f"""
MATCH (u:{NODE_LABEL} {{group_id: $group_id}})
RETURN count(u) AS matched
"""

AGGREGATION = f"""
MATCH (u:{NODE_LABEL})
RETURN u.group_id AS group_id, count(u) AS users
ORDER BY group_id
"""

MIXED_WRITE = f"""
MATCH (a:{NODE_LABEL} {{node_id: $source}})
MATCH (b:{NODE_LABEL} {{node_id: $target}})
CREATE (a)-[:{WRITE_RELATIONSHIP}]->(b)
"""

DELETE_NODES = f"""
MATCH (n:{NODE_LABEL})
WITH n LIMIT $batch
DETACH DELETE n
RETURN count(n) AS deleted
"""

DELETE_WRITES = f"""
MATCH ()-[r:{WRITE_RELATIONSHIP}]->()
WITH r LIMIT $batch
DELETE r
RETURN count(r) AS deleted
"""

COUNT_NODES = f"MATCH (n:{NODE_LABEL}) RETURN count(n) AS value"
COUNT_RELATIONSHIPS = f"MATCH ()-[r:{RELATIONSHIP_TYPE}]->() RETURN count(r) AS value"

# `dbms.components()` is not available on every Bolt server (CognoDB rejects it),
# so the server identity comes from the Bolt handshake instead.
PING = "RETURN 1 AS one"
SHOW_INDEXES = "SHOW INDEXES"


class BoltAdapter(BaseGraphAdapter):
    """A Bolt + Cypher platform. One instance owns one session, so concurrent
    clients each build their own adapter."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._driver = None
        self._session = None
        self._server_agent = NOT_OBSERVABLE

    def connect(self) -> None:
        timeout = self.settings.query_timeout_s
        try:
            self._driver = GraphDatabase.driver(
                self.target.uri,
                auth=(self.target.username, self.target.password),
                connection_timeout=timeout,
                connection_acquisition_timeout=timeout,
                max_transaction_retry_time=timeout,
            )
            self._driver.verify_connectivity()
            self._session = self._driver.session(database=self.target.database or None)
            summary = self._session.run(PING).consume()
            self._server_agent = summary.server.agent or NOT_OBSERVABLE
        except Exception as exc:
            self.close()
            raise AdapterError(f"{self.name}: cannot connect: {describe_error(exc)}") from exc
        log.info("%s: connected to %s, server %s", self.name, self.target.safe_uri, self._server_agent)

    def close(self) -> None:
        for resource in (self._session, self._driver):
            try:
                if resource is not None:
                    resource.close()
            except Exception as exc:  # closing must never mask the real error
                log.debug("%s: error while closing: %s", self.name, describe_error(exc))
        self._session = None
        self._driver = None

    def smoke_test(self) -> None:
        marker = f"smoke-{uuid.uuid4()}"
        created = self._single(
            f"CREATE (n:{SMOKE_LABEL} {{marker: $marker}}) RETURN n.marker AS marker",
            marker=marker,
        )
        if not created or created["marker"] != marker:
            raise AdapterError(f"{self.name}: created node did not read back")

        found = self._single(
            f"MATCH (n:{SMOKE_LABEL} {{marker: $marker}}) RETURN count(n) AS found", marker=marker
        )
        deleted = self._single(
            f"MATCH (n:{SMOKE_LABEL} {{marker: $marker}}) DELETE n RETURN count(n) AS deleted",
            marker=marker,
        )
        if not found or found["found"] != 1 or not deleted or deleted["deleted"] != 1:
            raise AdapterError(f"{self.name}: smoke test could not match and delete its own node")
        log.info("%s: smoke test passed (create, match, delete)", self.name)

    def reset_test_data(self) -> None:
        self.cleanup_writes()
        deleted = self._delete_in_batches(DELETE_NODES)
        log.info("%s: reset removed %d nodes", self.name, deleted)

    def create_indexes(self) -> None:
        for statement in CREATE_INDEXES:
            try:
                self._execute(statement)
            except AdapterError as exc:
                self.add_caveat(f"index not created ({statement.split()[2]}): {exc}")

    def load_nodes(self, nodes: Iterable[tuple[int, int]]) -> int:
        loaded = 0
        for batch in batched(nodes, self.settings.batch_size):
            rows = [{"node_id": node_id, "group_id": group_id} for node_id, group_id in batch]
            self._execute(LOAD_NODES, rows=rows)
            loaded += len(rows)
        return loaded

    def load_relationships(self, edges: Iterable[tuple[int, int]]) -> int:
        loaded = 0
        for batch in batched(edges, self.settings.batch_size):
            rows = [{"source": source, "target": target} for source, target in batch]
            self._execute(LOAD_RELATIONSHIPS, rows=rows)
            loaded += len(rows)
        return loaded

    def traversal(self, depth: int, start_node: int) -> int:
        try:
            query = TRAVERSALS[depth]
        except KeyError:
            raise AdapterError(f"unsupported traversal depth {depth}") from None
        record = self._single(query, node_id=start_node)
        return int(record["reached"]) if record else 0

    def point_lookup(self, node_id: int) -> int:
        record = self._single(POINT_LOOKUP, node_id=node_id)
        return 1 if record else 0

    def filtered_lookup(self, group_id: int) -> int:
        record = self._single(FILTERED_LOOKUP, group_id=group_id)
        return int(record["matched"]) if record else 0

    def aggregation(self) -> int:
        return len(self._all(AGGREGATION))

    def mixed_read(self, node_id: int) -> int:
        return self.traversal(1, node_id)

    def mixed_write(self, source: int, target: int) -> None:
        self._execute(MIXED_WRITE, source=source, target=target)

    def cleanup_writes(self) -> int:
        return self._delete_in_batches(DELETE_WRITES)

    def footprint(self) -> dict:
        observed: dict = {
            "configured_specs": self.target.specs,
            "server": self._server_agent,
            "indexes": self._index_summary(),
        }

        for key, query in (
            ("loaded_nodes", COUNT_NODES),
            ("loaded_relationships", COUNT_RELATIONSHIPS),
        ):
            record = self._optional_single(query)
            observed[key] = int(record["value"]) if record else NOT_OBSERVABLE

        # Bolt exposes no portable store-size metric; say so rather than guess.
        observed["stored_data_size"] = NOT_OBSERVABLE
        return observed

    def _index_summary(self) -> list[str] | str:
        """Which properties are indexed, as the platform itself reports them.

        Servers disagree on the shape of ``SHOW INDEXES``: Neo4j returns
        ``labelsOrTypes``/``properties`` as lists, CognoDB returns ``label`` and
        a single property string.
        """
        try:
            rows = self._all(SHOW_INDEXES)
        except AdapterError:
            return NOT_OBSERVABLE
        described = []
        for row in rows:
            labels = ",".join(_as_list(row.get("labelsOrTypes") or row.get("label")))
            properties = ",".join(_as_list(row.get("properties")))
            described.append(f"{labels}({properties})" if properties else str(row.get("name")))
        return described

    def _delete_in_batches(self, query: str) -> int:
        total = 0
        while True:
            record = self._single(query, batch=self.settings.batch_size)
            deleted = int(record["deleted"]) if record else 0
            total += deleted
            if deleted == 0:
                return total

    def _execute(self, query: str, **parameters: object) -> None:
        self._run(query, **parameters).consume()

    def _run(self, query: str, **parameters: object):
        if self._session is None:
            raise AdapterError(f"{self.name}: not connected; call connect() first")
        try:
            return self._session.run(query, **parameters)
        except Exception as exc:
            raise AdapterError(f"{self.name}: query failed: {describe_error(exc)}") from exc

    def _single(self, query: str, **parameters: object) -> dict | None:
        result = self._run(query, **parameters)
        try:
            record = result.single()
            result.consume()
        except Exception as exc:
            raise AdapterError(f"{self.name}: query failed: {describe_error(exc)}") from exc
        return dict(record) if record is not None else None

    def _all(self, query: str, **parameters: object) -> list[dict]:
        result = self._run(query, **parameters)
        try:
            records = [dict(record) for record in result]
        except Exception as exc:
            raise AdapterError(f"{self.name}: query failed: {describe_error(exc)}") from exc
        return records

    def _optional_single(self, query: str, **parameters: object) -> dict | None:
        """Run a query that may be unsupported; report nothing rather than fail."""
        try:
            return self._single(query, **parameters)
        except AdapterError as exc:
            log.debug("%s: not observable: %s", self.name, exc)
            return None


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]

"""FalkorDB: Cypher over the Redis protocol.

FalkorDB accepts the same logical Cypher as the Bolt platforms, so the read and
write statements are imported from the Bolt adapter unchanged — only the
transport, the index DDL and the catalogue queries differ. Where behaviour is
genuinely different it is recorded as a caveat rather than hidden:

* index DDL uses the legacy ``CREATE INDEX ON :Label(property)`` form.
* indexes are listed through ``CALL db.indexes()``.
* results come back as positional rows, not records keyed by name.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Iterable, Sequence

from falkordb import FalkorDB

from benchmark.adapters.base import AdapterError, BaseGraphAdapter, batched
from benchmark.adapters.bolt import (
    AGGREGATION,
    CONNECT_ATTEMPTS,
    CONNECT_BACKOFF_S,
    COUNT_NODES,
    COUNT_RELATIONSHIPS,
    DELETE_WRITES,
    FILTERED_LOOKUP,
    LOAD_NODES,
    LOAD_RELATIONSHIPS,
    MIXED_WRITE,
    POINT_LOOKUP,
    SMOKE_LABEL,
    TRAVERSALS,
)
from benchmark.config import NOT_OBSERVABLE
from benchmark.dataset import NODE_LABEL
from benchmark.metrics import describe_error

log = logging.getLogger(__name__)

DEFAULT_GRAPH = "benchmark"

CREATE_INDEXES = (
    f"CREATE INDEX ON :{NODE_LABEL}(node_id)",
    f"CREATE INDEX ON :{NODE_LABEL}(group_id)",
)

SHOW_INDEXES = "CALL db.indexes()"
PING = "RETURN 1"


class FalkorDBAdapter(BaseGraphAdapter):
    """One FalkorDB graph, driven through the FalkorDB Python client."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._client = None
        self._graph = None
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
            log.info("%s: connected to %s, server %s", self.name, self.target.safe_uri, self._server)
            return

        raise AdapterError(
            f"{self.name}: cannot connect after {CONNECT_ATTEMPTS} attempts: {last_error}"
        )

    def _open(self) -> None:
        self._client = FalkorDB.from_url(self.target.uri)
        info = self._client.connection.info("server")
        self._server = f"FalkorDB/redis {info.get('redis_version', 'unknown')}"
        self._graph = self._client.select_graph(self.target.database or DEFAULT_GRAPH)
        self._query(PING)

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.connection.close()
        except Exception as exc:
            log.debug("%s: error while closing: %s", self.name, describe_error(exc))
        self._client = None
        self._graph = None

    def smoke_test(self) -> None:
        marker = f"smoke-{uuid.uuid4()}"
        created = self._query(
            f"CREATE (n:{SMOKE_LABEL} {{marker: $marker}}) RETURN n.marker", marker=marker
        )
        if not created or created[0][0] != marker:
            raise AdapterError(f"{self.name}: created node did not read back")

        deleted = self._query(
            f"MATCH (n:{SMOKE_LABEL} {{marker: $marker}}) DELETE n RETURN count(n)", marker=marker
        )
        if not deleted or int(deleted[0][0]) != 1:
            raise AdapterError(f"{self.name}: smoke test could not delete its own node")
        log.info("%s: smoke test passed (create, match, delete)", self.name)

    def ping(self) -> None:
        self._query(PING)

    def reset_test_data(self) -> None:
        """Drop the whole graph key: cheaper and more thorough than deleting rows."""
        try:
            if self._graph is not None:
                self._graph.delete()
        except Exception as exc:
            log.debug("%s: nothing to drop: %s", self.name, describe_error(exc))
        self._graph = self._client.select_graph(self.target.database or DEFAULT_GRAPH)

    def create_indexes(self) -> None:
        for statement in CREATE_INDEXES:
            try:
                self._query(statement)
            except AdapterError as exc:
                if "already indexed" in str(exc).lower():
                    continue
                self.add_caveat(f"index not created ({statement}): {exc}")

    def load_nodes(self, nodes: Iterable[tuple[int, int]]) -> int:
        loaded = 0
        for batch in batched(nodes, self.settings.batch_size):
            rows = [{"node_id": node_id, "group_id": group_id} for node_id, group_id in batch]
            self._query(LOAD_NODES, rows=rows)
            loaded += len(rows)
        return loaded

    def load_relationships(self, edges: Iterable[tuple[int, int]]) -> int:
        loaded = 0
        for batch in batched(edges, self.settings.batch_size):
            rows = [{"source": source, "target": target} for source, target in batch]
            self._query(LOAD_RELATIONSHIPS, rows=rows)
            loaded += len(rows)
        return loaded

    def traversal(self, depth: int, start_node: int) -> int:
        try:
            query = TRAVERSALS[depth]
        except KeyError:
            raise AdapterError(f"unsupported traversal depth {depth}") from None
        return self._scalar(query, node_id=start_node)

    def point_lookup(self, node_id: int) -> int:
        return len(self._query(POINT_LOOKUP, node_id=node_id))

    def filtered_lookup(self, group_id: int) -> int:
        return self._scalar(FILTERED_LOOKUP, group_id=group_id)

    def aggregation(self) -> int:
        return len(self._query(AGGREGATION))

    def mixed_read(self, node_id: int) -> int:
        return self.traversal(1, node_id)

    def mixed_write(self, source: int, target: int) -> None:
        self._query(MIXED_WRITE, source=source, target=target)

    def cleanup_writes(self) -> int:
        removed = 0
        while True:
            deleted = self._scalar(DELETE_WRITES, batch=self.settings.batch_size)
            removed += deleted
            if deleted == 0:
                return removed

    def footprint(self) -> dict:
        observed: dict = {
            "configured_specs": self.target.specs,
            "server": self._server,
            "indexes": self._index_summary(),
            "stored_data_size": self._memory_usage(),
        }
        for key, query in (
            ("loaded_nodes", COUNT_NODES),
            ("loaded_relationships", COUNT_RELATIONSHIPS),
        ):
            try:
                observed[key] = self._scalar(query)
            except AdapterError:
                observed[key] = NOT_OBSERVABLE
        return observed

    def _index_summary(self) -> list[str] | str:
        try:
            rows = self._query(SHOW_INDEXES)
        except AdapterError:
            return NOT_OBSERVABLE
        return [f"{row[0]}({row[1]})" for row in rows if len(row) >= 2]

    def _memory_usage(self) -> str:
        """FalkorDB is in-memory, so Redis reports a real footprint."""
        try:
            info = self._client.connection.info("memory")
        except Exception:
            return NOT_OBSERVABLE
        return str(info.get("used_memory_human", NOT_OBSERVABLE))

    def _query(self, query: str, **parameters: object) -> Sequence[Sequence]:
        if self._graph is None:
            raise AdapterError(f"{self.name}: not connected; call connect() first")
        try:
            result = self._graph.query(query, params=parameters or None)
        except Exception as exc:
            raise AdapterError(f"{self.name}: query failed: {describe_error(exc)}") from exc
        return result.result_set or []

    def _scalar(self, query: str, **parameters: object) -> int:
        rows = self._query(query, **parameters)
        if not rows or not rows[0]:
            return 0
        return int(rows[0][0])

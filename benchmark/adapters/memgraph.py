"""Memgraph: Bolt and Cypher, but not identical Cypher.

Memgraph speaks Bolt, so it reuses the whole Bolt adapter. Two statements differ
and are overridden here rather than being papered over:

* indexes use the legacy ``CREATE INDEX ON :Label(property)`` form; the
  ``CREATE INDEX ... IF NOT EXISTS FOR (n:Label) ON (n.prop)`` form used by
  Neo4j 5 and CognoDB is a syntax error on Memgraph.
* ``SHOW INDEX INFO`` replaces ``SHOW INDEXES``.
"""

from __future__ import annotations

from benchmark.adapters.bolt import BoltAdapter
from benchmark.config import NOT_OBSERVABLE
from benchmark.dataset import NODE_LABEL

CREATE_INDEXES = (
    f"CREATE INDEX ON :{NODE_LABEL}(node_id)",
    f"CREATE INDEX ON :{NODE_LABEL}(group_id)",
)

SHOW_INDEXES = "SHOW INDEX INFO"


class MemgraphAdapter(BoltAdapter):
    """Bolt with Memgraph's index syntax."""

    def create_indexes(self) -> None:
        for statement in CREATE_INDEXES:
            try:
                self._execute(statement)
            except Exception as exc:
                self.add_caveat(f"index not created ({statement}): {exc}")

    def _index_summary(self) -> list[str] | str:
        try:
            rows = self._all(SHOW_INDEXES)
        except Exception:
            return NOT_OBSERVABLE
        return [f"{row.get('label')}({row.get('property')})" for row in rows]

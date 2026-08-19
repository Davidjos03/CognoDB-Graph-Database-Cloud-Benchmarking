"""An in-memory adapter, so the loader and runner can be tested without a network."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from benchmark.adapters.base import AdapterError, BaseGraphAdapter
from benchmark.config import Settings, Target

SETTINGS = Settings(
    seed=42,
    warmup_iterations=1,
    measured_iterations=3,
    concurrency_levels=(1, 2),
    read_ratio=0.9,
    batch_size=2,
    query_timeout_s=5.0,
    data_dir=Path("data"),
    results_dir=Path("results"),
)

TARGET = Target(
    name="fake",
    kind="fake",
    uri="bolt+s://host.example.com",
    username="user",
    password="secret",
    database="neo4j",
    specs="entry tier",
)


class FakeAdapter(BaseGraphAdapter):
    """Holds the graph in dictionaries and records how it was driven."""

    def __init__(self, target: Target = TARGET, settings: Settings = SETTINGS) -> None:
        super().__init__(target, settings)
        self.nodes: dict[int, int] = {}
        self.edges: list[tuple[int, int]] = []
        self.writes: list[tuple[int, int]] = []
        self.calls: list[str] = []
        self.indexes: list[str] = []
        self.connected = False
        self.drop_every_nth_node = 0
        self.fail_on: set[str] = set()

    def _record(self, call: str) -> None:
        self.calls.append(call)
        if call in self.fail_on:
            raise AdapterError(f"fake: {call} failed")

    def connect(self) -> None:
        self._record("connect")
        self.connected = True

    def close(self) -> None:
        self._record("close")
        self.connected = False

    def smoke_test(self) -> None:
        self._record("smoke_test")

    def ping(self) -> None:
        self._record("ping")

    def reset_test_data(self) -> None:
        self._record("reset")
        self.nodes.clear()
        self.edges.clear()
        self.writes.clear()

    def create_indexes(self) -> None:
        self._record("create_indexes")
        self.indexes = ["node_id", "group_id"]

    def load_nodes(self, nodes: Iterable[tuple[int, int]]) -> int:
        self._record("load_nodes")
        loaded = 0
        for index, (node_id, group_id) in enumerate(nodes):
            # Simulates a platform silently dropping rows, to test verification.
            if self.drop_every_nth_node and index % self.drop_every_nth_node == 0:
                continue
            self.nodes[node_id] = group_id
            loaded += 1
        return loaded

    def load_relationships(self, edges: Iterable[tuple[int, int]]) -> int:
        self._record("load_relationships")
        self.edges.extend(edges)
        return len(self.edges)

    def traversal(self, depth: int, start_node: int) -> int:
        self._record(f"traversal:{depth}")
        reached = {start_node}
        for _ in range(depth):
            reached = {target for source, target in self.edges if source in reached}
        return len(reached)

    def point_lookup(self, node_id: int) -> int:
        self._record("point_lookup")
        return 1 if node_id in self.nodes else 0

    def filtered_lookup(self, group_id: int) -> int:
        self._record("filtered_lookup")
        return sum(1 for group in self.nodes.values() if group == group_id)

    def aggregation(self) -> int:
        self._record("aggregation")
        return len(set(self.nodes.values()))

    def mixed_read(self, node_id: int) -> int:
        self._record("mixed_read")
        return self.traversal(1, node_id)

    def mixed_write(self, source: int, target: int) -> None:
        self._record("mixed_write")
        self.writes.append((source, target))

    def cleanup_writes(self) -> int:
        self._record("cleanup_writes")
        removed = len(self.writes)
        self.writes.clear()
        return removed

    def footprint(self) -> dict:
        self._record("footprint")
        return {
            "configured_specs": self.target.specs,
            "loaded_nodes": len(self.nodes),
            "loaded_relationships": len(self.edges),
        }

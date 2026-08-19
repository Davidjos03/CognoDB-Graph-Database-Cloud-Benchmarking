"""Load the shared dataset into one platform and time the ingest.

The order is the same everywhere — reset, index, nodes, relationships — because
loading order changes ingest speed. Indexes are created *before* the load so
that every platform resolves the endpoints of a relationship through an index
rather than a scan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, TypeVar

from benchmark.adapters.base import BaseGraphAdapter
from benchmark.config import NOT_OBSERVABLE
from benchmark.dataset import Dataset
from benchmark.metrics import stopwatch, throughput

log = logging.getLogger(__name__)

PROGRESS_EVERY = 20_000

T = TypeVar("T")


@dataclass(frozen=True)
class LoadReport:
    """What a load actually achieved, as opposed to what was asked for."""

    node_count: int
    relationship_count: int
    node_seconds: float
    relationship_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.node_seconds + self.relationship_seconds

    @property
    def nodes_per_second(self) -> float:
        return throughput(self.node_count, self.node_seconds)

    @property
    def relationships_per_second(self) -> float:
        return throughput(self.relationship_count, self.relationship_seconds)


def load_into(adapter: BaseGraphAdapter, data: Dataset) -> LoadReport:
    """Reset, index and bulk load ``data``, timing each phase separately."""
    log.info("%s: clearing any previous benchmark data", adapter.name)
    adapter.reset_test_data()

    log.info("%s: creating indexes before the load", adapter.name)
    adapter.create_indexes()

    log.info("%s: loading %d nodes", adapter.name, data.node_count)
    with stopwatch() as node_watch:
        nodes_loaded = adapter.load_nodes(
            _with_progress(data.nodes(), data.node_count, f"{adapter.name} nodes")
        )

    log.info("%s: loading %d relationships", adapter.name, data.relationship_count)
    with stopwatch() as edge_watch:
        edges_loaded = adapter.load_relationships(
            _with_progress(iter(data.edges), data.relationship_count, f"{adapter.name} relationships")
        )

    report = LoadReport(
        node_count=nodes_loaded,
        relationship_count=edges_loaded,
        node_seconds=node_watch.elapsed_s,
        relationship_seconds=edge_watch.elapsed_s,
    )
    _verify(adapter, data, report)

    log.info(
        "%s: loaded in %.1fs (%.0f nodes/s, %.0f relationships/s)",
        adapter.name,
        report.total_seconds,
        report.nodes_per_second,
        report.relationships_per_second,
    )
    return report


def _verify(adapter: BaseGraphAdapter, data: Dataset, report: LoadReport) -> None:
    """Compare what the client sent with what the server reports, and record any gap."""
    if report.node_count != data.node_count:
        adapter.add_caveat(
            f"sent {report.node_count} nodes but the dataset has {data.node_count}"
        )
    if report.relationship_count != data.relationship_count:
        adapter.add_caveat(
            f"sent {report.relationship_count} relationships but the dataset has "
            f"{data.relationship_count}"
        )

    footprint = adapter.footprint()
    for key, expected in (
        ("loaded_nodes", data.node_count),
        ("loaded_relationships", data.relationship_count),
    ):
        observed = footprint.get(key, NOT_OBSERVABLE)
        if isinstance(observed, int) and observed != expected:
            adapter.add_caveat(f"{key}: server reports {observed}, dataset has {expected}")


def _with_progress(items: Iterable[T], total: int, label: str) -> Iterator[T]:
    """Log progress on a long load, so a slow free tier does not look like a hang."""
    for index, item in enumerate(items, start=1):
        if index % PROGRESS_EVERY == 0:
            log.info("%s: %d/%d (%.0f%%)", label, index, total, 100 * index / total)
        yield item

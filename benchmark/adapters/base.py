"""The interface every platform adapter implements.

The benchmark only ever calls these methods, so a platform is added by writing
one adapter rather than by touching the runner. Each method is one logical
operation; where a platform cannot express it with the same semantics, the
adapter records a caveat that travels into the result file and the README.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Sequence

from benchmark.config import Settings, Target

log = logging.getLogger(__name__)


class AdapterError(Exception):
    """Raised when a platform cannot be reached or a statement is rejected."""


class BaseGraphAdapter(ABC):
    """One database under test.

    Lifecycle: ``connect`` → ``reset_test_data`` → ``create_indexes`` →
    ``load_nodes`` → ``load_relationships`` → workloads → ``close``.
    """

    def __init__(self, target: Target, settings: Settings) -> None:
        self.target = target
        self.settings = settings
        self._caveats: list[str] = []

    @property
    def name(self) -> str:
        return self.target.name

    @property
    def caveats(self) -> list[str]:
        return list(self._caveats)

    def add_caveat(self, caveat: str) -> None:
        """Record a limitation of this platform's mapping of a workload."""
        if caveat not in self._caveats:
            log.warning("%s: %s", self.name, caveat)
            self._caveats.append(caveat)

    def __enter__(self) -> BaseGraphAdapter:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @abstractmethod
    def connect(self) -> None:
        """Open a connection and fail fast if the platform is unreachable."""

    @abstractmethod
    def close(self) -> None:
        """Release the connection."""

    @abstractmethod
    def smoke_test(self) -> None:
        """Create, read and delete a throwaway node without touching the graph."""

    @abstractmethod
    def reset_test_data(self) -> None:
        """Delete the benchmark graph so a load starts from an empty database."""

    @abstractmethod
    def create_indexes(self) -> None:
        """Index ``node_id`` and ``group_id``; record a caveat if unsupported."""

    @abstractmethod
    def load_nodes(self, nodes: Iterable[tuple[int, int]]) -> int:
        """Insert ``(node_id, group_id)`` rows in batches, returning the count."""

    @abstractmethod
    def load_relationships(self, edges: Iterable[tuple[int, int]]) -> int:
        """Insert ``(source, target)`` edges in batches, returning the count."""

    @abstractmethod
    def traversal(self, depth: int, start_node: int) -> int:
        """Count distinct users reachable in exactly ``depth`` hops."""

    @abstractmethod
    def point_lookup(self, node_id: int) -> int:
        """Fetch one user by indexed ``node_id``."""

    @abstractmethod
    def filtered_lookup(self, group_id: int) -> int:
        """Count users matching an indexed ``group_id``."""

    @abstractmethod
    def aggregation(self) -> int:
        """Group users by ``group_id`` and count them."""

    @abstractmethod
    def mixed_read(self, node_id: int) -> int:
        """The read half of the mixed workload: a 1-hop neighbour count."""

    @abstractmethod
    def mixed_write(self, source: int, target: int) -> None:
        """The write half, using a separate relationship type so the canonical
        graph stays intact."""

    @abstractmethod
    def cleanup_writes(self) -> int:
        """Remove everything the mixed workload wrote, returning the count."""

    @abstractmethod
    def footprint(self) -> dict:
        """Report whatever the platform exposes about size and version."""


def batched(items: Iterable[tuple[int, int]], size: int) -> Iterator[Sequence[tuple[int, int]]]:
    """Yield lists of at most ``size`` items, so a load never buffers everything."""
    batch: list[tuple[int, int]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

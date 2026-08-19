"""The workload set every platform runs.

Query parameters are drawn from the dataset with the configured seed, so each
platform sees the *same* start nodes, the same lookup keys and the same groups,
in the same order. A workload is one logical operation; how a platform expresses
it is the adapter's business.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from benchmark import dataset
from benchmark.adapters.base import BaseGraphAdapter
from benchmark.config import Settings
from benchmark.dataset import Dataset

TRAVERSAL_DEPTHS = (1, 2, 3)

# Not a required metric, but without it there is no way to tell how much of a
# latency number is the query and how much is the trip to the region.
BASELINE_NAME = "network baseline"

# Distinct offsets keep the parameter streams independent: traversal start nodes
# must not be the same sequence as point-lookup keys.
TRAVERSAL_SEED_OFFSET = 0
POINT_LOOKUP_SEED_OFFSET = 1
FILTERED_LOOKUP_SEED_OFFSET = 2
MIXED_SEED_OFFSET = 3


@dataclass(frozen=True)
class Workload:
    """A named operation that can be run repeatedly with an iteration number."""

    name: str
    operation: Callable[[int], object]


def traversal_name(depth: int) -> str:
    return f"{depth}-hop traversal"


def build_read_workloads(
    adapter: BaseGraphAdapter, data: Dataset, settings: Settings
) -> list[Workload]:
    """Build the read workloads, with parameters fixed by the seed."""
    count = parameter_count(settings)
    start_nodes = dataset.sample_start_nodes(
        data, count, seed=settings.seed + TRAVERSAL_SEED_OFFSET
    )
    lookup_nodes = dataset.sample_node_ids(
        data, count, seed=settings.seed + POINT_LOOKUP_SEED_OFFSET
    )
    group_ids = dataset.sample_group_ids(count, seed=settings.seed + FILTERED_LOOKUP_SEED_OFFSET)

    workloads = [Workload(BASELINE_NAME, lambda _: adapter.ping())]
    workloads += [
        Workload(
            traversal_name(depth),
            lambda iteration, depth=depth: adapter.traversal(depth, start_nodes[iteration]),
        )
        for depth in TRAVERSAL_DEPTHS
    ]
    workloads.append(
        Workload("point lookup", lambda iteration: adapter.point_lookup(lookup_nodes[iteration]))
    )
    workloads.append(
        Workload(
            "filtered lookup", lambda iteration: adapter.filtered_lookup(group_ids[iteration])
        )
    )
    workloads.append(Workload("aggregation", lambda _: adapter.aggregation()))
    return workloads


def parameter_count(settings: Settings) -> int:
    """How many parameters a workload needs.

    ``measure`` numbers the warm-up pass and the measured pass from zero, so the
    longer of the two decides how many values must exist.
    """
    return max(settings.measured_iterations, settings.warmup_iterations, 1)

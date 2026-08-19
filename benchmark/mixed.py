"""Mixed read/write workload, run at several client concurrency levels.

Each client owns its own connection, because a driver session belongs to one
thread. Clients warm up, wait at a barrier, and only then start the measured
window, so the reported throughput covers a period when every client was
actually working.

Writes use the adapter's separate benchmark relationship type, so a concurrency
sweep never mutates the canonical graph.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from random import Random
from typing import Callable, Sequence

from benchmark import metrics
from benchmark.adapters.base import AdapterError, BaseGraphAdapter
from benchmark.config import Settings
from benchmark.dataset import Dataset, sample_node_ids, sample_start_nodes
from benchmark.metrics import Failure, Measurement, describe_error
from benchmark.workloads import MIXED_SEED_OFFSET

log = logging.getLogger(__name__)

WORKLOAD_NAME = "mixed read/write"

# How long a client waits for its peers before measuring on its own.
BARRIER_TIMEOUT_S = 120.0

AdapterFactory = Callable[[], BaseGraphAdapter]


@dataclass(frozen=True)
class Operation:
    """One unit of the mixed workload: a read, or a write of one edge."""

    is_write: bool
    first: int
    second: int


@dataclass
class ClientRun:
    """A single client's contribution to one concurrency level."""

    measurement: Measurement
    started_ns: int
    finished_ns: int


def build_operations(
    data: Dataset, settings: Settings, client_index: int, count: int
) -> list[Operation]:
    """Build one client's operation sequence, fixed by the seed.

    The read/write split is exact rather than probabilistic, then shuffled, so
    a run really does contain the configured ratio.
    """
    seed = settings.seed + MIXED_SEED_OFFSET + client_index
    rng = Random(seed)

    write_count = round(count * (1 - settings.read_ratio))
    flags = [True] * write_count + [False] * (count - write_count)
    rng.shuffle(flags)

    read_nodes = sample_start_nodes(data, count, seed=seed)
    write_sources = sample_node_ids(data, count, seed=seed + 1_000)
    write_targets = sample_node_ids(data, count, seed=seed + 2_000)

    return [
        Operation(is_write=flag, first=write_sources[index], second=write_targets[index])
        if flag
        else Operation(is_write=False, first=read_nodes[index], second=0)
        for index, flag in enumerate(flags)
    ]


def run_sweep(
    settings: Settings, data: Dataset, adapter_factory: AdapterFactory
) -> list[Measurement]:
    """Run the mixed workload once per configured concurrency level."""
    measurements = []
    for concurrency in settings.concurrency_levels:
        measurements.append(run_level(concurrency, settings, data, adapter_factory))
    return measurements


def run_level(
    concurrency: int, settings: Settings, data: Dataset, adapter_factory: AdapterFactory
) -> Measurement:
    """Run every client in parallel and report the level's sustained throughput."""
    operations = [
        build_operations(data, settings, client, settings.measured_iterations)
        for client in range(concurrency)
    ]
    write_share = sum(1 for operation in operations[0] if operation.is_write)
    log.info(
        "mixed workload at %d client(s): %d operations each, %d writes per client",
        concurrency,
        settings.measured_iterations,
        write_share,
    )

    barrier = threading.Barrier(concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        runs = list(
            pool.map(
                lambda client: _run_client(
                    client, operations[client], settings, adapter_factory, barrier
                ),
                range(concurrency),
            )
        )

    completed = [run for run in runs if run is not None]
    if not completed:
        # A tier that refuses this many clients is a result, not a reason to
        # discard the rest of the run.
        log.error(
            "mixed workload at %d client(s): no client could connect; recording the level as failed",
            concurrency,
        )
        return _refused_level(concurrency, settings)

    wall_time_s = (
        max(run.finished_ns for run in completed) - min(run.started_ns for run in completed)
    ) / metrics.NS_PER_S

    notes = [f"{concurrency} client(s), {settings.read_ratio:.0%} reads"]
    if len(completed) != concurrency:
        notes.append(f"only {len(completed)} of {concurrency} clients could connect")

    measurement = metrics.combine(
        WORKLOAD_NAME,
        [run.measurement for run in completed],
        concurrency=concurrency,
        wall_time_s=wall_time_s,
        notes=notes,
    )
    log.info(
        "mixed workload at %d client(s): %.1f ops/s, %d failures of %d",
        concurrency,
        measurement.ops_per_second,
        measurement.failure_count,
        measurement.attempted,
    )
    return measurement


def _refused_level(concurrency: int, settings: Settings) -> Measurement:
    """A level where the platform accepted no client at all."""
    attempted = concurrency * settings.measured_iterations
    return Measurement(
        name=WORKLOAD_NAME,
        attempted=attempted,
        warmup_iterations=settings.warmup_iterations,
        latencies_ms=(),
        failure_count=attempted,
        failure_samples=(
            Failure(iteration=0, error=f"no client could connect at {concurrency} clients"),
        ),
        wall_time_s=0.0,
        concurrency=concurrency,
        notes=[f"{concurrency} clients refused: the platform accepted no connection at this level"],
    )


def _run_client(
    client_index: int,
    operations: Sequence[Operation],
    settings: Settings,
    adapter_factory: AdapterFactory,
    barrier: threading.Barrier,
) -> ClientRun | None:
    """Connect, warm up, wait for the other clients, then run the measured window."""
    adapter = adapter_factory()
    try:
        adapter.connect()
    except AdapterError as exc:
        log.error("mixed client %d could not connect: %s", client_index, describe_error(exc))
        # Let the clients that did connect start instead of waiting for this one.
        _abort(barrier)
        return None

    try:
        for iteration in range(min(settings.warmup_iterations, len(operations))):
            try:
                _apply(adapter, operations[iteration])
            except AdapterError as exc:
                log.debug("mixed client %d warm-up failed: %s", client_index, describe_error(exc))

        _wait_for_other_clients(barrier, client_index)
        started_ns = time.perf_counter_ns()
        measurement = metrics.measure(
            f"{WORKLOAD_NAME} client {client_index}",
            lambda iteration: _apply(adapter, operations[iteration]),
            iterations=len(operations),
        )
        finished_ns = time.perf_counter_ns()
    finally:
        adapter.close()

    return ClientRun(measurement=measurement, started_ns=started_ns, finished_ns=finished_ns)


def _wait_for_other_clients(barrier: threading.Barrier, client_index: int) -> None:
    """Line the clients up so the measured window is genuinely concurrent.

    A broken barrier means some client failed; the survivors still measure, and
    the level records that it started unsynchronised.
    """
    try:
        barrier.wait(timeout=BARRIER_TIMEOUT_S)
    except threading.BrokenBarrierError:
        log.warning("mixed client %d starting unsynchronised", client_index)


def _apply(adapter: BaseGraphAdapter, operation: Operation) -> None:
    if operation.is_write:
        adapter.mixed_write(operation.first, operation.second)
    else:
        adapter.mixed_read(operation.first)


def _abort(barrier: threading.Barrier) -> None:
    try:
        barrier.abort()
    except Exception:  # already broken, nothing left to release
        pass

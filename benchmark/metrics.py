"""Timing and latency statistics.

Every workload is measured the same way: run ``warmup`` iterations that are
thrown away, then ``iterations`` measured ones, each timed with a monotonic
clock. Failed iterations are counted and reported instead of being dropped, so
a fast platform cannot look good by erroring out early.
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

log = logging.getLogger(__name__)

# Keep result files readable: count every failure, store only the first few messages.
MAX_FAILURE_SAMPLES = 10
MAX_ERROR_CHARS = 200

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


class MetricsError(Exception):
    """Raised when a statistic is requested that cannot be computed."""


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile: the smallest sample at or above ``pct`` percent.

    No interpolation, so every reported number is a latency that was actually
    observed.
    """
    if not values:
        raise MetricsError("cannot take a percentile of an empty sample")
    if not 0 < pct <= 100:
        raise MetricsError(f"percentile must be in (0, 100], got {pct}")
    ordered = sorted(values)
    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[rank - 1]


@dataclass(frozen=True)
class Failure:
    iteration: int
    error: str

    def to_dict(self) -> dict:
        return {"iteration": self.iteration, "error": self.error}


@dataclass(frozen=True)
class Measurement:
    """The outcome of one workload on one platform."""

    name: str
    attempted: int
    warmup_iterations: int
    latencies_ms: tuple[float, ...]
    failure_count: int = 0
    failure_samples: tuple[Failure, ...] = field(default_factory=tuple)
    wall_time_s: float = 0.0
    concurrency: int = 1
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def successes(self) -> int:
        return len(self.latencies_ms)

    @property
    def ops_per_second(self) -> float:
        if self.wall_time_s <= 0:
            return 0.0
        return self.successes * self.concurrency / self.wall_time_s

    def to_dict(self) -> dict:
        summary = {
            "workload": self.name,
            "concurrency": self.concurrency,
            "warmup_iterations": self.warmup_iterations,
            "attempted": self.attempted,
            "successes": self.successes,
            "failures": self.failure_count,
            "wall_time_s": round(self.wall_time_s, 4),
            "ops_per_second": round(self.ops_per_second, 2),
            "failure_samples": [failure.to_dict() for failure in self.failure_samples],
            "notes": list(self.notes),
        }
        if self.latencies_ms:
            summary |= {
                "p50_ms": _round(percentile(self.latencies_ms, 50)),
                "p95_ms": _round(percentile(self.latencies_ms, 95)),
                "p99_ms": _round(percentile(self.latencies_ms, 99)),
                "min_ms": _round(min(self.latencies_ms)),
                "max_ms": _round(max(self.latencies_ms)),
                "mean_ms": _round(sum(self.latencies_ms) / len(self.latencies_ms)),
            }
        return summary


def measure(
    name: str,
    operation: Callable[[int], object],
    *,
    iterations: int,
    warmup: int = 0,
    concurrency: int = 1,
    notes: Sequence[str] = (),
) -> Measurement:
    """Run ``operation(iteration)`` and return its latency profile.

    ``operation`` receives the iteration number so callers can vary parameters
    (a different start node per iteration, for example).
    """
    if iterations < 1:
        raise MetricsError("iterations must be at least 1")

    for iteration in range(warmup):
        try:
            operation(iteration)
        except Exception as exc:  # driver errors vary widely; warm-up must not abort a run
            log.warning("%s: warm-up iteration %d failed: %s", name, iteration, describe_error(exc))

    latencies_ms: list[float] = []
    failures: list[Failure] = []
    failure_count = 0

    run_started = time.perf_counter_ns()
    for iteration in range(iterations):
        started = time.perf_counter_ns()
        try:
            operation(iteration)
        except Exception as exc:  # counted, never silently skipped
            failure_count += 1
            if len(failures) < MAX_FAILURE_SAMPLES:
                failures.append(Failure(iteration, describe_error(exc)))
            continue
        latencies_ms.append((time.perf_counter_ns() - started) / NS_PER_MS)
    wall_time_s = (time.perf_counter_ns() - run_started) / NS_PER_S

    if failure_count:
        log.warning("%s: %d of %d iterations failed", name, failure_count, iterations)

    return Measurement(
        name=name,
        attempted=iterations,
        warmup_iterations=warmup,
        latencies_ms=tuple(latencies_ms),
        failure_count=failure_count,
        failure_samples=tuple(failures),
        wall_time_s=wall_time_s,
        concurrency=concurrency,
        notes=tuple(notes),
    )


class Stopwatch:
    """Monotonic wall-clock timer for one-off operations such as bulk loading."""

    def __init__(self) -> None:
        self._started_ns = time.perf_counter_ns()
        self._elapsed_ns: int | None = None

    def stop(self) -> None:
        self._elapsed_ns = time.perf_counter_ns() - self._started_ns

    @property
    def elapsed_s(self) -> float:
        elapsed = self._elapsed_ns
        if elapsed is None:
            elapsed = time.perf_counter_ns() - self._started_ns
        return elapsed / NS_PER_S


@contextmanager
def stopwatch() -> Iterator[Stopwatch]:
    watch = Stopwatch()
    try:
        yield watch
    finally:
        watch.stop()


def throughput(count: int, seconds: float) -> float:
    """Items per second, or 0.0 when the operation was too fast to time."""
    if seconds <= 0:
        return 0.0
    return count / seconds


def describe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    if len(text) <= MAX_ERROR_CHARS:
        return text
    return text[: MAX_ERROR_CHARS - 3] + "..."


def _round(value: float) -> float:
    return round(value, 3)

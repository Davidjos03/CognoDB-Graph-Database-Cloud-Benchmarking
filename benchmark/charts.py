"""PNG charts drawn from the summary tables.

Charts are a presentation of the same rows the CSVs contain, so a chart can
never disagree with a table. Latency charts plot p50 with a p95 marker, because
a bar chart of averages would hide exactly the tail behaviour this benchmark
cares about.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a benchmark machine or in CI

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

from benchmark.report import Summary  # noqa: E402

log = logging.getLogger(__name__)

LATENCY_PNG = "read-latency.png"
MIXED_PNG = "mixed-throughput.png"
INGEST_PNG = "ingest-throughput.png"

FIGURE_SIZE = (10, 5.5)
DPI = 150
BAR_GROUP_WIDTH = 0.8


def render(summary: Summary, charts_dir: Path) -> list[Path]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _read_latency(summary, charts_dir / LATENCY_PNG),
        _mixed_throughput(summary, charts_dir / MIXED_PNG),
        _ingest_throughput(summary, charts_dir / INGEST_PNG),
    ]
    return [path for path in written if path is not None]


def _read_latency(summary: Summary, path: Path) -> Path | None:
    rows = [row for row in summary.reads if row["p50_ms"] is not None]
    if not rows:
        log.warning("no read latencies to chart")
        return None

    workloads = _ordered(row["workload"] for row in rows)
    platforms = _ordered(row["platform"] for row in rows)
    positions = range(len(workloads))
    width = BAR_GROUP_WIDTH / len(platforms)

    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    for index, platform in enumerate(platforms):
        by_workload = {row["workload"]: row for row in rows if row["platform"] == platform}
        offsets = [position + index * width - BAR_GROUP_WIDTH / 2 + width / 2 for position in positions]
        p50 = [_value(by_workload, workload, "p50_ms") for workload in workloads]
        p95 = [_value(by_workload, workload, "p95_ms") for workload in workloads]

        axes.bar(offsets, p50, width=width, label=f"{platform} p50")
        axes.scatter(offsets, p95, marker="_", s=180, color="black", zorder=3)

    axes.scatter([], [], marker="_", s=180, color="black", label="p95")
    axes.set_xticks(list(positions))
    axes.set_xticklabels(workloads, rotation=20, ha="right")
    axes.set_ylabel("latency (ms, lower is better)")
    axes.set_title("Read workload latency: p50 bars, p95 markers")
    axes.grid(axis="y", alpha=0.3)
    axes.legend(fontsize="small")
    return _save(figure, path)


def _mixed_throughput(summary: Summary, path: Path) -> Path | None:
    if not summary.mixed:
        log.warning("no mixed workload results to chart")
        return None

    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    for platform in _ordered(row["platform"] for row in summary.mixed):
        rows = sorted(
            (row for row in summary.mixed if row["platform"] == platform),
            key=lambda row: row["concurrency"],
        )
        axes.plot(
            [row["concurrency"] for row in rows],
            [row["ops_per_second"] or 0.0 for row in rows],
            marker="o",
            label=platform,
        )
        for row in rows:
            if not row["successes"]:
                # A refused level is a result, so it is drawn rather than dropped.
                axes.annotate(
                    "refused",
                    (row["concurrency"], 0.0),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize="x-small",
                )

    axes.set_xlabel("concurrent clients")
    axes.set_ylabel("sustained ops/s (higher is better)")
    axes.set_title("Mixed 90/10 read/write throughput vs client concurrency")
    axes.grid(alpha=0.3)
    axes.legend(fontsize="small")
    return _save(figure, path)


def _ingest_throughput(summary: Summary, path: Path) -> Path | None:
    if not summary.ingest:
        log.warning("no ingest results to chart")
        return None

    platforms = [row["platform"] for row in summary.ingest]
    positions = range(len(platforms))
    width = 0.4

    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    axes.bar(
        [position - width / 2 for position in positions],
        [row["nodes_per_second"] for row in summary.ingest],
        width=width,
        label="nodes/s",
    )
    axes.bar(
        [position + width / 2 for position in positions],
        [row["relationships_per_second"] for row in summary.ingest],
        width=width,
        label="relationships/s",
    )
    axes.set_xticks(list(positions))
    axes.set_xticklabels(platforms)
    axes.set_ylabel("items per second (higher is better)")
    axes.set_title("Bulk load throughput, identical batched driver writes")
    axes.grid(axis="y", alpha=0.3)
    axes.legend(fontsize="small")
    return _save(figure, path)


def _ordered(values) -> list[str]:
    """Stable, de-duplicated order so colours stay consistent between charts."""
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def _value(rows: dict, workload: str, field: str) -> float:
    row = rows.get(workload)
    if row is None or row.get(field) is None:
        return 0.0
    return float(row[field])


def _save(figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    log.info("wrote %s", path)
    return path

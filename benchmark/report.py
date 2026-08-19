"""Turn raw result files into the tables that go in the README.

Nothing here computes a measurement; it only reads what a run wrote. That
separation is deliberate: a reviewer can delete every derived file, re-run
``gdbbench report`` and get byte-identical tables from the same raw JSON.

When a platform has been benchmarked more than once, the newest run supplies
the headline numbers and the older ones supply the run-to-run spread, so the
tables show how stable a figure is rather than implying one run is the truth.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from benchmark.config import NOT_OBSERVABLE

log = logging.getLogger(__name__)

SUMMARY_CSV = "summary.csv"
INGEST_CSV = "ingest.csv"
FOOTPRINT_CSV = "footprint.csv"
TABLES_MD = "tables.md"

MISSING = "—"

SUMMARY_FIELDS = (
    "platform",
    "category",
    "workload",
    "concurrency",
    "attempted",
    "successes",
    "failures",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "ops_per_second",
    "runs",
    "p50_spread_ms",
)

INGEST_FIELDS = (
    "platform",
    "measured_at",
    "node_count",
    "relationship_count",
    "node_load_s",
    "relationship_load_s",
    "total_load_s",
    "nodes_per_second",
    "relationships_per_second",
)

FOOTPRINT_FIELDS = (
    "platform",
    "configured_specs",
    "server",
    "indexes",
    "stored_data_size",
    "loaded_nodes",
    "loaded_relationships",
)


class ReportError(Exception):
    """Raised when there is nothing to report on."""


@dataclass(frozen=True)
class Summary:
    """Derived tables, one row per dict, ready for CSV or Markdown."""

    reads: list[dict]
    mixed: list[dict]
    ingest: list[dict]
    footprint: list[dict]
    caveats: list[tuple[str, str]]
    platforms: list[str]

    @property
    def rows(self) -> list[dict]:
        return self.reads + self.mixed


def summarise(runs: list[dict]) -> Summary:
    """Fold every run into per-platform tables."""
    if not runs:
        raise ReportError("no result files found; run `gdbbench benchmark` first")

    by_platform: dict[str, list[dict]] = {}
    for run in runs:
        by_platform.setdefault(run["metadata"]["platform"], []).append(run)

    reads: list[dict] = []
    mixed: list[dict] = []
    ingest: list[dict] = []
    footprint: list[dict] = []
    caveats: list[tuple[str, str]] = []

    for platform in sorted(by_platform):
        history = sorted(by_platform[platform], key=lambda run: run["metadata"]["started_at"])
        latest = history[-1]

        reads += _read_rows(platform, latest, history)
        mixed += _mixed_rows(platform, latest, history)
        loaded = _latest_ingest(history)
        if loaded is not None:
            ingest.append({"platform": platform, **loaded})
        footprint.append(_footprint_row(platform, latest))
        caveats += [(platform, caveat) for caveat in latest.get("caveats", [])]

    return Summary(
        reads=reads,
        mixed=mixed,
        ingest=ingest,
        footprint=footprint,
        caveats=caveats,
        platforms=sorted(by_platform),
    )


def _latest_ingest(history: list[dict]) -> dict | None:
    """Ingest from the newest run that actually loaded data.

    A ``--skip-load`` run measures queries against data loaded earlier, so its
    result file has no ingest section. Falling back to the last run that did
    load keeps the load figures in the table, and ``measured_at`` says which run
    they came from instead of implying they were measured alongside the queries.
    """
    for run in reversed(history):
        if run.get("ingest"):
            return {"measured_at": run["metadata"]["started_at"], **run["ingest"]}
    return None


def _read_rows(platform: str, latest: dict, history: list[dict]) -> list[dict]:
    rows = []
    for measurement in latest.get("workloads", []):
        name = measurement["workload"]
        rows.append(
            _row(
                platform,
                "read",
                name,
                measurement,
                spread=_spread(history, "workloads", name),
                runs=len(history),
            )
        )
    return rows


def _mixed_rows(platform: str, latest: dict, history: list[dict]) -> list[dict]:
    rows = []
    for measurement in sorted(latest.get("mixed", []), key=lambda item: item["concurrency"]):
        rows.append(
            _row(
                platform,
                "mixed",
                measurement["workload"],
                measurement,
                spread=_spread(history, "mixed", measurement["workload"], measurement["concurrency"]),
                runs=len(history),
            )
        )
    return rows


def _row(
    platform: str,
    category: str,
    workload: str,
    measurement: dict,
    *,
    spread: float | None,
    runs: int,
) -> dict:
    return {
        "platform": platform,
        "category": category,
        "workload": workload,
        "concurrency": measurement.get("concurrency", 1),
        "attempted": measurement.get("attempted", 0),
        "successes": measurement.get("successes", 0),
        "failures": measurement.get("failures", 0),
        "p50_ms": measurement.get("p50_ms"),
        "p95_ms": measurement.get("p95_ms"),
        "p99_ms": measurement.get("p99_ms"),
        "ops_per_second": measurement.get("ops_per_second"),
        "runs": runs,
        "p50_spread_ms": spread,
    }


def _spread(
    history: list[dict], section: str, workload: str, concurrency: int | None = None
) -> float | None:
    """Max minus min p50 for this workload across repeated runs.

    ``None`` when there is only one run to compare, which is honest about not
    knowing the variance rather than printing a reassuring zero.
    """
    values = [
        measurement["p50_ms"]
        for run in history
        for measurement in run.get(section, [])
        if measurement["workload"] == workload
        and "p50_ms" in measurement
        and (concurrency is None or measurement.get("concurrency") == concurrency)
    ]
    if len(values) < 2:
        return None
    return round(max(values) - min(values), 3)


def _footprint_row(platform: str, latest: dict) -> dict:
    observed = latest.get("footprint") or {}
    row = {"platform": platform}
    for field in FOOTPRINT_FIELDS[1:]:
        value = observed.get(field, NOT_OBSERVABLE)
        row[field] = ", ".join(str(item) for item in value) if isinstance(value, list) else value
    return row


def write_csvs(summary: Summary, results_dir: Path) -> list[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _write_csv(results_dir / SUMMARY_CSV, SUMMARY_FIELDS, summary.rows),
        _write_csv(results_dir / INGEST_CSV, INGEST_FIELDS, summary.ingest),
        _write_csv(results_dir / FOOTPRINT_CSV, FOOTPRINT_FIELDS, summary.footprint),
    ]
    return [path for path in written if path is not None]


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> Path | None:
    if not rows:
        log.warning("no rows for %s; not writing it", path.name)
        return None
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s (%d rows)", path, len(rows))
    return path


def write_markdown(summary: Summary, results_dir: Path) -> Path:
    """Emit the exact tables used in the README, so they are never hand-typed."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / TABLES_MD

    sections = [
        "<!-- Generated by `gdbbench report`. Do not edit by hand. -->",
        "## Ingest",
        _markdown_table(
            (
                "Platform",
                "Nodes",
                "Relationships",
                "Total load (s)",
                "Nodes/s",
                "Rels/s",
                "Measured (UTC)",
            ),
            [
                (
                    row["platform"],
                    f"{row['node_count']:,}",
                    f"{row['relationship_count']:,}",
                    f"{row['total_load_s']:.1f}",
                    f"{row['nodes_per_second']:,.0f}",
                    f"{row['relationships_per_second']:,.0f}",
                    row["measured_at"],
                )
                for row in summary.ingest
            ],
        ),
        "## Read workloads (p50 / p95, ms)",
        _markdown_table(
            ("Platform", "Workload", "p50", "p95", "p99", "Failures", "Runs", "p50 spread"),
            [
                (
                    row["platform"],
                    row["workload"],
                    _number(row["p50_ms"]),
                    _number(row["p95_ms"]),
                    _number(row["p99_ms"]),
                    f"{row['failures']}/{row['attempted']}",
                    str(row["runs"]),
                    _number(row["p50_spread_ms"]),
                )
                for row in summary.reads
            ],
        ),
        "## Mixed read/write sweep",
        _markdown_table(
            ("Platform", "Clients", "Ops/s", "p50", "p95", "Successes", "Failures"),
            [
                (
                    row["platform"],
                    str(row["concurrency"]),
                    _number(row["ops_per_second"]),
                    _number(row["p50_ms"]),
                    _number(row["p95_ms"]),
                    str(row["successes"]),
                    str(row["failures"]),
                )
                for row in summary.mixed
            ],
        ),
        "## Footprint",
        _markdown_table(
            ("Platform", "Configured specs", "Server", "Indexes", "Stored size"),
            [
                (
                    row["platform"],
                    row["configured_specs"],
                    row["server"],
                    row["indexes"],
                    row["stored_data_size"],
                )
                for row in summary.footprint
            ],
        ),
        "## Caveats recorded during the runs",
        "\n".join(f"- **{platform}**: {caveat}" for platform, caveat in summary.caveats)
        or "- none recorded",
    ]

    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    log.info("wrote %s", path)
    return path


def _markdown_table(headers: tuple[str, ...], rows: list[tuple]) -> str:
    if not rows:
        return "_No data yet._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _number(value: float | None) -> str:
    if value is None:
        return MISSING
    return f"{value:,.1f}"

"""Result records: one JSON file per platform per run.

Each file carries enough metadata (commit, environment, seed, dataset counts,
caveats) to tell where a number came from and under what conditions.
"""

from __future__ import annotations

import json
import logging
import platform as platform_info
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from benchmark.config import NOT_OBSERVABLE, Settings, Target
from benchmark.metrics import Measurement, throughput

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunMetadata:
    """Everything needed to reproduce, or fairly criticise, one run."""

    platform: str
    platform_specs: str
    uri: str
    started_at: str
    git_commit: str
    python_version: str
    operating_system: str
    seed: int
    warmup_iterations: int
    measured_iterations: int
    concurrency_levels: tuple[int, ...]
    read_ratio: float
    batch_size: int
    dataset: dict

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "platform_specs": self.platform_specs,
            "uri": self.uri,
            "started_at": self.started_at,
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "operating_system": self.operating_system,
            "seed": self.seed,
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations,
            "concurrency_levels": list(self.concurrency_levels),
            "read_ratio": self.read_ratio,
            "batch_size": self.batch_size,
            "dataset": self.dataset,
        }


def build_metadata(target: Target, settings: Settings, dataset_metadata: dict) -> RunMetadata:
    return RunMetadata(
        platform=target.name,
        platform_specs=target.specs,
        uri=target.safe_uri,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        python_version=sys.version.split()[0],
        operating_system=f"{platform_info.system()} {platform_info.release()}",
        seed=settings.seed,
        warmup_iterations=settings.warmup_iterations,
        measured_iterations=settings.measured_iterations,
        concurrency_levels=settings.concurrency_levels,
        read_ratio=settings.read_ratio,
        batch_size=settings.batch_size,
        dataset=dataset_metadata,
    )


@dataclass
class PlatformResult:
    """Collected measurements for one platform, ready to serialise."""

    metadata: RunMetadata
    ingest: dict | None = None
    workloads: list[Measurement] = field(default_factory=list)
    mixed: list[Measurement] = field(default_factory=list)
    footprint: dict = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def record_ingest(
        self, node_count: int, relationship_count: int, nodes_s: float, relationships_s: float
    ) -> None:
        """Store bulk-load timings, split so each rate is traceable to its phase."""
        self.ingest = {
            "node_count": node_count,
            "relationship_count": relationship_count,
            "node_load_s": round(nodes_s, 3),
            "relationship_load_s": round(relationships_s, 3),
            "total_load_s": round(nodes_s + relationships_s, 3),
            "nodes_per_second": round(throughput(node_count, nodes_s), 2),
            "relationships_per_second": round(throughput(relationship_count, relationships_s), 2),
        }

    def record_workload(self, measurement: Measurement) -> None:
        self.workloads.append(measurement)

    def record_mixed(self, measurement: Measurement) -> None:
        self.mixed.append(measurement)

    def record_footprint(self, footprint: dict | None) -> None:
        self.footprint = footprint or {"status": NOT_OBSERVABLE}

    def add_caveat(self, caveat: str) -> None:
        """Record a limitation. Hidden caveats are worse than ugly ones."""
        if caveat not in self.caveats:
            self.caveats.append(caveat)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "metadata": self.metadata.to_dict(),
            "ingest": self.ingest,
            "workloads": [measurement.to_dict() for measurement in self.workloads],
            "mixed": [measurement.to_dict() for measurement in self.mixed],
            "footprint": self.footprint or {"status": NOT_OBSERVABLE},
            "caveats": list(self.caveats),
        }

    def save(self, raw_results_dir: Path) -> Path:
        raw_results_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.metadata.started_at.replace(":", "").replace("-", "")
        path = raw_results_dir / f"{self.metadata.platform}-{stamp}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        log.info("wrote %s", path)
        return path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_all(raw_results_dir: Path) -> list[dict]:
    """Read every result file, oldest first."""
    return [load(path) for path in sorted(raw_results_dir.glob("*.json"))]


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"

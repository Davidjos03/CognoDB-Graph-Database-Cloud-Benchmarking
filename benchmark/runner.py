"""Run one platform end to end: load, measure every workload, collect footprint."""

from __future__ import annotations

import logging

from benchmark import adapters, loader, metrics, results, workloads
from benchmark.config import Settings, Target
from benchmark.dataset import Dataset

log = logging.getLogger(__name__)


def run_platform(
    target: Target,
    settings: Settings,
    data: Dataset,
    dataset_metadata: dict,
    *,
    load: bool = True,
) -> results.PlatformResult:
    """Benchmark one platform and return its unsaved result."""
    result = results.PlatformResult(
        metadata=results.build_metadata(target, settings, dataset_metadata)
    )
    adapter = adapters.create(target, settings)

    with adapter:
        if load:
            report = loader.load_into(adapter, data)
            result.record_ingest(
                node_count=report.node_count,
                relationship_count=report.relationship_count,
                nodes_s=report.node_seconds,
                relationships_s=report.relationship_seconds,
            )
        else:
            result.add_caveat(
                "load phase skipped: ingest metrics come from a previous run of this platform"
            )

        for workload in workloads.build_read_workloads(adapter, data, settings):
            measurement = metrics.measure(
                workload.name,
                workload.operation,
                iterations=settings.measured_iterations,
                warmup=settings.warmup_iterations,
            )
            result.record_workload(measurement)
            _log_measurement(target.name, measurement)

        result.record_footprint(adapter.footprint())

    for caveat in adapter.caveats:
        result.add_caveat(caveat)
    return result


def _log_measurement(platform: str, measurement: metrics.Measurement) -> None:
    summary = measurement.to_dict()
    if measurement.successes:
        log.info(
            "%s: %s p50=%.1fms p95=%.1fms failures=%d/%d",
            platform,
            measurement.name,
            summary["p50_ms"],
            summary["p95_ms"],
            summary["failures"],
            summary["attempted"],
        )
    else:
        log.error(
            "%s: %s produced no successful iteration (%d failures)",
            platform,
            measurement.name,
            summary["failures"],
        )

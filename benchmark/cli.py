"""Command line entry point for the benchmark harness."""

from __future__ import annotations

import argparse
import logging
import sys

from benchmark import __version__, config, dataset
from benchmark.log import setup_logging

log = logging.getLogger("benchmark")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gdbbench",
        description="Benchmark CognoDB Cloud against other graph databases on identical data and workloads.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true", help="log at DEBUG level")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check configuration only: never connect to a database and never change data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="report the configured targets and benchmark settings"
    )
    validate.add_argument(
        "--targets",
        nargs="+",
        metavar="PLATFORM",
        help=f"require these platforms to be configured ({', '.join(config.PLATFORMS)})",
    )
    validate.set_defaults(handler=cmd_validate)

    download = subparsers.add_parser(
        "download-data",
        help=f"download {dataset.DATASET_NAME} and export the shared nodes/edges CSV files",
    )
    download.add_argument(
        "--force", action="store_true", help="re-download even if the archive is cached"
    )
    download.set_defaults(handler=cmd_download_data)

    return parser


def cmd_validate(args: argparse.Namespace) -> int:
    """Print settings and configured targets without connecting anywhere."""
    settings = config.load_settings()
    targets = config.load_targets(args.targets)

    log.info("seed=%d warmup=%d measured=%d", settings.seed, settings.warmup_iterations, settings.measured_iterations)
    log.info(
        "concurrency=%s read_ratio=%.2f batch_size=%d timeout=%.1fs",
        ",".join(str(level) for level in settings.concurrency_levels),
        settings.read_ratio,
        settings.batch_size,
        settings.query_timeout_s,
    )
    log.info("data_dir=%s results_dir=%s", settings.data_dir, settings.results_dir)

    if not targets:
        log.error("no platform is configured; copy .env.example to .env and fill in at least one URI")
        return 1

    for target in targets:
        log.info("configured: %s specs=%s", target.summary(), target.specs)
    for name in config.unconfigured_platforms():
        log.warning("not configured: %s (set %s_URI to include it)", name, name.upper())

    try:
        metadata = dataset.read_metadata(settings.data_dir)
        log.info(
            "dataset ready: %s, %d nodes, %d relationships",
            metadata["name"],
            metadata["node_count"],
            metadata["relationship_count"],
        )
    except dataset.DatasetError as exc:
        log.warning("dataset not prepared: %s", exc)
    return 0


def cmd_download_data(args: argparse.Namespace) -> int:
    """Download the source dataset and export the CSV files used by every platform."""
    settings = config.load_settings()
    if args.dry_run:
        log.info("would download %s into %s", dataset.SOURCE_URL, settings.data_dir / "raw")
        return 0

    archive = dataset.download_archive(settings.data_dir, force=args.force)
    parsed = dataset.read_archive(archive)
    dataset.validate(parsed)

    nodes_csv, edges_csv = dataset.export_csv(parsed, settings.data_dir)
    metadata_path = dataset.write_metadata(parsed, archive, settings.data_dir)

    log.info(
        "%s: %d nodes, %d relationships (%d duplicate edges skipped)",
        dataset.DATASET_NAME,
        parsed.node_count,
        parsed.relationship_count,
        parsed.duplicate_edges,
    )
    log.info("wrote %s, %s and %s", nodes_csv.name, edges_csv.name, metadata_path.name)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    if args.dry_run:
        log.info("dry run: no database connection will be opened and no data will change")
    try:
        return args.handler(args)
    except config.ConfigError as exc:
        log.error("configuration error: %s", exc)
        return 2
    except dataset.DatasetError as exc:
        log.error("dataset error: %s", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())

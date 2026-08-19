"""Command line entry point for the benchmark harness."""

from __future__ import annotations

import argparse
import logging
import sys

from benchmark import __version__, config
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


if __name__ == "__main__":
    sys.exit(main())

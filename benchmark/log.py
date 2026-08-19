"""Logging setup shared by every command."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=_FORMAT,
        datefmt="%H:%M:%S",
    )
    # Driver internals are noisy at DEBUG and drown out the benchmark output.
    logging.getLogger("neo4j").setLevel(logging.WARNING)

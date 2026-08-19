"""The benchmark dataset: SNAP wiki-Vote, downloaded once and exported as CSV.

Every platform is loaded from the same two CSV files, so the graph is identical
everywhere. Parsing is deterministic: node ids keep their original SNAP values,
edges and nodes are sorted, and ``group_id`` is derived from the node id.

Graph model used by every adapter:
    (:User {node_id, group_id})-[:VOTED_FOR]->(:User)
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable, Iterator

log = logging.getLogger(__name__)

DATASET_NAME = "SNAP wiki-Vote"
SOURCE_URL = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"
ARCHIVE_NAME = "wiki-Vote.txt.gz"

NODE_LABEL = "User"
RELATIONSHIP_TYPE = "VOTED_FOR"

# The assignment requires at least 100k relationships; wiki-Vote has 103,689.
MIN_RELATIONSHIPS = 100_000

# group_id = node_id % GROUP_COUNT. It gives the filtered lookup and the
# aggregation workload a low-cardinality indexed property without inventing
# data that is not in the source.
GROUP_COUNT = 50

NODES_CSV = "nodes.csv"
EDGES_CSV = "edges.csv"
METADATA_JSON = "dataset.json"


class DatasetError(Exception):
    """Raised when the dataset is missing, malformed or too small."""


@dataclass(frozen=True)
class Dataset:
    """A parsed directed graph, in deterministic order."""

    node_ids: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    duplicate_edges: int = 0

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def relationship_count(self) -> int:
        return len(self.edges)

    def nodes(self) -> Iterator[tuple[int, int]]:
        """Yield ``(node_id, group_id)`` pairs in ascending node id order."""
        for node_id in self.node_ids:
            yield node_id, group_id(node_id)


def group_id(node_id: int) -> int:
    return node_id % GROUP_COUNT


def parse_edge_list(lines: Iterable[str]) -> Dataset:
    """Parse a SNAP edge list (``# comment`` lines, then ``source<TAB>target``)."""
    node_ids: set[int] = set()
    edges: set[tuple[int, int]] = set()
    duplicates = 0

    for lineno, line in enumerate(lines, start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) != 2:
            raise DatasetError(f"line {lineno}: expected two node ids, got {text!r}")
        try:
            source, target = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise DatasetError(f"line {lineno}: node ids must be integers, got {text!r}") from exc

        node_ids.update((source, target))
        if (source, target) in edges:
            duplicates += 1
        else:
            edges.add((source, target))

    return Dataset(
        node_ids=tuple(sorted(node_ids)),
        edges=tuple(sorted(edges)),
        duplicate_edges=duplicates,
    )


def validate(dataset: Dataset) -> None:
    """Fail loudly if the dataset is too small for the assignment's rules."""
    if dataset.relationship_count < MIN_RELATIONSHIPS:
        raise DatasetError(
            f"dataset has {dataset.relationship_count} relationships, "
            f"at least {MIN_RELATIONSHIPS} are required"
        )
    if dataset.node_count == 0:
        raise DatasetError("dataset has no nodes")


def sample_start_nodes(dataset: Dataset, count: int, seed: int) -> list[int]:
    """Pick traversal start nodes deterministically from the same seed.

    Only nodes with at least one outgoing edge are eligible, so a traversal
    measures graph work rather than empty results.
    """
    candidates = sorted({source for source, _ in dataset.edges})
    if not candidates:
        raise DatasetError("dataset has no node with an outgoing relationship")
    return _sample(candidates, count, seed)


def sample_node_ids(dataset: Dataset, count: int, seed: int) -> list[int]:
    """Pick nodes for point lookups, including those with no outgoing edges."""
    if not dataset.node_ids:
        raise DatasetError("dataset has no nodes")
    return _sample(list(dataset.node_ids), count, seed)


def sample_group_ids(count: int, seed: int) -> list[int]:
    """Pick ``group_id`` values for the filtered lookup workload."""
    return _sample(list(range(GROUP_COUNT)), count, seed)


def _sample(candidates: list[int], count: int, seed: int) -> list[int]:
    if count < 1:
        raise DatasetError("count must be at least 1")
    rng = Random(seed)
    if count <= len(candidates):
        return rng.sample(candidates, count)
    # Fewer candidates than requested iterations: sample with replacement.
    return [rng.choice(candidates) for _ in range(count)]


def download_archive(data_dir: Path, force: bool = False) -> Path:
    """Download the source archive into ``data/raw``, reusing any cached copy."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / ARCHIVE_NAME

    if archive.exists() and not force:
        log.info("using cached archive %s", archive)
        return archive

    log.info("downloading %s", SOURCE_URL)
    try:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            with archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except OSError as exc:
        archive.unlink(missing_ok=True)
        raise DatasetError(f"could not download {SOURCE_URL}: {exc}") from exc
    return archive


def read_archive(archive: Path) -> Dataset:
    try:
        with gzip.open(archive, "rt", encoding="utf-8") as handle:
            return parse_edge_list(handle)
    except OSError as exc:
        raise DatasetError(f"could not read {archive}: {exc}") from exc


def export_csv(dataset: Dataset, data_dir: Path) -> tuple[Path, Path]:
    """Write the nodes and edges CSV files every platform is loaded from."""
    data_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = data_dir / NODES_CSV
    edges_path = data_dir / EDGES_CSV

    _write_rows(nodes_path, ("node_id", "group_id"), dataset.nodes())
    _write_rows(edges_path, ("source", "target"), dataset.edges)
    return nodes_path, edges_path


def write_metadata(dataset: Dataset, archive: Path, data_dir: Path) -> Path:
    """Record provenance and counts so a run can be traced back to the source."""
    metadata = {
        "name": DATASET_NAME,
        "source_url": SOURCE_URL,
        "archive_sha256": file_sha256(archive),
        "node_count": dataset.node_count,
        "relationship_count": dataset.relationship_count,
        "duplicate_relationships_skipped": dataset.duplicate_edges,
        "node_label": NODE_LABEL,
        "relationship_type": RELATIONSHIP_TYPE,
        "group_count": GROUP_COUNT,
    }
    path = data_dir / METADATA_JSON
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def read_metadata(data_dir: Path) -> dict:
    path = data_dir / METADATA_JSON
    if not path.exists():
        raise DatasetError(f"{path} is missing; run `gdbbench download-data` first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(data_dir: Path) -> Dataset:
    """Read back the exported CSV files, so loaders never re-parse the archive."""
    nodes_path = data_dir / NODES_CSV
    edges_path = data_dir / EDGES_CSV
    for path in (nodes_path, edges_path):
        if not path.exists():
            raise DatasetError(f"{path} is missing; run `gdbbench download-data` first")

    node_ids = tuple(int(row["node_id"]) for row in _read_rows(nodes_path))
    edges = tuple((int(row["source"]), int(row["target"])) for row in _read_rows(edges_path))
    return Dataset(node_ids=node_ids, edges=edges)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path: Path, header: tuple[str, ...], rows: Iterable[tuple[int, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _read_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)

import pytest

from benchmark import dataset

EDGE_LIST = """\
# Directed graph: wiki-Vote.txt
# Nodes: 4 Edges: 4
# FromNodeId\tToNodeId
30\t1412
30\t3352
3\t28
3\t28
"""


def parse(text: str) -> dataset.Dataset:
    return dataset.parse_edge_list(text.splitlines())


def test_parse_skips_comments_and_deduplicates_edges():
    parsed = parse(EDGE_LIST)

    assert parsed.node_ids == (3, 28, 30, 1412, 3352)
    assert parsed.edges == ((3, 28), (30, 1412), (30, 3352))
    assert parsed.duplicate_edges == 1
    assert parsed.node_count == 5
    assert parsed.relationship_count == 3


def test_parse_is_deterministic_regardless_of_input_order():
    shuffled = "\n".join(reversed(EDGE_LIST.strip().splitlines()))

    assert parse(shuffled).edges == parse(EDGE_LIST).edges


def test_nodes_carry_derived_group_id():
    parsed = parse(EDGE_LIST)

    assert list(parsed.nodes()) == [
        (3, 3),
        (28, 28),
        (30, 30),
        (1412, 1412 % dataset.GROUP_COUNT),
        (3352, 3352 % dataset.GROUP_COUNT),
    ]


@pytest.mark.parametrize("line", ["30", "30\t1412\t7", "30\tabc"])
def test_parse_rejects_malformed_lines(line):
    with pytest.raises(dataset.DatasetError):
        parse(line + "\n")


def test_validate_rejects_dataset_below_the_required_relationship_count():
    with pytest.raises(dataset.DatasetError, match="at least 100000"):
        dataset.validate(parse(EDGE_LIST))


def test_validate_accepts_a_dataset_at_the_threshold():
    edges = tuple((source, source + 1) for source in range(dataset.MIN_RELATIONSHIPS))
    big = dataset.Dataset(node_ids=tuple(range(dataset.MIN_RELATIONSHIPS + 1)), edges=edges)

    dataset.validate(big)


def test_sampling_is_reproducible_for_a_given_seed():
    parsed = parse(EDGE_LIST)

    assert dataset.sample_start_nodes(parsed, 2, seed=42) == dataset.sample_start_nodes(
        parsed, 2, seed=42
    )


def test_sampling_only_returns_nodes_with_outgoing_edges():
    parsed = parse(EDGE_LIST)

    assert set(dataset.sample_start_nodes(parsed, 2, seed=42)) <= {3, 30}


def test_sampling_more_nodes_than_candidates_samples_with_replacement():
    parsed = parse(EDGE_LIST)

    sampled = dataset.sample_start_nodes(parsed, 5, seed=7)

    assert len(sampled) == 5
    assert set(sampled) <= {3, 30}


def test_csv_export_round_trips_through_load(tmp_path):
    parsed = parse(EDGE_LIST)

    dataset.export_csv(parsed, tmp_path)
    reloaded = dataset.load_csv(tmp_path)

    assert reloaded.node_ids == parsed.node_ids
    assert reloaded.edges == parsed.edges


def test_load_csv_reports_a_missing_export(tmp_path):
    with pytest.raises(dataset.DatasetError, match="download-data"):
        dataset.load_csv(tmp_path)


def test_metadata_records_provenance_and_counts(tmp_path):
    parsed = parse(EDGE_LIST)
    archive = tmp_path / "archive.gz"
    archive.write_bytes(b"not a real archive")

    dataset.write_metadata(parsed, archive, tmp_path)
    metadata = dataset.read_metadata(tmp_path)

    assert metadata["source_url"] == dataset.SOURCE_URL
    assert metadata["relationship_count"] == 3
    assert metadata["duplicate_relationships_skipped"] == 1
    assert metadata["archive_sha256"] == dataset.file_sha256(archive)

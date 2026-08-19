from pathlib import Path

import pytest

from benchmark import metrics, results
from benchmark.config import Settings, Target

TARGET = Target(
    name="cognodb",
    kind="bolt",
    uri="bolt+s://secret-host.example.com",
    username="cognodb",
    password="secret",
    database="neo4j",
    specs="free c0, 0.5 vCPU / 512 MB / 1 GB",
)

SETTINGS = Settings(
    seed=42,
    warmup_iterations=20,
    measured_iterations=100,
    concurrency_levels=(1, 10, 40),
    read_ratio=0.9,
    batch_size=1000,
    query_timeout_s=30.0,
    data_dir=Path("data"),
    results_dir=Path("results"),
)

DATASET = {"name": "SNAP wiki-Vote", "node_count": 7115, "relationship_count": 103689}


@pytest.fixture
def result() -> results.PlatformResult:
    metadata = results.build_metadata(TARGET, SETTINGS, DATASET)
    return results.PlatformResult(metadata=metadata)


def test_metadata_captures_environment_and_settings(result):
    metadata = result.metadata.to_dict()

    assert metadata["platform"] == "cognodb"
    assert metadata["seed"] == 42
    assert metadata["measured_iterations"] == 100
    assert metadata["concurrency_levels"] == [1, 10, 40]
    assert metadata["dataset"]["relationship_count"] == 103689
    assert metadata["python_version"].startswith("3.")
    assert metadata["started_at"].endswith("+00:00")
    assert metadata["operating_system"]
    assert metadata["git_commit"]


def test_metadata_never_stores_the_real_host(result):
    assert result.metadata.to_dict()["uri"] == "bolt+s://***"


def test_ingest_reports_separate_node_and_relationship_rates(result):
    result.record_ingest(node_count=100, relationship_count=1000, nodes_s=2.0, relationships_s=5.0)

    assert result.ingest == {
        "node_count": 100,
        "relationship_count": 1000,
        "node_load_s": 2.0,
        "relationship_load_s": 5.0,
        "total_load_s": 7.0,
        "nodes_per_second": 50.0,
        "relationships_per_second": 200.0,
    }


def test_footprint_falls_back_to_not_observable(result):
    result.record_footprint(None)

    assert result.to_dict()["footprint"] == {"status": "not observable"}


def test_caveats_are_recorded_once(result):
    result.add_caveat("free tier throttling observed")
    result.add_caveat("free tier throttling observed")

    assert result.to_dict()["caveats"] == ["free tier throttling observed"]


def test_serialised_result_has_the_expected_shape(result):
    result.record_workload(
        metrics.Measurement(
            name="1-hop", attempted=2, warmup_iterations=1, latencies_ms=(1.0, 3.0)
        )
    )
    result.record_mixed(
        metrics.Measurement(
            name="mixed", attempted=2, warmup_iterations=0, latencies_ms=(2.0,), concurrency=10
        )
    )

    payload = result.to_dict()

    assert payload["schema_version"] == results.SCHEMA_VERSION
    assert set(payload) == {
        "schema_version",
        "metadata",
        "ingest",
        "workloads",
        "mixed",
        "footprint",
        "caveats",
    }
    assert payload["workloads"][0]["workload"] == "1-hop"
    assert payload["mixed"][0]["concurrency"] == 10


def test_save_and_load_round_trip(tmp_path, result):
    result.record_footprint({"stored_data_size_mb": 12.5})

    path = result.save(tmp_path)

    assert path.parent == tmp_path
    assert path.name.startswith("cognodb-")
    assert path.name.endswith("Z.json")  # UTC stamp, safe on every filesystem
    assert results.load(path) == result.to_dict()
    assert results.load_all(tmp_path) == [result.to_dict()]

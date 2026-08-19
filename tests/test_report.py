"""The report must present the raw files faithfully, including their bad news."""

from __future__ import annotations

import csv

import pytest

from benchmark import report


def run(platform: str, started_at: str, *, p50: float, ingest: bool = True) -> dict:
    return {
        "schema_version": 1,
        "metadata": {
            "platform": platform,
            "platform_specs": "free tier",
            "started_at": started_at,
        },
        "ingest": {
            "node_count": 7115,
            "relationship_count": 103689,
            "node_load_s": 5.0,
            "relationship_load_s": 45.0,
            "total_load_s": 50.0,
            "nodes_per_second": 1423.0,
            "relationships_per_second": 2304.2,
        }
        if ingest
        else None,
        "workloads": [
            {
                "workload": "1-hop traversal",
                "concurrency": 1,
                "attempted": 100,
                "successes": 100,
                "failures": 0,
                "p50_ms": p50,
                "p95_ms": p50 + 10,
                "p99_ms": p50 + 20,
                "ops_per_second": 2.7,
                "warmup": {"first_ms": p50 + 900, "p50_ms": p50 + 5, "max_ms": p50 + 900},
            }
        ],
        "mixed": [
            {
                "workload": "mixed 90/10 read/write",
                "concurrency": 10,
                "attempted": 1000,
                "successes": 1000,
                "failures": 0,
                "p50_ms": 380.0,
                "p95_ms": 420.0,
                "ops_per_second": 26.2,
            },
            {
                "workload": "mixed 90/10 read/write",
                "concurrency": 40,
                "attempted": 4000,
                "successes": 0,
                "failures": 4000,
                "ops_per_second": 0.0,
            },
        ],
        "footprint": {
            "configured_specs": "free c0",
            "server": "CognoDB/1.0",
            "indexes": ["User(node_id)", "User(group_id)"],
        },
        "caveats": ["40 clients refused: the platform accepted no connection at this level"],
    }


def test_summarise_refuses_to_invent_a_report_from_nothing():
    with pytest.raises(report.ReportError, match="no result files"):
        report.summarise([])


def test_latest_run_supplies_the_headline_numbers():
    summary = report.summarise(
        [
            run("cognodb", "2026-08-19T10:00:00+00:00", p50=400.0),
            run("cognodb", "2026-08-19T12:00:00+00:00", p50=360.0),
        ]
    )

    assert [row["p50_ms"] for row in summary.reads] == [360.0]


def test_repeated_runs_report_their_spread_and_a_single_run_does_not():
    two_runs = report.summarise(
        [
            run("cognodb", "2026-08-19T10:00:00+00:00", p50=400.0),
            run("cognodb", "2026-08-19T12:00:00+00:00", p50=360.0),
        ]
    )
    one_run = report.summarise([run("cognodb", "2026-08-19T10:00:00+00:00", p50=400.0)])

    assert two_runs.reads[0]["p50_spread_ms"] == 40.0
    assert two_runs.reads[0]["runs"] == 2
    # None, not 0.0: one run says nothing about variance.
    assert one_run.reads[0]["p50_spread_ms"] is None


def test_a_refused_concurrency_level_stays_in_the_tables():
    summary = report.summarise([run("cognodb", "2026-08-19T10:00:00+00:00", p50=360.0)])
    refused = [row for row in summary.mixed if row["concurrency"] == 40]

    assert refused[0]["successes"] == 0
    assert refused[0]["failures"] == 4000
    assert refused[0]["p50_ms"] is None


def test_platforms_are_compared_side_by_side():
    summary = report.summarise(
        [
            run("cognodb", "2026-08-19T10:00:00+00:00", p50=360.0),
            run("neo4j", "2026-08-19T11:00:00+00:00", p50=12.0),
        ]
    )

    assert summary.platforms == ["cognodb", "neo4j"]
    assert len(summary.ingest) == 2


def test_a_platform_without_an_ingest_run_still_reports_its_reads():
    summary = report.summarise(
        [run("cognodb", "2026-08-19T10:00:00+00:00", p50=360.0, ingest=False)]
    )

    assert summary.ingest == []
    assert summary.reads


def test_a_skip_load_run_keeps_the_load_figures_of_the_run_that_measured_them():
    summary = report.summarise(
        [
            run("cognodb", "2026-08-19T10:00:00+00:00", p50=400.0),
            run("cognodb", "2026-08-19T12:00:00+00:00", p50=360.0, ingest=False),
        ]
    )

    assert len(summary.ingest) == 1
    assert summary.ingest[0]["measured_at"] == "2026-08-19T10:00:00+00:00"


def test_csvs_hold_every_summarised_row(tmp_path):
    summary = report.summarise([run("cognodb", "2026-08-19T10:00:00+00:00", p50=360.0)])
    written = report.write_csvs(summary, tmp_path)

    assert {path.name for path in written} == {
        report.SUMMARY_CSV,
        report.INGEST_CSV,
        report.FOOTPRINT_CSV,
    }
    rows = list(csv.DictReader((tmp_path / report.SUMMARY_CSV).open(encoding="utf-8")))
    assert len(rows) == len(summary.rows)
    assert rows[0]["platform"] == "cognodb"


def test_footprint_csv_flattens_the_index_list(tmp_path):
    summary = report.summarise([run("cognodb", "2026-08-19T10:00:00+00:00", p50=360.0)])
    report.write_csvs(summary, tmp_path)

    rows = list(csv.DictReader((tmp_path / report.FOOTPRINT_CSV).open(encoding="utf-8")))
    assert rows[0]["indexes"] == "User(node_id), User(group_id)"
    assert rows[0]["stored_data_size"] == "not observable"


def test_warm_up_cost_is_reported_separately_from_measured_latency():
    summary = report.summarise([run("cognodb", "2026-08-19T10:00:00+00:00", p50=360.0)])
    row = summary.reads[0]

    # The first call is reported, never blended into the measured percentiles.
    assert row["warmup_first_ms"] == 1260.0
    assert row["p50_ms"] == 360.0


def test_an_older_result_without_warm_up_data_still_reports(tmp_path):
    payload = run("cognodb", "2026-08-19T10:00:00+00:00", p50=360.0)
    del payload["workloads"][0]["warmup"]

    summary = report.summarise([payload])
    report.write_markdown(summary, tmp_path)

    assert summary.reads[0]["warmup_first_ms"] is None


def test_markdown_tables_carry_the_caveats(tmp_path):
    summary = report.summarise([run("cognodb", "2026-08-19T10:00:00+00:00", p50=360.0)])
    text = report.write_markdown(summary, tmp_path).read_text(encoding="utf-8")

    assert "## Read workloads" in text
    assert "40 clients refused" in text
    assert "Do not edit by hand" in text


def test_charts_are_written_for_every_section(tmp_path):
    from benchmark import charts

    summary = report.summarise([run("cognodb", "2026-08-19T10:00:00+00:00", p50=360.0)])
    written = charts.render(summary, tmp_path)

    assert {path.name for path in written} == {
        charts.LATENCY_PNG,
        charts.MIXED_PNG,
        charts.INGEST_PNG,
    }
    assert all(path.stat().st_size > 0 for path in written)

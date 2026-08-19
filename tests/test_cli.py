import logging

import pytest

from benchmark import cli, config, dataset

SETTING_VARS = (
    "BENCH_SEED",
    "BENCH_WARMUP_ITERATIONS",
    "BENCH_MEASURED_ITERATIONS",
    "BENCH_CONCURRENCY_LEVELS",
    "BENCH_READ_RATIO",
    "BENCH_BATCH_SIZE",
    "BENCH_QUERY_TIMEOUT_S",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Run the CLI against an empty environment, ignoring the developer's .env."""
    monkeypatch.setattr(config, "load_env", lambda: None)
    for platform in config.PLATFORMS:
        for suffix in ("URI", "USERNAME", "PASSWORD", "DATABASE", "SPECS"):
            monkeypatch.delenv(f"{platform.upper()}_{suffix}", raising=False)
    for name in SETTING_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BENCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BENCH_RESULTS_DIR", str(tmp_path / "results"))


def test_validate_fails_when_no_platform_is_configured():
    assert cli.main(["validate"]) == 1


def test_validate_succeeds_with_one_configured_platform(monkeypatch):
    monkeypatch.setenv("COGNODB_URI", "bolt+s://host.example.com")

    assert cli.main(["validate"]) == 0


def test_validate_never_logs_the_real_host(monkeypatch, caplog):
    monkeypatch.setenv("COGNODB_URI", "bolt+s://host.example.com")
    monkeypatch.setenv("COGNODB_PASSWORD", "super-secret")
    caplog.set_level(logging.INFO)

    cli.main(["validate"])

    logged = caplog.text
    assert "bolt+s://***" in logged
    assert "host.example.com" not in logged
    assert "super-secret" not in logged


def test_validate_warns_when_the_dataset_is_missing(caplog, monkeypatch):
    monkeypatch.setenv("COGNODB_URI", "bolt+s://host.example.com")

    cli.main(["validate"])

    assert "dataset not prepared" in caplog.text


def test_requesting_an_unconfigured_platform_is_a_configuration_error():
    assert cli.main(["validate", "--targets", "neo4j"]) == 2


def test_unknown_platform_is_a_configuration_error():
    assert cli.main(["validate", "--targets", "sqlite"]) == 2


def test_unusable_setting_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv("BENCH_SEED", "not-a-number")

    assert cli.main(["validate"]) == 2


def test_download_data_dry_run_does_not_touch_the_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("dry run must not download anything")

    monkeypatch.setattr(dataset, "download_archive", fail)

    assert cli.main(["--dry-run", "download-data"]) == 0


def test_download_failure_exits_with_the_dataset_error_code(monkeypatch):
    def unavailable(*args, **kwargs):
        raise dataset.DatasetError("source is unreachable")

    monkeypatch.setattr(dataset, "download_archive", unavailable)

    assert cli.main(["download-data"]) == 3


def test_unknown_command_is_rejected_by_the_parser():
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["benchmark-everything"])

    assert exit_info.value.code == 2

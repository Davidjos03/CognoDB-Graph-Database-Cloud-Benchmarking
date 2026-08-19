"""Configuration read from the environment.

Every credential and benchmark setting comes from environment variables (or a
git-ignored .env file) so that no connection detail is ever committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Env prefix -> adapter kind. One entry per benchmarked platform.
PLATFORMS: dict[str, str] = {
    "cognodb": "bolt",
    "neo4j": "bolt",
    "memgraph": "memgraph",  # Bolt, but its own index syntax
    "falkordb": "falkordb",
    "arangodb": "arangodb",
}

NOT_OBSERVABLE = "not observable"


class ConfigError(Exception):
    """Raised when the environment is missing or holds an unusable value."""


@dataclass(frozen=True)
class Target:
    """One database under test."""

    name: str
    kind: str
    uri: str
    username: str
    password: str
    database: str
    specs: str

    @property
    def safe_uri(self) -> str:
        """URI with the host masked, safe to log or store in results."""
        scheme, _, _ = self.uri.partition("://")
        return f"{scheme}://***"

    def summary(self) -> str:
        return f"{self.name} (kind={self.kind}, uri={self.safe_uri}, database={self.database})"


@dataclass(frozen=True)
class Settings:
    """Benchmark settings, applied identically to every platform."""

    seed: int
    warmup_iterations: int
    measured_iterations: int
    concurrency_levels: tuple[int, ...]
    read_ratio: float
    batch_size: int
    query_timeout_s: float
    data_dir: Path
    results_dir: Path

    @property
    def raw_results_dir(self) -> Path:
        return self.results_dir / "raw"

    @property
    def charts_dir(self) -> Path:
        return self.results_dir / "charts"


def load_env() -> None:
    """Load .env from the repository root if present."""
    load_dotenv(REPO_ROOT / ".env")


def load_settings() -> Settings:
    load_env()
    settings = Settings(
        seed=_env_int("BENCH_SEED", 42),
        warmup_iterations=_env_int("BENCH_WARMUP_ITERATIONS", 20),
        measured_iterations=_env_int("BENCH_MEASURED_ITERATIONS", 100),
        concurrency_levels=_env_int_tuple("BENCH_CONCURRENCY_LEVELS", (1, 10, 40)),
        read_ratio=_env_float("BENCH_READ_RATIO", 0.9),
        batch_size=_env_int("BENCH_BATCH_SIZE", 1000),
        query_timeout_s=_env_float("BENCH_QUERY_TIMEOUT_S", 30.0),
        data_dir=_env_path("BENCH_DATA_DIR", "data"),
        results_dir=_env_path("BENCH_RESULTS_DIR", "results"),
    )
    _validate_settings(settings)
    return settings


def load_targets(names: list[str] | None = None) -> list[Target]:
    """Build the configured targets, in the order given by ``PLATFORMS``.

    Without ``names``, every platform that has a URI is returned. With
    ``names``, an unknown or unconfigured platform is an error rather than a
    silent skip.
    """
    load_env()
    wanted = list(PLATFORMS) if names is None else names
    targets: list[Target] = []
    for name in wanted:
        if name not in PLATFORMS:
            known = ", ".join(PLATFORMS)
            raise ConfigError(f"unknown platform {name!r}; known platforms: {known}")
        target = _build_target(name)
        if target is None:
            if names is not None:
                raise ConfigError(f"{name} is not configured: set {name.upper()}_URI")
            continue
        targets.append(target)
    return targets


def unconfigured_platforms() -> list[str]:
    load_env()
    return [name for name in PLATFORMS if not os.environ.get(f"{name.upper()}_URI", "").strip()]


def _build_target(name: str) -> Target | None:
    prefix = name.upper()
    uri = _env_str(f"{prefix}_URI")
    if not uri:
        return None
    return Target(
        name=name,
        kind=PLATFORMS[name],
        uri=uri,
        username=_env_str(f"{prefix}_USERNAME"),
        password=_env_str(f"{prefix}_PASSWORD"),
        database=_env_str(f"{prefix}_DATABASE"),
        specs=_env_str(f"{prefix}_SPECS") or NOT_OBSERVABLE,
    )


def _validate_settings(settings: Settings) -> None:
    if settings.measured_iterations < 1:
        raise ConfigError("BENCH_MEASURED_ITERATIONS must be at least 1")
    if settings.warmup_iterations < 0:
        raise ConfigError("BENCH_WARMUP_ITERATIONS must not be negative")
    if settings.batch_size < 1:
        raise ConfigError("BENCH_BATCH_SIZE must be at least 1")
    if settings.query_timeout_s <= 0:
        raise ConfigError("BENCH_QUERY_TIMEOUT_S must be positive")
    if not 0.0 <= settings.read_ratio <= 1.0:
        raise ConfigError("BENCH_READ_RATIO must be between 0.0 and 1.0")
    if not settings.concurrency_levels:
        raise ConfigError("BENCH_CONCURRENCY_LEVELS must list at least one level")
    if any(level < 1 for level in settings.concurrency_levels):
        raise ConfigError("BENCH_CONCURRENCY_LEVELS must all be at least 1")


def _env_str(name: str) -> str:
    return os.environ.get(name, "").strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a comma-separated integer list, got {raw!r}") from exc


def _env_path(name: str, default: str) -> Path:
    raw = _env_str(name) or default
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path

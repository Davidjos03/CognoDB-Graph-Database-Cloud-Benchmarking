"""Adapter registry: one entry per driver family.

``config.PLATFORMS`` maps a platform name to a kind; this maps a kind to the
class that implements it. Adding a database means adding one adapter here.
"""

from __future__ import annotations

from benchmark.adapters.base import AdapterError, BaseGraphAdapter
from benchmark.adapters.bolt import BoltAdapter
from benchmark.config import Settings, Target

ADAPTERS: dict[str, type[BaseGraphAdapter]] = {
    "bolt": BoltAdapter,
}


def create(target: Target, settings: Settings) -> BaseGraphAdapter:
    adapter_class = ADAPTERS.get(target.kind)
    if adapter_class is None:
        raise AdapterError(f"{target.name}: no adapter implemented for kind {target.kind!r} yet")
    return adapter_class(target, settings)


__all__ = ["ADAPTERS", "AdapterError", "BaseGraphAdapter", "BoltAdapter", "create"]

"""Training utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["train", "evaluate_checkpoint"]


def __getattr__(name: str) -> Any:
    if name == "train":
        return import_module(".trainer", __name__).train
    if name == "evaluate_checkpoint":
        return import_module(".evaluate", __name__).evaluate_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

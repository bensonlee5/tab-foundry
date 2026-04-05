"""Training utilities."""

from __future__ import annotations

from typing import Any


__all__ = ["train", "evaluate_checkpoint"]


def __getattr__(name: str) -> Any:
    if name == "evaluate_checkpoint":
        from .evaluate import evaluate_checkpoint

        return evaluate_checkpoint
    if name == "train":
        from .trainer import train

        return train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

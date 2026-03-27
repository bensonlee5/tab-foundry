"""Architecture family implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "TabFoundrySandwichClassifier",
    "TabFoundrySimpleClassifier",
    "TabFoundryStagedClassifier",
]


if TYPE_CHECKING:
    from .tabfoundry_sandwich import TabFoundrySandwichClassifier
    from .tabfoundry_simple import TabFoundrySimpleClassifier
    from .tabfoundry_staged import TabFoundryStagedClassifier


def __getattr__(name: str) -> Any:
    if name == "TabFoundrySandwichClassifier":
        from .tabfoundry_sandwich import TabFoundrySandwichClassifier

        return TabFoundrySandwichClassifier
    if name == "TabFoundrySimpleClassifier":
        from .tabfoundry_simple import TabFoundrySimpleClassifier

        return TabFoundrySimpleClassifier
    if name == "TabFoundryStagedClassifier":
        from .tabfoundry_staged import TabFoundryStagedClassifier

        return TabFoundryStagedClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

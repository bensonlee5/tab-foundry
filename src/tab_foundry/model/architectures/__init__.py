"""Architecture family implementations."""

from .tabfoundry_simple import TabFoundrySimpleClassifier
from .tabfoundry_staged import TabFoundryStagedClassifier

__all__ = ["TabFoundrySimpleClassifier", "TabFoundryStagedClassifier"]

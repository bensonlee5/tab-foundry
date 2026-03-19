"""Shared configuration objects for OpenML benchmark bundle building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tab_foundry.bench.openml_task_source_registry import DEFAULT_OPENML_TASK_SOURCE, task_ids_for_source


@dataclass(slots=True, frozen=True)
class OpenMLBenchmarkBundleConfig:
    """Configuration for generating a pinned OpenML bundle."""

    bundle_name: str
    version: int
    task_source: str = DEFAULT_OPENML_TASK_SOURCE
    task_type: str = "supervised_classification"
    new_instances: int = 200
    max_features: int = 10
    max_classes: int | None = 2
    max_missing_pct: float = 0.0
    min_minority_class_pct: float = 2.5
    task_ids: tuple[int, ...] | None = None
    discover_from_openml: bool = False
    min_instances: int = 1
    min_task_count: int = 1

    def resolved_task_ids(self) -> tuple[int, ...]:
        """Resolve custom task ids or fall back to the named pinned source pool."""

        if self.discover_from_openml:
            raise RuntimeError("OpenML discovery mode does not use pinned task ids")
        if self.task_ids is not None:
            return tuple(int(task_id) for task_id in self.task_ids)
        return task_ids_for_source(self.task_source)


@dataclass(slots=True, frozen=True)
class OpenMLBenchmarkTaskCandidate:
    """Notebook-source task metadata used for bundle filtering."""

    task_id: int
    number_of_features: float
    number_of_classes: float | None
    missing_pct: float
    minority_class_pct: float | None
    dataset_id: int | None = None
    dataset_name: str | None = None
    estimation_procedure: str | None = None
    number_of_instances: float | None = None


@dataclass(slots=True, frozen=True)
class OpenMLBenchmarkCandidateReportEntry:
    """One task-candidate decision recorded during discovery."""

    task_id: int
    status: str
    reason: str
    dataset_id: int | None = None
    dataset_name: str | None = None
    estimation_procedure: str | None = None


@dataclass(slots=True, frozen=True)
class OpenMLBenchmarkBundleBuildResult:
    """Final bundle payload and optional discovery report entries."""

    bundle: dict[str, Any]
    report_entries: tuple[OpenMLBenchmarkCandidateReportEntry, ...] = ()


def parse_max_classes_arg(raw_value: str) -> int | None:
    normalized = str(raw_value).strip().lower()
    if normalized == "auto":
        return None
    value = int(normalized)
    if value <= 0:
        raise ValueError("max_classes must be a positive int or 'auto'")
    return value

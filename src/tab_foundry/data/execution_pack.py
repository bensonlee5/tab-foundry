"""Derived exact-shape execution-pack materialization."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


_REGIME_PATTERN = re.compile(r"(?:^|[/_-])(mcar|mar|mnar)(?=$|[/_-])", re.IGNORECASE)
_REGIME_COLUMNS = ("missing_mechanism", "missingness_mechanism", "execution_pack_regime")
_REQUIRED_COLUMNS = {
    "split",
    "task",
    "dataset_index",
    "train_path",
    "test_path",
    "catalog_path",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
}


@dataclass(slots=True, frozen=True)
class ExecutionPackSummary:
    """Summary of one derived execution-pack manifest."""

    source_manifest_path: Path
    out_manifest_path: Path
    selected_records: int
    total_microbatches: int
    total_blocks: int
    task_batch_size: int
    grad_accum_steps: int
    max_steps: int
    signature_family_optimizer_step_block_length: int
    signature_family_count: int
    exact_signature_count: int
    regime_counts: dict[str, int]
    class_counts: dict[str, int]
    feature_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_manifest_path": str(self.source_manifest_path),
            "out_manifest_path": str(self.out_manifest_path),
            "selected_records": int(self.selected_records),
            "total_microbatches": int(self.total_microbatches),
            "total_blocks": int(self.total_blocks),
            "task_batch_size": int(self.task_batch_size),
            "grad_accum_steps": int(self.grad_accum_steps),
            "max_steps": int(self.max_steps),
            "signature_family_optimizer_step_block_length": int(
                self.signature_family_optimizer_step_block_length
            ),
            "signature_family_count": int(self.signature_family_count),
            "exact_signature_count": int(self.exact_signature_count),
            "regime_counts": dict(self.regime_counts),
            "class_counts": dict(self.class_counts),
            "feature_counts": dict(self.feature_counts),
        }


def _positive_int(value: int, *, name: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be >= 1, got {resolved}")
    return resolved


def _infer_missingness_regime(row: pd.Series) -> str:
    for column in _REGIME_COLUMNS:
        if column in row and pd.notna(row[column]):
            value = str(row[column]).strip().lower()
            if value:
                return value
    for column in ("train_path", "catalog_path", "test_path", "dataset_id", "source_shard_relpath"):
        if column not in row or pd.isna(row[column]):
            continue
        match = _REGIME_PATTERN.search(str(row[column]))
        if match is not None:
            return match.group(1).lower()
    return "unknown"


def _portable_record_path(raw_path: Any, *, source_manifest_path: Path, out_manifest_path: Path) -> str:
    raw_text = str(raw_path)
    path = Path(raw_text)
    resolved = path if path.is_absolute() else (source_manifest_path.parent / path).resolve()
    try:
        return os.path.relpath(resolved, start=out_manifest_path.parent)
    except ValueError:
        return str(resolved)


def _full_batches(rows: pd.DataFrame, *, task_batch_size: int) -> deque[list[int]]:
    ordered_indices = [int(index) for index in rows.index.tolist()]
    batches: deque[list[int]] = deque()
    for start in range(0, len(ordered_indices) - task_batch_size + 1, task_batch_size):
        batches.append(ordered_indices[start : start + task_batch_size])
    return batches


def _sorted_group_keys(keys: Iterable[tuple[int, int, int, int, str]]) -> list[tuple[int, int, int, int, str]]:
    return sorted(keys, key=lambda item: (item[0], item[1], item[2], item[3], item[4]))


def _write_pack_manifest(
    frame: pd.DataFrame,
    *,
    source_manifest_path: Path,
    out_manifest_path: Path,
) -> None:
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_metadata = pq.ParquetFile(source_manifest_path).schema_arrow.metadata or {}
    metadata = dict(source_metadata)
    summary: dict[str, Any] = {}
    raw_summary = metadata.get(b"tab_foundry_manifest_summary")
    if raw_summary is not None:
        try:
            summary = json.loads(raw_summary.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            summary = {}
    selected_records = int(len(frame))
    summary.update(
        {
            "discovered_records": selected_records,
            "excluded_records": 0,
            "total_records": selected_records,
            "train_records": selected_records,
            "val_records": 0,
            "test_records": 0,
            "filter_status_counts": (
                frame["filter_status"].value_counts().sort_index().astype(int).to_dict()
                if "filter_status" in frame
                else {}
            ),
            "missing_value_status_counts": (
                frame["missing_value_status"].value_counts().sort_index().astype(int).to_dict()
                if "missing_value_status" in frame
                else {}
            ),
        }
    )
    metadata[b"tab_foundry_manifest_summary"] = json.dumps(
        summary,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, out_manifest_path, compression="zstd")


def materialize_exact_shape_execution_pack(
    *,
    source_manifest_path: Path,
    out_manifest_path: Path,
    max_steps: int = 2500,
    grad_accum_steps: int = 4,
    task_batch_size: int = 16,
    signature_family_optimizer_step_block_length: int = 2,
    total_train_tasks: int | None = 160000,
    summary_out: Path | None = None,
) -> ExecutionPackSummary:
    """Build an ordered, train-only, no-reuse exact-shape execution-pack manifest."""

    resolved_source = source_manifest_path.expanduser().resolve()
    resolved_out = out_manifest_path.expanduser().resolve()
    resolved_max_steps = _positive_int(max_steps, name="max_steps")
    resolved_grad_accum_steps = _positive_int(grad_accum_steps, name="grad_accum_steps")
    resolved_task_batch_size = _positive_int(task_batch_size, name="task_batch_size")
    resolved_block_length = _positive_int(
        signature_family_optimizer_step_block_length,
        name="signature_family_optimizer_step_block_length",
    )
    total_microbatches = resolved_max_steps * resolved_grad_accum_steps
    expected_train_tasks = total_microbatches * resolved_task_batch_size
    if total_train_tasks is not None and int(total_train_tasks) != expected_train_tasks:
        raise ValueError(
            "total_train_tasks must equal max_steps * grad_accum_steps * task_batch_size: "
            f"expected={expected_train_tasks}, got={int(total_train_tasks)}"
        )
    block_microbatches = resolved_grad_accum_steps * resolved_block_length
    if total_microbatches % block_microbatches != 0:
        raise ValueError(
            "max_steps * grad_accum_steps must be divisible by "
            "grad_accum_steps * signature_family_optimizer_step_block_length"
        )
    total_blocks = total_microbatches // block_microbatches

    table = pq.read_table(resolved_source)
    frame = table.to_pandas()
    missing_columns = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"source manifest is missing required columns: {missing_columns}")
    frame = frame[
        (frame["split"].astype(str) == "train")
        & (frame["task"].astype(str) == "classification")
    ].copy()
    if frame.empty:
        raise RuntimeError(f"source manifest has no train classification rows: {resolved_source}")
    frame["execution_pack_regime"] = frame.apply(_infer_missingness_regime, axis=1)
    for path_column in ("train_path", "test_path", "catalog_path", "teacher_conditionals_path"):
        if path_column not in frame:
            continue
        frame[path_column] = frame[path_column].map(
            lambda value: (
                None
                if value is None or pd.isna(value)
                else _portable_record_path(
                    value,
                    source_manifest_path=resolved_source,
                    out_manifest_path=resolved_out,
                )
            )
        )

    grouped_batches: dict[tuple[int, int, int, int, str], deque[list[int]]] = {}
    for group_key, group_rows in frame.groupby(
        ["n_train", "n_test", "n_features", "n_classes", "execution_pack_regime"],
        sort=True,
        dropna=False,
    ):
        resolved_key = (
            int(group_key[0]),
            int(group_key[1]),
            int(group_key[2]),
            int(group_key[3]),
            str(group_key[4]),
        )
        batches = _full_batches(group_rows, task_batch_size=resolved_task_batch_size)
        if batches:
            grouped_batches[resolved_key] = batches
    if not grouped_batches:
        raise RuntimeError(
            "source manifest does not contain any full exact-signature task batches "
            f"of size {resolved_task_batch_size}"
        )

    family_to_group_keys: dict[tuple[int, int, int], deque[tuple[int, int, int, int, str]]] = {}
    for group_key in _sorted_group_keys(grouped_batches):
        family = group_key[:3]
        family_to_group_keys.setdefault(family, deque()).append(group_key)
    family_cycle: deque[tuple[int, int, int]] = deque(sorted(family_to_group_keys))
    selected_indices: list[int] = []
    block_index = 0
    while len(selected_indices) < expected_train_tasks:
        if not family_cycle:
            raise RuntimeError(
                "source manifest exhausted before execution pack budget was filled: "
                f"selected={len(selected_indices)}, required={expected_train_tasks}"
            )
        family = family_cycle.popleft()
        group_cycle = family_to_group_keys.get(family)
        if not group_cycle:
            continue
        block_batches = 0
        while block_batches < block_microbatches:
            if not group_cycle:
                break
            group_key = group_cycle.popleft()
            group_batches = grouped_batches.get(group_key)
            if group_batches is None or not group_batches:
                grouped_batches.pop(group_key, None)
                continue
            selected_indices.extend(group_batches.popleft())
            block_batches += 1
            if group_batches:
                group_cycle.append(group_key)
            else:
                grouped_batches.pop(group_key, None)
        if block_batches == block_microbatches and group_cycle:
            family_cycle.append(family)
        elif group_cycle:
            family_cycle.append(family)
        block_index += 1
        if block_index > total_blocks * max(2, len(family_to_group_keys)):
            raise RuntimeError("execution pack scheduler failed to make bounded progress")

    selected_indices = selected_indices[:expected_train_tasks]
    selected = frame.loc[selected_indices].copy()
    selected["split"] = "train"
    selected["execution_pack_index"] = range(len(selected))
    selected["execution_pack_microbatch_index"] = (
        selected["execution_pack_index"] // resolved_task_batch_size
    ).astype(int)
    selected["execution_pack_block_index"] = (
        selected["execution_pack_microbatch_index"] // block_microbatches
    ).astype(int)
    selected["execution_pack_signature_family"] = selected.apply(
        lambda row: f"{int(row['n_train'])}x{int(row['n_test'])}x{int(row['n_features'])}",
        axis=1,
    )
    selected["execution_pack_signature"] = selected.apply(
        lambda row: (
            f"{int(row['n_train'])}x{int(row['n_test'])}x"
            f"{int(row['n_features'])}x{int(row['n_classes'])}"
        ),
        axis=1,
    )
    selected["execution_pack_source_manifest_path"] = str(resolved_source)

    _write_pack_manifest(
        selected,
        source_manifest_path=resolved_source,
        out_manifest_path=resolved_out,
    )
    summary = ExecutionPackSummary(
        source_manifest_path=resolved_source,
        out_manifest_path=resolved_out,
        selected_records=int(len(selected)),
        total_microbatches=total_microbatches,
        total_blocks=total_blocks,
        task_batch_size=resolved_task_batch_size,
        grad_accum_steps=resolved_grad_accum_steps,
        max_steps=resolved_max_steps,
        signature_family_optimizer_step_block_length=resolved_block_length,
        signature_family_count=int(selected["execution_pack_signature_family"].nunique()),
        exact_signature_count=int(selected["execution_pack_signature"].nunique()),
        regime_counts={
            str(key): int(value)
            for key, value in selected["execution_pack_regime"].value_counts().sort_index().items()
        },
        class_counts={
            str(key): int(value)
            for key, value in selected["n_classes"].value_counts().sort_index().items()
        },
        feature_counts={
            str(key): int(value)
            for key, value in selected["n_features"].value_counts().sort_index().items()
        },
    )
    if summary_out is not None:
        resolved_summary_out = summary_out.expanduser().resolve()
        resolved_summary_out.parent.mkdir(parents=True, exist_ok=True)
        resolved_summary_out.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


__all__ = ["ExecutionPackSummary", "materialize_exact_shape_execution_pack"]

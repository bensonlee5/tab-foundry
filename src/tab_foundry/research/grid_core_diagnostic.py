"""Grid-core checkpoint perturbation diagnostics for grid sandwich models."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence, cast

import torch

from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.checkpoint import TabFoundryClassifier, load_checkpoint_classifier_model
from tab_foundry.bench.openml_benchmark import (
    default_benchmark_manifest_path,
    load_benchmark_manifest_datasets,
)
from tab_foundry.bench.openml_benchmark.metrics import evaluate_classifier
from tab_foundry.device import resolve_device
from tab_foundry.repo_paths import repo_root


GRID_CORE_INTERVENTION_MODES = ("ablate_chunk", "repeat_chunk")
GRID_CORE_CHUNK_SCOPES = ("all", "middle")
DEFAULT_REPEAT_COUNTS = (2, 4)
_CLASSIFICATION_TASK_TYPE = "supervised_classification"
_BENCHMARK_SURFACE_TUPLE_SIZE = 3


def contiguous_layer_chunks(layer_count: int, *, scope: str = "all") -> list[tuple[int, int]]:
    """Enumerate inclusive contiguous layer chunks."""

    count = int(layer_count)
    if count <= 0:
        raise ValueError(f"layer_count must be positive, got {layer_count!r}")
    normalized_scope = str(scope).strip().lower()
    if normalized_scope not in GRID_CORE_CHUNK_SCOPES:
        raise ValueError(f"chunk scope must be one of {GRID_CORE_CHUNK_SCOPES}, got {scope!r}")
    chunks = [
        (start, end)
        for start in range(count)
        for end in range(start, count)
    ]
    if normalized_scope == "middle":
        chunks = [(start, end) for start, end in chunks if start > 0 and end < count - 1]
        if not chunks:
            raise ValueError(
                f"middle chunk scope is empty for layer_count={count}; use scope='all'"
            )
    return chunks


def default_grid_core_diagnostic_out_dir(checkpoint_path: Path) -> Path:
    checkpoint = checkpoint_path.expanduser().resolve()
    run_label = checkpoint.parent.parent.name if checkpoint.parent.name == "checkpoints" else checkpoint.stem
    return repo_root() / "outputs" / "research" / "grid_core_diagnostics" / run_label


def _checkpoint_grid_layer_count(checkpoint_path: Path) -> int:
    checkpoint = checkpoint_path.expanduser().resolve()
    _model, spec = load_checkpoint_classifier_model(checkpoint, device=torch.device("cpu"))
    if str(spec.arch).strip().lower() != "grid_sandwich":
        raise RuntimeError(
            "grid-core perturbation diagnostics require a grid_sandwich checkpoint, "
            f"got arch={spec.arch!r}"
        )
    if spec.grid_recurrence_steps is not None:
        raise RuntimeError(
            "grid-core perturbation diagnostics require distinct grid layers; "
            f"got grid_recurrence_steps={spec.grid_recurrence_steps!r}"
        )
    return int(spec.sandwich_layers)


def _load_benchmark_surface(
    benchmark_manifest_path: Path | None,
) -> tuple[Mapping[str, Any], str, bool, Path]:
    manifest_path = (
        default_benchmark_manifest_path()
        if benchmark_manifest_path is None
        else benchmark_manifest_path.expanduser().resolve()
    )
    loaded = load_benchmark_manifest_datasets(benchmark_manifest_path=manifest_path)
    if not isinstance(loaded, tuple) or len(loaded) != _BENCHMARK_SURFACE_TUPLE_SIZE:
        raise RuntimeError("load_benchmark_manifest_datasets returned an unexpected shape")
    datasets, _benchmark_tasks, benchmark_surface = loaded
    if not isinstance(datasets, Mapping) or not isinstance(benchmark_surface, Mapping):
        raise RuntimeError("benchmark manifest loader returned invalid payload types")
    task_type = str(benchmark_surface["task_type"])
    if task_type != _CLASSIFICATION_TASK_TYPE:
        raise RuntimeError(
            "grid-core perturbation diagnostics are classification-only, "
            f"got task_type={task_type!r}"
        )
    return datasets, task_type, bool(benchmark_surface["allow_missing_values"]), manifest_path


def _apply_grid_core_intervention(
    classifier: TabFoundryClassifier,
    *,
    mode: str,
    start_layer: int,
    end_layer: int,
    repeat_count: int,
) -> None:
    setter = getattr(classifier.model, "set_grid_core_intervention", None)
    if not callable(setter):
        raise RuntimeError(
            "loaded checkpoint model does not expose set_grid_core_intervention(); "
            "expected grid_sandwich"
        )
    setter(
        mode=mode,
        start_layer=int(start_layer),
        end_layer=int(end_layer),
        repeat_count=int(repeat_count),
    )


def _metric_payload(metrics: Mapping[str, float]) -> dict[str, float | None]:
    return {
        "log_loss": None if metrics.get("Log Loss") is None else float(metrics["Log Loss"]),
        "brier_score": None
        if metrics.get("Brier Score") is None
        else float(metrics["Brier Score"]),
        "roc_auc": None if metrics.get("ROC AUC") is None else float(metrics["ROC AUC"]),
    }


def _evaluate_checkpoint_metrics(
    *,
    checkpoint_path: Path,
    datasets: Mapping[str, Any],
    device: str,
    allow_missing_values: bool,
    intervention: Mapping[str, int | str] | None,
) -> dict[str, Any]:
    resolved_device = resolve_device(device)
    started = time.perf_counter()
    classifier = TabFoundryClassifier(checkpoint_path, device=resolved_device)
    if intervention is not None:
        _apply_grid_core_intervention(
            classifier,
            mode=str(intervention["mode"]),
            start_layer=int(intervention["start_layer"]),
            end_layer=int(intervention["end_layer"]),
            repeat_count=int(intervention.get("repeat_count", 2)),
        )
    parameter_count = sum(int(parameter.numel()) for parameter in classifier.model.parameters())
    metrics = evaluate_classifier(
        classifier,
        datasets,
        allow_missing_values=allow_missing_values,
    )
    return {
        "metrics": _metric_payload(metrics),
        "parameter_count": int(parameter_count),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _metric_delta(
    candidate: Mapping[str, float | None],
    baseline: Mapping[str, float | None],
    key: str,
) -> float | None:
    candidate_value = candidate.get(key)
    baseline_value = baseline.get(key)
    if candidate_value is None or baseline_value is None:
        return None
    return float(candidate_value) - float(baseline_value)


def _candidate_record(
    *,
    mode: str,
    start_layer: int,
    end_layer: int,
    repeat_count: int,
    evaluation: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = cast(Mapping[str, float | None], evaluation["metrics"])
    baseline_metrics = cast(Mapping[str, float | None], baseline["metrics"])
    candidate_id = (
        f"{mode}_r{repeat_count}_{start_layer}_{end_layer}"
        if mode == "repeat_chunk"
        else f"{mode}_{start_layer}_{end_layer}"
    )
    return {
        "id": candidate_id,
        "mode": mode,
        "chunk": {
            "start_layer": int(start_layer),
            "end_layer": int(end_layer),
            "layers": list(range(int(start_layer), int(end_layer) + 1)),
        },
        "repeat_count": int(repeat_count) if mode == "repeat_chunk" else None,
        "metrics": dict(metrics),
        "deltas": {
            "log_loss": _metric_delta(metrics, baseline_metrics, "log_loss"),
            "brier_score": _metric_delta(metrics, baseline_metrics, "brier_score"),
            "roc_auc": _metric_delta(metrics, baseline_metrics, "roc_auc"),
            "elapsed_seconds": float(evaluation["elapsed_seconds"])
            - float(baseline["elapsed_seconds"]),
        },
        "parameter_count": int(evaluation["parameter_count"]),
        "elapsed_seconds": float(evaluation["elapsed_seconds"]),
    }


def _finite_sort_key(value: Any, *, descending: bool = False) -> float:
    if value is None:
        return math.inf
    value_float = float(value)
    if not math.isfinite(value_float):
        return math.inf
    return -value_float if descending else value_float


def _rankings(candidates: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    repeat_records = [record for record in candidates if record.get("mode") == "repeat_chunk"]
    ablate_records = [record for record in candidates if record.get("mode") == "ablate_chunk"]
    return {
        "repeat_by_log_loss_delta": [
            str(record["id"])
            for record in sorted(
                repeat_records,
                key=lambda record: _finite_sort_key(
                    cast(Mapping[str, Any], record["deltas"]).get("log_loss")
                ),
            )
        ],
        "ablation_by_log_loss_harm": [
            str(record["id"])
            for record in sorted(
                ablate_records,
                key=lambda record: _finite_sort_key(
                    cast(Mapping[str, Any], record["deltas"]).get("log_loss"),
                    descending=True,
                ),
            )
        ],
    }


def _chunk_decisions(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_chunk: dict[tuple[int, int], dict[str, Any]] = {}
    for record in candidates:
        chunk = cast(Mapping[str, Any], record["chunk"])
        key = (int(chunk["start_layer"]), int(chunk["end_layer"]))
        bucket = by_chunk.setdefault(key, {"ablate_chunk": None, "repeat_chunk": []})
        if str(record["mode"]) == "ablate_chunk":
            bucket["ablate_chunk"] = record
        if str(record["mode"]) == "repeat_chunk":
            cast(list[Mapping[str, Any]], bucket["repeat_chunk"]).append(record)

    decisions: list[dict[str, Any]] = []
    for (start, end), records in sorted(by_chunk.items()):
        ablation = cast(Mapping[str, Any] | None, records.get("ablate_chunk"))
        repeat_records = cast(list[Mapping[str, Any]], records.get("repeat_chunk", []))
        best_repeat = (
            None
            if not repeat_records
            else min(
                repeat_records,
                key=lambda record: _finite_sort_key(
                    cast(Mapping[str, Any], record["deltas"]).get("log_loss")
                ),
            )
        )
        ablation_log_loss_delta = (
            None
            if ablation is None
            else cast(Mapping[str, Any], ablation["deltas"]).get("log_loss")
        )
        best_repeat_log_loss_delta = (
            None
            if best_repeat is None
            else cast(Mapping[str, Any], best_repeat["deltas"]).get("log_loss")
        )
        repeat_deltas = [
            {
                "repeat_count": record.get("repeat_count"),
                "log_loss_delta": cast(Mapping[str, Any], record["deltas"]).get("log_loss"),
                "brier_score_delta": cast(Mapping[str, Any], record["deltas"]).get(
                    "brier_score"
                ),
                "roc_auc_delta": cast(Mapping[str, Any], record["deltas"]).get("roc_auc"),
            }
            for record in sorted(
                repeat_records,
                key=lambda record: int(cast(int, record.get("repeat_count") or 0)),
            )
        ]
        if ablation_log_loss_delta is None or best_repeat_log_loss_delta is None:
            label = "insufficient_metrics"
        elif float(ablation_log_loss_delta) <= 0.0:
            label = "efficiency_or_pruning_candidate"
        elif float(best_repeat_log_loss_delta) <= 0.0:
            label = "recurrence_promising"
        else:
            label = "necessary_not_recurrence_friendly"
        decisions.append(
            {
                "chunk": {
                    "start_layer": start,
                    "end_layer": end,
                    "layers": list(range(start, end + 1)),
                },
                "decision_label": label,
                "ablation_log_loss_delta": ablation_log_loss_delta,
                "best_repeat_count": None if best_repeat is None else best_repeat.get("repeat_count"),
                "best_repeat_log_loss_delta": best_repeat_log_loss_delta,
                "repeat_deltas": repeat_deltas,
            }
        )
    return decisions


def render_grid_core_perturbation_markdown(payload: Mapping[str, Any]) -> str:
    baseline = cast(Mapping[str, Any], payload["baseline"])
    candidates = cast(Sequence[Mapping[str, Any]], payload["candidates"])
    lines = [
        "# Grid-Core Perturbation Diagnostic",
        "",
        f"- Checkpoint: `{payload['checkpoint_path']}`",
        f"- Benchmark manifest: `{payload['benchmark_manifest_path']}`",
        f"- Layer count: `{payload['layer_count']}`",
        f"- Baseline parameter count: `{baseline['parameter_count']}`",
        "",
        "## Baseline",
        "",
        "| log loss | Brier | ROC AUC | elapsed seconds |",
        "| ---: | ---: | ---: | ---: |",
        (
            f"| {_format_metric(cast(Mapping[str, Any], baseline['metrics']).get('log_loss'))} "
            f"| {_format_metric(cast(Mapping[str, Any], baseline['metrics']).get('brier_score'))} "
            f"| {_format_metric(cast(Mapping[str, Any], baseline['metrics']).get('roc_auc'))} "
            f"| {_format_metric(baseline.get('elapsed_seconds'))} |"
        ),
        "",
        "## Perturbation Ranking",
        "",
        "| rank | mode | repeats | chunk | log loss | d log loss | Brier | d Brier | ROC AUC | d ROC AUC | elapsed | d elapsed |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked_candidates = sorted(
        candidates,
        key=lambda record: (
            str(record["mode"]),
            _finite_sort_key(cast(Mapping[str, Any], record["deltas"]).get("log_loss")),
        ),
    )
    for rank, record in enumerate(ranked_candidates, start=1):
        metrics = cast(Mapping[str, Any], record["metrics"])
        deltas = cast(Mapping[str, Any], record["deltas"])
        chunk = cast(Mapping[str, Any], record["chunk"])
        chunk_label = f"{chunk['start_layer']}..{chunk['end_layer']}"
        repeat_label = "n/a" if record.get("repeat_count") is None else str(record["repeat_count"])
        lines.append(
            f"| {rank} | `{record['mode']}` | {repeat_label} | `{chunk_label}` "
            f"| {_format_metric(metrics.get('log_loss'))} "
            f"| {_format_metric(deltas.get('log_loss'), signed=True)} "
            f"| {_format_metric(metrics.get('brier_score'))} "
            f"| {_format_metric(deltas.get('brier_score'), signed=True)} "
            f"| {_format_metric(metrics.get('roc_auc'))} "
            f"| {_format_metric(deltas.get('roc_auc'), signed=True)} "
            f"| {_format_metric(record.get('elapsed_seconds'))} "
            f"| {_format_metric(deltas.get('elapsed_seconds'), signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## Decision Labels",
            "",
            "| chunk | label | ablation d log loss | best repeats | best repeat d log loss |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for decision in cast(Sequence[Mapping[str, Any]], payload["chunk_decisions"]):
        chunk = cast(Mapping[str, Any], decision["chunk"])
        lines.append(
            f"| `{chunk['start_layer']}..{chunk['end_layer']}` "
            f"| `{decision['decision_label']}` "
            f"| {_format_metric(decision.get('ablation_log_loss_delta'), signed=True)} "
            f"| {decision.get('best_repeat_count') or 'n/a'} "
            f"| {_format_metric(decision.get('best_repeat_log_loss_delta'), signed=True)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_metric(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    value_float = float(value)
    if not math.isfinite(value_float):
        return "n/a"
    prefix = "+" if signed and value_float >= 0.0 else ""
    return f"{prefix}{value_float:.6f}"


def render_grid_core_perturbation_text(payload: Mapping[str, Any]) -> str:
    artifacts = cast(Mapping[str, Any], payload["artifacts"])
    rankings = cast(Mapping[str, Any], payload["rankings"])
    top_repeat = cast(Sequence[str], rankings["repeat_by_log_loss_delta"])[:3]
    return "\n".join(
        [
            "grid-core perturbation diagnostic complete:",
            f"  checkpoint={payload['checkpoint_path']}",
            f"  layer_count={payload['layer_count']}",
            f"  candidate_count={len(cast(Sequence[Any], payload['candidates']))}",
            f"  top_repeat_by_log_loss_delta={list(top_repeat)}",
            f"  json={artifacts['json']}",
            f"  markdown={artifacts['markdown']}",
        ]
    )


def run_grid_core_perturbation_diagnostic(
    *,
    checkpoint_path: Path,
    benchmark_manifest_path: Path | None = None,
    out_dir: Path | None = None,
    device: str = "auto",
    repeat_count: int | None = None,
    repeat_counts: Sequence[int] | None = None,
    chunk_scope: str = "all",
    modes: Sequence[str] = GRID_CORE_INTERVENTION_MODES,
    layer_count: int | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint_path.expanduser().resolve()
    if not checkpoint.exists():
        raise RuntimeError(f"checkpoint does not exist: {checkpoint}")
    normalized_repeat_counts: tuple[int, ...]
    if repeat_counts is None:
        normalized_repeat_counts = (
            DEFAULT_REPEAT_COUNTS if repeat_count is None else (int(repeat_count),)
        )
    else:
        normalized_repeat_counts = tuple(int(value) for value in repeat_counts)
    if not normalized_repeat_counts:
        raise ValueError("repeat_counts must include at least one value")
    if any(value <= 0 for value in normalized_repeat_counts):
        raise ValueError("repeat_counts must be positive")
    normalized_repeat_counts = tuple(sorted(set(normalized_repeat_counts)))
    normalized_modes = tuple(str(mode).strip().lower() for mode in modes)
    unsupported_modes = [mode for mode in normalized_modes if mode not in GRID_CORE_INTERVENTION_MODES]
    if unsupported_modes:
        raise ValueError(f"unsupported grid-core perturbation modes: {unsupported_modes}")
    if not normalized_modes:
        normalized_modes = GRID_CORE_INTERVENTION_MODES
    resolved_layer_count = (
        _checkpoint_grid_layer_count(checkpoint) if layer_count is None else int(layer_count)
    )
    chunks = contiguous_layer_chunks(resolved_layer_count, scope=chunk_scope)
    datasets, task_type, allow_missing_values, manifest_path = _load_benchmark_surface(
        benchmark_manifest_path
    )
    baseline = _evaluate_checkpoint_metrics(
        checkpoint_path=checkpoint,
        datasets=datasets,
        device=device,
        allow_missing_values=allow_missing_values,
        intervention=None,
    )
    candidates: list[dict[str, Any]] = []
    for start_layer, end_layer in chunks:
        for mode in normalized_modes:
            current_repeat_counts = (
                normalized_repeat_counts if mode == "repeat_chunk" else (0,)
            )
            for current_repeat_count in current_repeat_counts:
                evaluation = _evaluate_checkpoint_metrics(
                    checkpoint_path=checkpoint,
                    datasets=datasets,
                    device=device,
                    allow_missing_values=allow_missing_values,
                    intervention={
                        "mode": mode,
                        "start_layer": start_layer,
                        "end_layer": end_layer,
                        "repeat_count": current_repeat_count,
                    },
                )
                if int(evaluation["parameter_count"]) != int(baseline["parameter_count"]):
                    raise RuntimeError(
                        "grid-core perturbation changed parameter count: "
                        f"baseline={baseline['parameter_count']}, "
                        f"candidate={evaluation['parameter_count']}, "
                        f"mode={mode}, chunk={start_layer}..{end_layer}, "
                        f"repeat_count={current_repeat_count}"
                    )
                candidates.append(
                    _candidate_record(
                        mode=mode,
                        start_layer=start_layer,
                        end_layer=end_layer,
                        repeat_count=current_repeat_count,
                        evaluation=evaluation,
                        baseline=baseline,
                    )
                )

    resolved_out_dir = (
        default_grid_core_diagnostic_out_dir(checkpoint)
        if out_dir is None
        else out_dir.expanduser().resolve()
    )
    json_path = resolved_out_dir / "grid_core_perturbation_results.json"
    markdown_path = resolved_out_dir / "grid_core_perturbation_results.md"
    payload: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "benchmark_manifest_path": str(manifest_path),
        "device": str(device),
        "task_type": task_type,
        "allow_missing_values": allow_missing_values,
        "layer_count": int(resolved_layer_count),
        "chunk_scope": str(chunk_scope).strip().lower(),
        "repeat_counts": list(normalized_repeat_counts),
        "modes": list(normalized_modes),
        "chunks": [
            {"start_layer": start, "end_layer": end, "layers": list(range(start, end + 1))}
            for start, end in chunks
        ],
        "baseline": baseline,
        "candidates": candidates,
        "rankings": _rankings(candidates),
        "chunk_decisions": _chunk_decisions(candidates),
        "artifacts": {
            "json": str(json_path),
            "markdown": str(markdown_path),
        },
    }
    write_json(json_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_grid_core_perturbation_markdown(payload), encoding="utf-8")
    return payload

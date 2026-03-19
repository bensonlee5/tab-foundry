"""Developer-facing inspection CLI commands."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from omegaconf import OmegaConf
import torch

from tab_foundry.config import compose_config, config_dir
from tab_foundry.data.surface import resolve_data_surface
from tab_foundry.export.contracts import SCHEMA_VERSION_V3
from tab_foundry.export.inspection import export_check
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.inspection import (
    model_surface_payload,
    parameter_counts_from_model_spec,
    synthetic_forward_batch,
)
from tab_foundry.model.outputs import ClassificationOutput
from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.preprocessing import resolve_preprocessing_surface
from tab_foundry.training.batching import move_batch
from tab_foundry.training.health import health_check, run_inspect


_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
_MISSING_MARKER = "<missing>"
_DIFF_EXCLUDED_PATHS = {"runtime.output_dir"}


def _mapping_from_node(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    payload = OmegaConf.to_container(value, resolve=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must resolve to a mapping")
    return {str(key): item for key, item in payload.items()}


def _project_default_experiment() -> str | None:
    root_payload = OmegaConf.to_container(OmegaConf.load(config_dir() / "config.yaml"), resolve=True)
    if not isinstance(root_payload, dict):
        return None
    raw_defaults = root_payload.get("defaults")
    if not isinstance(raw_defaults, list):
        return None
    for entry in raw_defaults:
        if not isinstance(entry, Mapping):
            continue
        raw_experiment = entry.get("experiment")
        if isinstance(raw_experiment, str) and raw_experiment.strip():
            return str(raw_experiment).strip()
    return None


def _resolved_experiment_name(overrides: Sequence[str]) -> str | None:
    for override in reversed(list(overrides)):
        token = str(override).strip()
        if not token.startswith("experiment="):
            continue
        value = token.split("=", 1)[1].strip()
        return value or None
    return _project_default_experiment()


def _training_surface_payload(
    training_cfg: Mapping[str, Any],
    *,
    optimizer_cfg: Mapping[str, Any],
    schedule_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    raw_stages = schedule_cfg.get("stages")
    rendered_stages: list[dict[str, Any]] = []
    if isinstance(raw_stages, list):
        for item in raw_stages:
            if not isinstance(item, Mapping):
                continue
            rendered_stages.append(
                {
                    "name": item.get("name"),
                    "steps": None if item.get("steps") is None else int(item["steps"]),
                    "lr_max": None if item.get("lr_max") is None else float(item["lr_max"]),
                    "warmup_ratio": (
                        None if item.get("warmup_ratio") is None else float(item["warmup_ratio"])
                    ),
                    "lr_schedule": None if item.get("lr_schedule") is None else str(item["lr_schedule"]),
                }
            )
    return {
        "surface_label": str(training_cfg.get("surface_label", "training_default")),
        "apply_schedule": bool(training_cfg.get("apply_schedule", False)),
        "prior_dump_non_finite_policy": training_cfg.get("prior_dump_non_finite_policy"),
        "prior_dump_batch_size": (
            None if training_cfg.get("prior_dump_batch_size") is None else int(training_cfg["prior_dump_batch_size"])
        ),
        "prior_dump_lr_scale_rule": training_cfg.get("prior_dump_lr_scale_rule"),
        "prior_dump_batch_reference_size": (
            None
            if training_cfg.get("prior_dump_batch_reference_size") is None
            else int(training_cfg["prior_dump_batch_reference_size"])
        ),
        "overrides": dict(cast(dict[str, Any], training_cfg.get("overrides", {}))),
        "optimizer_name": None if optimizer_cfg.get("name") is None else str(optimizer_cfg["name"]),
        "schedule_stages": rendered_stages,
    }


def resolve_config_payload(overrides: Sequence[str]) -> dict[str, Any]:
    cfg = compose_config(list(overrides))
    task = str(getattr(cfg, "task", "classification")).strip().lower()
    model_cfg = _mapping_from_node(getattr(cfg, "model", None), context="cfg.model")
    spec = model_build_spec_from_mappings(task=task, primary=model_cfg)
    data_surface = resolve_data_surface(_mapping_from_node(getattr(cfg, "data", None), context="cfg.data"))
    preprocessing_surface = resolve_preprocessing_surface(
        _mapping_from_node(getattr(cfg, "preprocessing", None), context="cfg.preprocessing")
    )
    training_payload = _training_surface_payload(
        _mapping_from_node(getattr(cfg, "training", None), context="cfg.training"),
        optimizer_cfg=_mapping_from_node(getattr(cfg, "optimizer", None), context="cfg.optimizer"),
        schedule_cfg=_mapping_from_node(getattr(cfg, "schedule", None), context="cfg.schedule"),
    )
    runtime_cfg = _mapping_from_node(getattr(cfg, "runtime", None), context="cfg.runtime")
    return {
        "experiment": _resolved_experiment_name(overrides),
        "task": task,
        "model": {
            **model_surface_payload(spec),
            "parameter_counts": parameter_counts_from_model_spec(spec),
        },
        "data": data_surface.to_dict(),
        "preprocessing": preprocessing_surface.to_dict(),
        "training": training_payload,
        "runtime": {
            "device": runtime_cfg.get("device"),
            "output_dir": runtime_cfg.get("output_dir"),
            "seed": runtime_cfg.get("seed"),
        },
    }


def _format_jsonable(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _diff_path(prefix: str, key: str) -> str:
    return key if not prefix else f"{prefix}.{key}"


def _diff_config_values(
    left: Any,
    right: Any,
    *,
    path: str,
    differences: list[dict[str, Any]],
) -> None:
    if path in _DIFF_EXCLUDED_PATHS:
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            _diff_config_values(
                left.get(key, _MISSING_MARKER),
                right.get(key, _MISSING_MARKER),
                path=_diff_path(path, str(key)),
                differences=differences,
            )
        return
    if isinstance(left, list) and isinstance(right, list):
        if left != right:
            differences.append({"path": path, "left": left, "right": right})
        return
    if left != right:
        differences.append({"path": path, "left": left, "right": right})


def diff_config_payloads(
    left_overrides: Sequence[str],
    right_overrides: Sequence[str],
) -> dict[str, Any]:
    left = resolve_config_payload(left_overrides)
    right = resolve_config_payload(right_overrides)
    differences: list[dict[str, Any]] = []
    _diff_config_values(left, right, path="", differences=differences)
    normalized_differences = [
        {
            "path": difference["path"],
            "left": None if difference["left"] == _MISSING_MARKER else difference["left"],
            "right": None if difference["right"] == _MISSING_MARKER else difference["right"],
        }
        for difference in differences
        if difference["path"]
    ]
    return {
        "left": left,
        "right": right,
        "differences": normalized_differences,
    }


def render_resolve_config_text(payload: Mapping[str, Any]) -> str:
    model_payload = cast(Mapping[str, Any], payload["model"])
    parameter_counts = cast(Mapping[str, int], model_payload["parameter_counts"])
    data_payload = cast(Mapping[str, Any], payload["data"])
    preprocessing_payload = cast(Mapping[str, Any], payload["preprocessing"])
    training_payload = cast(Mapping[str, Any], payload["training"])
    lines = [
        "Resolved config.",
        f"experiment={payload.get('experiment') or 'unknown'}",
        f"task={payload['task']}",
        f"model.arch={model_payload['arch']}",
        f"model.stage={model_payload['stage']}",
        f"model.stage_label={model_payload['stage_label']}",
        f"model.benchmark_profile={model_payload.get('benchmark_profile')}",
        f"model.parameters.total={parameter_counts['total_params']}",
        f"model.parameters.trainable={parameter_counts['trainable_params']}",
        f"data.surface_label={data_payload['surface_label']}",
        f"data.source={data_payload['source']}",
        f"data.manifest_path={data_payload['manifest_path']}",
        f"preprocessing.surface_label={preprocessing_payload['surface_label']}",
        f"training.surface_label={training_payload['surface_label']}",
    ]
    module_selection = model_payload.get("module_selection")
    if isinstance(module_selection, Mapping):
        lines.append(f"model.module_selection={_format_jsonable(module_selection)}")
    module_hyperparameters = model_payload.get("module_hyperparameters")
    if isinstance(module_hyperparameters, Mapping):
        lines.append(f"model.module_hyperparameters={_format_jsonable(module_hyperparameters)}")
    schedule_stages = training_payload.get("schedule_stages")
    if isinstance(schedule_stages, list) and schedule_stages:
        lines.append(f"training.schedule_stages={_format_jsonable(schedule_stages)}")
    return "\n".join(lines)


def render_diff_config_text(payload: Mapping[str, Any]) -> str:
    lines = [
        "Resolved config diff.",
        f"left.experiment={cast(Mapping[str, Any], payload['left']).get('experiment') or 'unknown'}",
        f"right.experiment={cast(Mapping[str, Any], payload['right']).get('experiment') or 'unknown'}",
    ]
    differences = cast(list[Mapping[str, Any]], payload["differences"])
    if not differences:
        lines.append("differences=none")
        return "\n".join(lines)
    for difference in differences:
        lines.append(
            f"{difference['path']}: {_format_jsonable(difference['left'])} -> {_format_jsonable(difference['right'])}"
        )
    return "\n".join(lines)


def _resolve_device(requested: str) -> torch.device:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested --device cuda, but CUDA is not available")
        return torch.device("cuda")
    if normalized == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("requested --device mps, but MPS is not available")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"unsupported device: {requested!r}")


def _require_finite_tensor(tensor: torch.Tensor, *, context: str) -> torch.Tensor:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{context} contains non-finite values")
    return tensor


def _validate_forward_output(
    output: ClassificationOutput,
    *,
    expected_output_kind: str,
    expected_num_classes: int,
    expected_test_rows: int,
) -> torch.Tensor:
    if expected_output_kind == "logits":
        logits = output.logits
        if logits is None:
            raise RuntimeError("forward() returned no logits")
        validated = _require_finite_tensor(logits, context="forward logits")
    else:
        class_probs = output.class_probs
        if class_probs is None:
            raise RuntimeError("forward() returned no class_probs")
        validated = _require_finite_tensor(class_probs, context="forward class_probs")
    if validated.ndim != 2:
        raise RuntimeError(f"forward output must have shape [R_test, C], got {tuple(validated.shape)}")
    if int(validated.shape[0]) != expected_test_rows:
        raise RuntimeError(
            "forward output row count mismatch: "
            f"expected {expected_test_rows}, got {validated.shape[0]}"
        )
    if int(validated.shape[1]) != expected_num_classes:
        raise RuntimeError(
            "forward output class count mismatch: "
            f"expected {expected_num_classes}, got {validated.shape[1]}"
        )
    if int(output.num_classes) != expected_num_classes:
        raise RuntimeError(
            "forward output num_classes mismatch: "
            f"expected {expected_num_classes}, got {output.num_classes}"
        )
    return validated


def forward_check(
    overrides: Sequence[str],
    *,
    requested_device: str,
    seed: int,
) -> dict[str, Any]:
    cfg = compose_config(list(overrides))
    task = str(getattr(cfg, "task", "classification")).strip().lower()
    model_cfg = _mapping_from_node(getattr(cfg, "model", None), context="cfg.model")
    spec = model_build_spec_from_mappings(task=task, primary=model_cfg)
    device = _resolve_device(requested_device)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    model = build_model_from_spec(spec).to(device)
    model.eval()
    synthetic_batch = synthetic_forward_batch(spec)
    moved_batch = move_batch(synthetic_batch.task_batch, device)
    x_all = synthetic_batch.x_all.to(device)
    y_train_batched = synthetic_batch.y_train_batched.to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        output = model(moved_batch)
    elapsed_seconds = float(time.perf_counter() - start)
    if not isinstance(output, ClassificationOutput):
        raise RuntimeError(f"forward() must return ClassificationOutput, got {type(output).__name__}")

    validated = _validate_forward_output(
        output,
        expected_output_kind=synthetic_batch.expected_output_kind,
        expected_num_classes=synthetic_batch.expected_num_classes,
        expected_test_rows=synthetic_batch.expected_test_rows,
    )

    batched_summary: dict[str, Any] | None = None
    if synthetic_batch.expected_output_kind == "logits":
        forward_batched = getattr(model, "forward_batched", None)
        if callable(forward_batched):
            with torch.inference_mode():
                batched_logits = cast(
                    torch.Tensor,
                    forward_batched(
                        x_all=x_all,
                        y_train=y_train_batched,
                        train_test_split_index=synthetic_batch.train_test_split_index,
                    ),
                )
            validated_batched = _require_finite_tensor(batched_logits, context="forward_batched logits")
            expected_batched_shape = (
                1,
                synthetic_batch.expected_test_rows,
                synthetic_batch.expected_num_classes,
            )
            if tuple(int(value) for value in validated_batched.shape) != expected_batched_shape:
                raise RuntimeError(
                    "forward_batched output shape mismatch: "
                    f"expected {expected_batched_shape}, got {tuple(validated_batched.shape)}"
                )
            squeezed = validated_batched.squeeze(0)
            if not torch.allclose(squeezed, validated, atol=1.0e-6, rtol=1.0e-5):
                raise RuntimeError("forward_batched logits disagree with forward() logits")
            batched_summary = {
                "shape": [int(dimension) for dimension in validated_batched.shape],
                "dtype": str(validated_batched.dtype),
            }

    surface_label = spec.stage_label or spec.stage or spec.arch
    return {
        "experiment": _resolved_experiment_name(overrides),
        "task": task,
        "surface_label": surface_label,
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "output_kind": synthetic_batch.expected_output_kind,
        "output_shape": [int(dimension) for dimension in validated.shape],
        "output_dtype": str(validated.dtype),
        "expected_num_classes": synthetic_batch.expected_num_classes,
        "batched_output": batched_summary,
        "model": model_surface_payload(spec),
    }


def render_forward_check_text(payload: Mapping[str, Any]) -> str:
    lines = [
        "Forward check passed.",
        f"experiment={payload.get('experiment') or 'unknown'}",
        f"task={payload['task']}",
        f"surface_label={payload['surface_label']}",
        f"device={payload['device']}",
        f"output_kind={payload['output_kind']}",
        f"output_shape={payload['output_shape']}",
        f"output_dtype={payload['output_dtype']}",
        f"expected_num_classes={payload['expected_num_classes']}",
        f"elapsed_seconds={payload['elapsed_seconds']:.3f}",
    ]
    batched_output = payload.get("batched_output")
    if isinstance(batched_output, Mapping):
        lines.append(f"batched_output_shape={batched_output['shape']}")
        lines.append(f"batched_output_dtype={batched_output['dtype']}")
    return "\n".join(lines)


def render_export_check_text(payload: Mapping[str, Any]) -> str:
    model_payload = cast(Mapping[str, Any], payload["model"])
    reference_smoke = cast(Mapping[str, Any], payload["reference_smoke"])
    preprocessor = cast(Mapping[str, Any] | None, payload.get("preprocessor"))
    lines = [
        "Export check passed.",
        f"checkpoint={payload['checkpoint']}",
        f"bundle_dir={payload['bundle_dir']}",
        f"bundle_dir_kept={payload['bundle_dir_kept']}",
        f"schema_version={payload['schema_version']}",
        f"task={payload['task']}",
        f"model.arch={model_payload['arch']}",
        f"model.stage={model_payload['stage']}",
        f"model.stage_label={model_payload['stage_label']}",
        f"reference_output_shape={reference_smoke['output_shape']}",
        f"reference_output_dtype={reference_smoke['output_dtype']}",
        f"reference_num_classes={reference_smoke['num_classes']}",
        f"reference_used_missing_inputs={reference_smoke['used_missing_inputs']}",
        f"elapsed_seconds={payload['elapsed_seconds']:.3f}",
    ]
    if isinstance(preprocessor, Mapping):
        missing_value_policy = preprocessor.get("missing_value_policy")
        classification_label_policy = preprocessor.get("classification_label_policy")
        lines.append(f"preprocessor.feature_order_policy={preprocessor.get('feature_order_policy')}")
        if isinstance(missing_value_policy, Mapping):
            lines.append(f"preprocessor.missing_value_policy={_format_jsonable(dict(missing_value_policy))}")
        if isinstance(classification_label_policy, Mapping):
            lines.append(
                "preprocessor.classification_label_policy="
                f"{_format_jsonable(dict(classification_label_policy))}"
            )
    return "\n".join(lines)


def render_health_check_text(payload: Mapping[str, Any]) -> str:
    metrics = cast(Mapping[str, Any], payload["metrics"])
    lines = [
        f"{payload['verdict']}: {payload['summary']}",
        f"source={payload['source']}",
        f"run_dir={payload['run_dir']}",
        f"clipped_step_fraction={metrics['clipped_step_fraction']}",
        f"upper_block_post_warmup_mean_slope={metrics['upper_block_post_warmup_mean_slope']}",
        f"upper_block_final_to_early_ratio={metrics['upper_block_final_to_early_ratio']}",
        f"initial_train_loss={metrics['initial_train_loss']}",
        f"final_train_loss={metrics['final_train_loss']}",
        f"telemetry_error={payload['telemetry_error']}",
    ]
    return "\n".join(lines)


def render_run_inspect_text(payload: Mapping[str, Any]) -> str:
    lines = [
        "Run inspection.",
        f"run_dir={payload['run_dir']}",
    ]
    surface_labels = payload.get("surface_labels")
    if isinstance(surface_labels, Mapping):
        lines.append(f"surface_labels={_format_jsonable(dict(surface_labels))}")

    health_payload = payload.get("health")
    if isinstance(health_payload, Mapping):
        lines.append(
            f"health={health_payload['verdict']}: {health_payload['summary']}"
        )
    elif payload.get("health_error") is not None:
        lines.append(f"health=unavailable: {payload['health_error']}")

    training_surface_record = payload.get("training_surface_record")
    if isinstance(training_surface_record, Mapping):
        model_payload = training_surface_record.get("model")
        if isinstance(model_payload, Mapping):
            lines.append(f"model.stage_label={model_payload.get('stage_label')}")
            lines.append(f"model.arch={model_payload.get('arch')}")
        data_payload = training_surface_record.get("data")
        if isinstance(data_payload, Mapping):
            lines.append(f"data.surface_label={data_payload.get('surface_label')}")
        preprocessing_payload = training_surface_record.get("preprocessing")
        if isinstance(preprocessing_payload, Mapping):
            lines.append(f"preprocessing.surface_label={preprocessing_payload.get('surface_label')}")
        training_payload = training_surface_record.get("training")
        if isinstance(training_payload, Mapping):
            lines.append(f"training.surface_label={training_payload.get('surface_label')}")

    comparison_summary = payload.get("comparison_summary")
    if isinstance(comparison_summary, Mapping):
        lines.append(f"benchmark_profile={comparison_summary.get('benchmark_profile')}")
        if comparison_summary.get("best_roc_auc") is not None:
            lines.append(f"best_roc_auc={comparison_summary['best_roc_auc']}")
        if comparison_summary.get("final_roc_auc") is not None:
            lines.append(f"final_roc_auc={comparison_summary['final_roc_auc']}")

    benchmark_run_record = payload.get("benchmark_run_record")
    if isinstance(benchmark_run_record, Mapping) and benchmark_run_record.get("run_id") is not None:
        lines.append(f"registry.run_id={benchmark_run_record['run_id']}")
        lines.append(f"registry.track={benchmark_run_record.get('track')}")

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, Mapping):
        present = sorted(
            name
            for name, entry in artifacts.items()
            if isinstance(entry, Mapping) and bool(entry.get("exists"))
        )
        lines.append(f"artifacts.present={_format_jsonable(present)}")
    return "\n".join(lines)


def _run_resolve_config(args: argparse.Namespace) -> int:
    payload = resolve_config_payload(args.overrides)
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_resolve_config_text(payload))
    return 0


def _run_forward_check(args: argparse.Namespace) -> int:
    payload = forward_check(
        args.overrides,
        requested_device=str(args.device),
        seed=int(args.seed),
    )
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_forward_check_text(payload))
    return 0


def _run_diff_config(args: argparse.Namespace) -> int:
    payload = diff_config_payloads(
        [str(value) for value in args.left],
        [str(value) for value in args.right],
    )
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_diff_config_text(payload))
    return 0


def _run_export_check(args: argparse.Namespace) -> int:
    payload = export_check(
        Path(str(args.checkpoint)),
        out_dir=None if args.out_dir is None else Path(str(args.out_dir)),
        artifact_version=str(args.artifact_version),
    )
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_export_check_text(payload))
    return 0


def _run_health_check(args: argparse.Namespace) -> int:
    payload = health_check(Path(str(args.run_dir)))
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_health_check_text(payload))
    return 0


def _run_run_inspect(args: argparse.Namespace) -> int:
    payload = run_inspect(Path(str(args.run_dir)))
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_run_inspect_text(payload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Developer tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser(
        "resolve-config",
        help="Compose one config and print the resolved build surface",
    )
    resolve_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    resolve_parser.add_argument("overrides", nargs="*", help="Hydra override strings")
    resolve_parser.set_defaults(func=_run_resolve_config)

    forward_parser = subparsers.add_parser(
        "forward-check",
        help="Build one model and run a synthetic forward-only smoke check",
    )
    forward_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    forward_parser.add_argument(
        "--device",
        choices=_DEVICE_CHOICES,
        default="auto",
        help="Execution device; defaults to auto",
    )
    forward_parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Deterministic seed for model construction and synthetic inputs",
    )
    forward_parser.add_argument("overrides", nargs="*", help="Hydra override strings")
    forward_parser.set_defaults(func=_run_forward_check)

    diff_parser = subparsers.add_parser(
        "diff-config",
        help="Compare two resolved config surfaces and print only effective differences",
    )
    diff_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    diff_parser.add_argument(
        "--left",
        action="append",
        default=[],
        help="Hydra override applied to the left-hand config",
    )
    diff_parser.add_argument(
        "--right",
        action="append",
        default=[],
        help="Hydra override applied to the right-hand config",
    )
    diff_parser.set_defaults(func=_run_diff_config)

    export_parser = subparsers.add_parser(
        "export-check",
        help="Export one checkpoint, validate the bundle, and run a reference smoke",
    )
    export_parser.add_argument("--checkpoint", required=True, help="Input training checkpoint path")
    export_parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output bundle directory; omit to use a temporary bundle",
    )
    export_parser.add_argument(
        "--artifact-version",
        default=SCHEMA_VERSION_V3,
        help="Export artifact schema version",
    )
    export_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    export_parser.set_defaults(func=_run_export_check)

    health_parser = subparsers.add_parser(
        "health-check",
        help="Summarize run telemetry and instability signals",
    )
    health_parser.add_argument("--run-dir", required=True, help="Run directory to inspect")
    health_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    health_parser.set_defaults(func=_run_health_check)

    inspect_parser = subparsers.add_parser(
        "run-inspect",
        help="Inspect one run directory, its local artifacts, and any available benchmark metadata",
    )
    inspect_parser.add_argument("--run-dir", required=True, help="Run directory to inspect")
    inspect_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    inspect_parser.set_defaults(func=_run_run_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))

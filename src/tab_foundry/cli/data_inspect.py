"""Data inspection CLI commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from tab_foundry.config import compose_config
from tab_foundry.data.inspection import inspect_manifest
from tab_foundry.data.surface import resolve_data_surface
from tab_foundry.model.inspection import model_surface_payload
from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.preprocessing import resolve_preprocessing_surface

from .dev import _mapping_from_node, _resolved_experiment_name, _training_surface_payload


def _format_jsonable(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _compatibility_overrides(
    *,
    experiment: str | None,
    overrides: Sequence[str],
) -> list[str]:
    resolved: list[str] = []
    if experiment is not None and str(experiment).strip():
        resolved.append(f"experiment={str(experiment).strip()}")
    resolved.extend(str(value) for value in overrides)
    return resolved


def _resolved_compatibility_config(overrides: Sequence[str]) -> dict[str, Any]:
    cfg = compose_config(list(overrides))
    task = str(getattr(cfg, "task", "classification")).strip().lower()
    model_cfg = _mapping_from_node(getattr(cfg, "model", None), context="cfg.model")
    spec = model_build_spec_from_mappings(task=task, primary=model_cfg)
    return {
        "experiment": _resolved_experiment_name(overrides),
        "task": task,
        "model": model_surface_payload(spec),
        "data": resolve_data_surface(_mapping_from_node(getattr(cfg, "data", None), context="cfg.data")).to_dict(),
        "preprocessing": resolve_preprocessing_surface(
            _mapping_from_node(getattr(cfg, "preprocessing", None), context="cfg.preprocessing")
        ).to_dict(),
        "training": _training_surface_payload(
            _mapping_from_node(getattr(cfg, "training", None), context="cfg.training"),
            optimizer_cfg=_mapping_from_node(getattr(cfg, "optimizer", None), context="cfg.optimizer"),
            schedule_cfg=_mapping_from_node(getattr(cfg, "schedule", None), context="cfg.schedule"),
        ),
    }


def _classification_contract(
    resolved_config: Mapping[str, Any],
) -> dict[str, int | None] | None:
    model_payload = resolved_config.get("model")
    if not isinstance(model_payload, Mapping):
        return None
    task_contract = model_payload.get("task_contract")
    if not isinstance(task_contract, Mapping):
        return None
    return {
        "min_classes": (
            None if task_contract.get("min_classes") is None else int(task_contract["min_classes"])
        ),
        "max_classes": (
            None if task_contract.get("max_classes") is None else int(task_contract["max_classes"])
        ),
    }


def _compatibility_summary(
    *,
    verdict: str,
    issues: list[str],
    warnings: list[str],
    data_source: str,
) -> str:
    if verdict == "not_applicable":
        return f"resolved data source is {data_source!r}, so manifest compatibility does not apply"
    if issues:
        return "; ".join(issues)
    if warnings:
        return "; ".join(warnings)
    return "resolved config is statically compatible with the inspected manifest"


def _manifest_compatibility(
    *,
    manifest_payload: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = Path(str(manifest_payload["manifest_path"])).expanduser().resolve()
    task = str(resolved_config.get("task", "classification"))
    data_payload = cast(Mapping[str, Any], resolved_config["data"])
    data_source = str(data_payload.get("source", "manifest"))
    resolved_manifest_path = (
        None
        if data_payload.get("manifest_path") is None
        else str(Path(str(data_payload["manifest_path"])).expanduser().resolve())
    )
    allow_missing_values = bool(data_payload.get("allow_missing_values", False))
    task_split_counts = cast(dict[str, dict[str, int]], manifest_payload["task_split_counts"])
    task_counts = cast(dict[str, int], manifest_payload["task_counts"])
    raw_task_train_test_record_counts = manifest_payload.get("task_train_test_record_counts")
    task_train_test_record_counts = (
        cast(dict[str, dict[str, int]], raw_task_train_test_record_counts)
        if isinstance(raw_task_train_test_record_counts, Mapping)
        else {}
    )
    missing_value_status_counts = cast(dict[str, int], manifest_payload["missing_value_status_counts"])
    raw_task_missing_value_status_counts = manifest_payload.get("task_missing_value_status_counts")
    task_missing_value_status_counts = (
        cast(dict[str, dict[str, int]], raw_task_missing_value_status_counts)
        if isinstance(raw_task_missing_value_status_counts, Mapping)
        else {}
    )

    issues: list[str] = []
    warnings: list[str] = []
    contract = _classification_contract(resolved_config)

    if data_source != "manifest":
        verdict = "not_applicable"
        return {
            "verdict": verdict,
            "summary": _compatibility_summary(
                verdict=verdict,
                issues=[],
                warnings=[],
                data_source=data_source,
            ),
            "task": task,
            "data_source": data_source,
            "resolved_manifest_path": resolved_manifest_path,
            "manifest_path_matches": None,
            "allow_missing_values": allow_missing_values,
            "has_task_rows": False,
            "has_train_rows": False,
            "has_test_rows": False,
            "contains_non_finite_rows": False,
            "class_contract": contract,
            "issues": [],
            "warnings": [],
        }

    manifest_path_matches = resolved_manifest_path == str(manifest_path)
    if resolved_manifest_path is None:
        issues.append("resolved manifest data surface has no manifest_path")
    elif not manifest_path_matches:
        warnings.append("resolved manifest_path does not match the inspected manifest")

    task_row_count = int(task_counts.get(task, 0))
    has_task_rows = task_row_count > 0
    task_splits = task_split_counts.get(task, {})
    task_train_test_counts = task_train_test_record_counts.get(task, {})
    has_train_rows = (
        int(task_train_test_counts.get("train", 0)) > 0
        if isinstance(task_train_test_counts, Mapping)
        else int(task_splits.get("train", 0)) > 0
    )
    has_test_rows = (
        int(task_train_test_counts.get("test", 0)) > 0
        if isinstance(task_train_test_counts, Mapping)
        else int(task_splits.get("test", 0)) > 0
    )
    task_missing_value_counts = task_missing_value_status_counts.get(task)
    contains_non_finite_rows = (
        int(task_missing_value_counts.get("contains_nan_or_inf", 0)) > 0
        if isinstance(task_missing_value_counts, Mapping)
        else int(missing_value_status_counts.get("contains_nan_or_inf", 0)) > 0
    )

    if not has_task_rows:
        issues.append(f"manifest has no rows for task={task!r}")
    if contains_non_finite_rows and not allow_missing_values:
        issues.append("manifest contains NaN or Inf rows while allow_missing_values=false")

    class_counts = manifest_payload.get("classification_n_classes")
    if task == "classification" and contract is not None:
        if isinstance(class_counts, Mapping):
            min_classes = int(class_counts["min"])
            max_classes = int(class_counts["max"])
            if contract["min_classes"] is not None and min_classes < int(contract["min_classes"]):
                issues.append(
                    "manifest classification rows fall below the resolved min_classes contract"
                )
            if contract["max_classes"] is not None and max_classes > int(contract["max_classes"]):
                issues.append(
                    "manifest classification rows exceed the resolved max_classes contract"
                )
        elif has_task_rows:
            warnings.append("manifest did not persist classification n_classes, so class-contract checks were skipped")

    verdict = "incompatible" if issues else "warn" if warnings else "compatible"
    return {
        "verdict": verdict,
        "summary": _compatibility_summary(
            verdict=verdict,
            issues=issues,
            warnings=warnings,
            data_source=data_source,
        ),
        "task": task,
        "data_source": data_source,
        "resolved_manifest_path": resolved_manifest_path,
        "manifest_path_matches": manifest_path_matches,
        "allow_missing_values": allow_missing_values,
        "has_task_rows": has_task_rows,
        "has_train_rows": has_train_rows,
        "has_test_rows": has_test_rows,
        "contains_non_finite_rows": contains_non_finite_rows,
        "class_contract": contract,
        "issues": issues,
        "warnings": warnings,
    }


def manifest_inspect_payload(
    manifest_path: Path,
    *,
    experiment: str | None,
    overrides: Sequence[str],
) -> dict[str, Any]:
    payload = inspect_manifest(manifest_path)
    config_overrides = _compatibility_overrides(experiment=experiment, overrides=overrides)
    if config_overrides:
        resolved_config = _resolved_compatibility_config(config_overrides)
        payload["compatibility"] = _manifest_compatibility(
            manifest_payload=payload,
            resolved_config=resolved_config,
        )
        payload["resolved_config"] = {
            "experiment": resolved_config.get("experiment"),
            "task": resolved_config.get("task"),
            "data": resolved_config.get("data"),
            "preprocessing": resolved_config.get("preprocessing"),
            "training": resolved_config.get("training"),
            "model": {
                "arch": cast(Mapping[str, Any], resolved_config["model"]).get("arch"),
                "stage": cast(Mapping[str, Any], resolved_config["model"]).get("stage"),
                "stage_label": cast(Mapping[str, Any], resolved_config["model"]).get("stage_label"),
                "task_contract": cast(Mapping[str, Any], resolved_config["model"]).get("task_contract"),
            },
        }
    else:
        payload["compatibility"] = None
    return payload


def render_manifest_inspect_text(payload: Mapping[str, Any]) -> str:
    lines = [
        "Manifest inspection.",
        f"manifest_path={payload['manifest_path']}",
        f"total_records={payload['total_records']}",
        f"split_counts={_format_jsonable(payload['split_counts'])}",
        f"task_counts={_format_jsonable(payload['task_counts'])}",
        f"missing_value_status_counts={_format_jsonable(payload['missing_value_status_counts'])}",
    ]
    if payload.get("n_features") is not None:
        lines.append(f"n_features={_format_jsonable(payload['n_features'])}")
    if payload.get("classification_n_classes") is not None:
        lines.append(f"classification_n_classes={_format_jsonable(payload['classification_n_classes'])}")
    persisted_summary = payload.get("persisted_summary")
    if isinstance(persisted_summary, Mapping):
        excerpt = {
            "filter_policy": persisted_summary.get("filter_policy"),
            "missing_value_policy": persisted_summary.get("missing_value_policy"),
            "excluded_records": persisted_summary.get("excluded_records"),
            "excluded_for_missing_values": persisted_summary.get("excluded_for_missing_values"),
            "dagzoo_handoff": persisted_summary.get("dagzoo_handoff"),
        }
        lines.append(f"persisted_summary={_format_jsonable(excerpt)}")
    compatibility = payload.get("compatibility")
    if isinstance(compatibility, Mapping):
        lines.append(f"compatibility={compatibility['verdict']}: {compatibility['summary']}")
        lines.append(f"compatibility.data_source={compatibility['data_source']}")
        lines.append(f"compatibility.manifest_path_matches={compatibility['manifest_path_matches']}")
        lines.append(f"compatibility.has_task_rows={compatibility['has_task_rows']}")
        lines.append(f"compatibility.has_train_rows={compatibility['has_train_rows']}")
        lines.append(f"compatibility.has_test_rows={compatibility['has_test_rows']}")
        lines.append(f"compatibility.contains_non_finite_rows={compatibility['contains_non_finite_rows']}")
        if compatibility.get("class_contract") is not None:
            lines.append(f"compatibility.class_contract={_format_jsonable(compatibility['class_contract'])}")
    return "\n".join(lines)


def _run_manifest_inspect(args: argparse.Namespace) -> int:
    payload = manifest_inspect_payload(
        Path(str(args.manifest)),
        experiment=None if args.experiment is None else str(args.experiment),
        overrides=[str(value) for value in args.override],
    )
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_manifest_inspect_text(payload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manifest inspection")
    parser.add_argument("--manifest", required=True, help="Manifest parquet path to inspect")
    parser.add_argument("--experiment", default=None, help="Optional experiment name for compatibility preflight")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional Hydra override applied on top of --experiment or repo defaults",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.set_defaults(func=_run_manifest_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))

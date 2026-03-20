"""Shared Weights & Biases helpers for training entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import DictConfig, OmegaConf


def _normalize_wandb_value(value: object) -> Any | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(float(value)) else None
    return None


def _normalized_wandb_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        resolved = _normalize_wandb_value(value)
        if resolved is not None:
            normalized[str(key)] = resolved
    return normalized


def _flatten_summary_payload(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        key_str = str(key)
        dotted_key = key_str if not prefix else f"{prefix}/{key_str}"
        if isinstance(value, Mapping):
            flattened.update(_flatten_summary_payload(value, prefix=dotted_key))
            continue
        resolved = _normalize_wandb_value(value)
        if resolved is not None:
            flattened[dotted_key] = resolved
    return flattened


def _jsonable_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _wandb_public_path_parts(run: Any | None) -> tuple[str | None, str | None, str | None]:
    if run is None:
        return None, None, None
    raw_path = getattr(run, "path", None)
    parts: list[str] = []
    if isinstance(raw_path, str):
        parts = [part.strip() for part in raw_path.split("/") if part.strip()]
    elif isinstance(raw_path, (list, tuple)):
        parts = [str(part).strip() for part in raw_path if str(part).strip()]
    if len(parts) >= 3:
        return parts[-3], parts[-2], parts[-1]
    if len(parts) == 2:
        return None, parts[0], parts[1]
    return None, None, None


def _wandb_run_value(run: Any | None, *names: str) -> Any | None:
    if run is None:
        return None
    for name in names:
        raw_value = getattr(run, name, None)
        if raw_value is not None:
            return raw_value
    settings = getattr(run, "settings", None)
    for name in names:
        raw_value = getattr(settings, name, None)
        if raw_value is not None:
            return raw_value
    return None


def resolve_wandb_api_key() -> str | None:
    value = os.getenv("WANDB_API_KEY")
    if value is not None:
        normalized = value.strip()
        if normalized:
            return normalized

    file_override = os.getenv("WANDB_API_KEY_FILE")
    candidate = (
        Path(file_override).expanduser()
        if file_override
        else Path("~/.wandb/wandb_api_key.txt").expanduser()
    )
    try:
        normalized = candidate.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not normalized:
        return None
    os.environ["WANDB_API_KEY"] = normalized
    return normalized


def init_wandb_run(cfg: DictConfig, *, enabled: bool) -> Any | None:
    if not enabled:
        return None
    try:
        import wandb
    except Exception:
        return None

    api_key = resolve_wandb_api_key()
    mode: Literal["online", "offline"] = "online" if api_key else "offline"
    cfg_payload = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    return wandb.init(
        project=str(cfg.logging.project),
        name=str(cfg.logging.run_name),
        mode=mode,
        config=cfg_payload,
    )


def wandb_identity_payload(
    run: Any | None,
    *,
    cfg: DictConfig | None = None,
) -> dict[str, Any] | None:
    path_entity, path_project, path_run_id = _wandb_public_path_parts(run)
    project_fallback = None
    run_name_fallback = None
    if cfg is not None:
        logging_cfg = cfg.get("logging")
        if logging_cfg is not None:
            project_raw = getattr(logging_cfg, "project", None)
            if project_raw is not None:
                project_fallback = str(project_raw).strip() or None
            run_name_raw = getattr(logging_cfg, "run_name", None)
            if run_name_raw is not None:
                run_name_fallback = str(run_name_raw).strip() or None

    entity_raw = _wandb_run_value(run, "entity")
    project_raw = _wandb_run_value(run, "project")
    run_id_raw = _wandb_run_value(run, "id")
    run_name_raw = _wandb_run_value(run, "name")
    mode_raw = _wandb_run_value(run, "mode")

    metadata = {
        "entity": path_entity if entity_raw is None else str(entity_raw).strip() or None,
        "project": path_project if project_raw is None else str(project_raw).strip() or None,
        "run_id": path_run_id if run_id_raw is None else str(run_id_raw).strip() or None,
        "run_name": run_name_fallback if run_name_raw is None else str(run_name_raw).strip() or None,
        "mode": None if mode_raw is None else str(mode_raw).strip() or None,
    }
    if metadata["project"] is None:
        metadata["project"] = project_fallback
    if metadata["run_name"] is None:
        metadata["run_name"] = run_name_fallback
    if not any(value is not None for value in metadata.values()):
        return None
    return metadata


def training_surface_wandb_summary_payload(
    training_surface_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(training_surface_record, Mapping):
        return {}
    summary: dict[str, Any] = {"surface": {}}
    labels = _jsonable_mapping(training_surface_record.get("labels"))
    if labels:
        summary["surface"]["labels"] = labels
    raw_model = _jsonable_mapping(training_surface_record.get("model"))
    if raw_model:
        model_summary: dict[str, Any] = {}
        for key in (
            "arch",
            "stage",
            "stage_label",
            "benchmark_profile",
            "input_normalization",
            "feature_group_size",
            "many_class_base",
        ):
            if key in raw_model:
                model_summary[key] = raw_model[key]
        module_selection = _jsonable_mapping(raw_model.get("module_selection"))
        if module_selection:
            model_summary["module_selection"] = module_selection
        module_hyperparameters = _jsonable_mapping(raw_model.get("module_hyperparameters"))
        if module_hyperparameters:
            model_summary["module_hyperparameters"] = module_hyperparameters
        if model_summary:
            summary["surface"]["model"] = model_summary
    return summary if summary["surface"] else {}


def log_wandb_metrics(run: Any | None, payload: Mapping[str, Any], *, step: int) -> None:
    if run is None:
        return
    log = getattr(run, "log", None)
    if not callable(log):
        return
    normalized = _normalized_wandb_payload(payload)
    if not normalized:
        return
    log(normalized, step=int(step))


def update_wandb_summary(run: Any | None, payload: Mapping[str, Any]) -> None:
    if run is None:
        return
    summary = getattr(run, "summary", None)
    if summary is None:
        return
    for key, value in _flatten_summary_payload(payload).items():
        try:
            summary[key] = value
        except Exception:
            continue


def finish_wandb_run(run: Any | None) -> None:
    if run is None:
        return
    finish = getattr(run, "finish", None)
    if callable(finish):
        finish()


def _telemetry_wandb_payload(telemetry_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(telemetry_path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    telemetry = _jsonable_mapping(payload)
    if telemetry is None:
        return None
    return _jsonable_mapping(telemetry.get("wandb"))


def posthoc_update_wandb_summary(
    *,
    telemetry_path: Path,
    payload: Mapping[str, Any],
) -> bool:
    metadata = _telemetry_wandb_payload(telemetry_path)
    if metadata is None:
        return False
    mode = str(metadata.get("mode") or "").strip().lower()
    if mode == "offline":
        return False
    entity = str(metadata.get("entity") or "").strip()
    project = str(metadata.get("project") or "").strip()
    run_id = str(metadata.get("run_id") or "").strip()
    if project and run_id:
        run_path = f"{project}/{run_id}" if not entity else f"{entity}/{project}/{run_id}"
    else:
        return False
    flattened = _flatten_summary_payload(payload)
    if not flattened:
        return False
    _ = resolve_wandb_api_key()
    try:
        import wandb
    except Exception:
        return False
    try:
        api = wandb.Api()
        api_run = api.run(run_path)
        for key, value in flattened.items():
            api_run.summary[key] = value
        update = getattr(api_run.summary, "update", None)
        if callable(update):
            update()
    except Exception:
        return False
    return True

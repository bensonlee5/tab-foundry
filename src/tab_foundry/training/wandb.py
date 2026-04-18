"""Shared Weights & Biases helpers for training entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from netrc import NetrcParseError, netrc
import os
from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import DictConfig, OmegaConf
from tab_foundry.hashing import sha256_text


_WANDB_ARTIFACT_NAME_MAXLEN = 128
_WANDB_ARTIFACT_NAME_HASH_HEX = 12


@dataclass(frozen=True, slots=True)
class WandbArtifactReference:
    """One uploaded or downloaded W&B artifact reference."""

    artifact_name: str
    artifact_ref: str
    local_path: Path


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


def _normalize_artifact_name_for_wandb(name: str) -> str:
    normalized_name = str(name).strip()
    if not normalized_name:
        raise RuntimeError("artifact_name must be a non-empty string for W&B checkpoint publication")
    if len(normalized_name) <= _WANDB_ARTIFACT_NAME_MAXLEN:
        return normalized_name
    hash_suffix = sha256_text(normalized_name)[:_WANDB_ARTIFACT_NAME_HASH_HEX]
    max_prefix_chars = _WANDB_ARTIFACT_NAME_MAXLEN - 1 - len(hash_suffix)
    if max_prefix_chars <= 0:  # pragma: no cover - defensive guard
        raise RuntimeError("W&B artifact name budget is too small to encode a checkpoint artifact")
    return f"{normalized_name[:max_prefix_chars]}-{hash_suffix}"


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


def _read_wandb_api_key_file(candidate: Path) -> str | None:
    try:
        normalized = candidate.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return normalized or None


def _read_wandb_api_key_netrc(candidate: Path) -> str | None:
    try:
        auth_file = netrc(str(candidate))
    except (FileNotFoundError, NetrcParseError, OSError):
        return None
    for machine in ("api.wandb.ai", "wandb.ai"):
        credentials = auth_file.authenticators(machine)
        if credentials is None:
            continue
        _login, _account, password = credentials
        if password is None:
            continue
        normalized = str(password).strip()
        if normalized:
            return normalized
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
    resolved_key = _read_wandb_api_key_file(candidate)
    if resolved_key is None:
        resolved_key = _read_wandb_api_key_netrc(Path("~/.netrc").expanduser())
    if resolved_key is None:
        return None
    os.environ["WANDB_API_KEY"] = resolved_key
    return resolved_key


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
    init_kwargs: dict[str, Any] = {
        "project": str(cfg.logging.project),
        "name": str(cfg.logging.run_name),
        "mode": mode,
        "config": cfg_payload,
    }
    group_raw = getattr(cfg.logging, "group", None)
    if group_raw is not None:
        group = str(group_raw).strip()
        if group:
            init_kwargs["group"] = group
    return wandb.init(
        **init_kwargs,
    )


def wandb_identity_payload(
    run: Any | None,
    *,
    cfg: DictConfig | None = None,
) -> dict[str, Any] | None:
    path_entity, path_project, path_run_id = _wandb_public_path_parts(run)
    project_fallback = None
    run_name_fallback = None
    group_fallback = None
    if cfg is not None:
        logging_cfg = cfg.get("logging")
        if logging_cfg is not None:
            project_raw = getattr(logging_cfg, "project", None)
            if project_raw is not None:
                project_fallback = str(project_raw).strip() or None
            run_name_raw = getattr(logging_cfg, "run_name", None)
            if run_name_raw is not None:
                run_name_fallback = str(run_name_raw).strip() or None
            group_raw = getattr(logging_cfg, "group", None)
            if group_raw is not None:
                group_fallback = str(group_raw).strip() or None

    entity_raw = _wandb_run_value(run, "entity")
    project_raw = _wandb_run_value(run, "project")
    run_id_raw = _wandb_run_value(run, "id")
    run_name_raw = _wandb_run_value(run, "name")
    mode_raw = _wandb_run_value(run, "mode")
    group_raw = _wandb_run_value(run, "group")

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
    group = group_fallback if group_raw is None else str(group_raw).strip() or None
    if group is not None:
        metadata["group"] = group
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


def _require_wandb_sdk() -> Any:
    _ = resolve_wandb_api_key()
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - import failure depends on environment
        raise RuntimeError("wandb is required for checkpoint artifact publication") from exc
    return wandb


def _normalize_artifact_metadata(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    normalized = _normalized_wandb_payload(payload)
    return normalized or None


def publish_checkpoint_artifact(
    *,
    checkpoint_path: Path,
    artifact_name: str,
    entity: str | None,
    project: str,
    run_id: str,
    run_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    aliases: list[str] | None = None,
) -> WandbArtifactReference:
    """Upload one checkpoint as a versioned W&B artifact and return its exact ref."""

    resolved_checkpoint = checkpoint_path.expanduser().resolve()
    if not resolved_checkpoint.exists():
        raise RuntimeError(f"checkpoint path does not exist: {resolved_checkpoint}")
    normalized_project = str(project).strip()
    normalized_run_id = str(run_id).strip()
    normalized_artifact_name = _normalize_artifact_name_for_wandb(artifact_name)
    if not normalized_project:
        raise RuntimeError("project must be a non-empty string for W&B checkpoint publication")
    if not normalized_run_id:
        raise RuntimeError("run_id must be a non-empty string for W&B checkpoint publication")
    normalized_entity = None if entity is None else str(entity).strip() or None
    normalized_run_name = None if run_name is None else str(run_name).strip() or None
    normalized_aliases = [str(value).strip() for value in (aliases or ["best"]) if str(value).strip()]
    if not normalized_aliases:
        normalized_aliases = ["best"]

    wandb = _require_wandb_sdk()
    init_kwargs: dict[str, Any] = {
        "project": normalized_project,
        "id": normalized_run_id,
        "resume": "allow",
        "job_type": "benchmark-checkpoint-publish",
        "mode": "online",
    }
    if normalized_entity is not None:
        init_kwargs["entity"] = normalized_entity
    if normalized_run_name is not None:
        init_kwargs["name"] = normalized_run_name

    run = wandb.init(**init_kwargs)
    if run is None:  # pragma: no cover - defensive branch
        raise RuntimeError("wandb.init returned no run while publishing checkpoint artifact")
    try:
        artifact = wandb.Artifact(
            name=normalized_artifact_name,
            type="model",
            metadata=_normalize_artifact_metadata(metadata),
        )
        artifact.add_file(str(resolved_checkpoint), name="best.pt")
        logged_artifact = run.log_artifact(artifact, aliases=normalized_aliases)
        logged_artifact = logged_artifact.wait()
        resolved_ref = str(getattr(logged_artifact, "name", "")).strip()
        if resolved_ref:
            path_parts = [part for part in resolved_ref.split("/") if part]
            if len(path_parts) == 1:
                base = f"{normalized_project}/{resolved_ref}"
                resolved_ref = base if normalized_entity is None else f"{normalized_entity}/{base}"
            elif len(path_parts) == 2 and normalized_entity is not None:
                resolved_ref = f"{normalized_entity}/{resolved_ref}"
        else:
            base = f"{normalized_project}/{normalized_artifact_name}:{normalized_aliases[0]}"
            resolved_ref = base if normalized_entity is None else f"{normalized_entity}/{base}"
        return WandbArtifactReference(
            artifact_name=normalized_artifact_name,
            artifact_ref=resolved_ref,
            local_path=resolved_checkpoint,
        )
    finally:
        finish_wandb_run(run)


def download_checkpoint_artifact(
    *,
    artifact_ref: str,
    out_dir: Path,
) -> WandbArtifactReference:
    """Download one checkpoint artifact and return the resolved local best.pt path."""

    normalized_ref = str(artifact_ref).strip()
    if not normalized_ref:
        raise RuntimeError("artifact_ref must be a non-empty string")
    resolved_out_dir = out_dir.expanduser().resolve()
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    cached_checkpoint = resolved_out_dir / "best.pt"
    if cached_checkpoint.exists():
        return WandbArtifactReference(
            artifact_name=normalized_ref.split("/")[-1].split(":")[0],
            artifact_ref=normalized_ref,
            local_path=cached_checkpoint,
        )

    wandb = _require_wandb_sdk()
    try:
        api = wandb.Api()
        artifact = api.artifact(normalized_ref)
        downloaded_root = Path(artifact.download(root=str(resolved_out_dir))).expanduser().resolve()
    except Exception as exc:  # pragma: no cover - network/API behavior
        raise RuntimeError(f"failed to download W&B artifact {normalized_ref!r}") from exc

    candidate = downloaded_root / "best.pt"
    if not candidate.exists():
        matches = sorted(downloaded_root.rglob("best.pt"))
        if len(matches) != 1:
            raise RuntimeError(
                "downloaded W&B artifact does not contain exactly one best.pt: "
                f"artifact_ref={normalized_ref} download_root={downloaded_root}"
            )
        candidate = matches[0]
    return WandbArtifactReference(
        artifact_name=normalized_ref.split("/")[-1].split(":")[0],
        artifact_ref=normalized_ref,
        local_path=candidate.resolve(),
    )


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

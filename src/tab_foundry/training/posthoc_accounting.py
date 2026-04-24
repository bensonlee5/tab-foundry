"""Posthoc training-run accounting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.checkpoint_state import normalize_checkpoint_model_state_dict
from tab_foundry.model.inspection import compute_accounting_from_model_spec
from tab_foundry.model.spec import checkpoint_model_build_spec_from_mappings
from tab_foundry.training.checkpoint_paths import resolve_latest_checkpoint_path


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else None


def _strict_or_none(message: str, *, strict: bool) -> None:
    if strict:
        raise RuntimeError(message)
    return None


def training_shape_summary_from_telemetry(
    telemetry_payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(telemetry_payload, Mapping):
        return None
    raw_summary = telemetry_payload.get("training_shape_summary")
    if not isinstance(raw_summary, Mapping):
        return None
    return dict(cast(Mapping[str, Any], raw_summary))


def regime_budget_from_telemetry(
    telemetry_payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(telemetry_payload, Mapping):
        return None
    raw_budget = telemetry_payload.get("regime_budget")
    if not isinstance(raw_budget, Mapping):
        return None
    return dict(cast(Mapping[str, Any], raw_budget))


def compute_accounting_from_checkpoint_config(
    raw_cfg: Mapping[str, Any],
    *,
    state_dict: Mapping[str, Any] | None,
    telemetry_payload: Mapping[str, Any] | None,
    strict: bool = True,
) -> dict[str, Any] | None:
    raw_task = raw_cfg.get("task")
    if not isinstance(raw_task, str) or not raw_task.strip():
        _strict_or_none("checkpoint config.task must be a non-empty string", strict=strict)
        return None
    raw_model_cfg = raw_cfg.get("model")
    if not isinstance(raw_model_cfg, Mapping):
        _strict_or_none("checkpoint config must include a model mapping", strict=strict)
        return None
    model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    raw_arch = model_cfg.get("arch")
    if not isinstance(raw_arch, str) or not raw_arch.strip():
        _strict_or_none(
            "checkpoint model config must include explicit model.arch metadata",
            strict=strict,
        )
        return None

    regime_budget = regime_budget_from_telemetry(telemetry_payload)
    if regime_budget is None:
        return None
    tokens_seen_raw = regime_budget.get("tokens_seen")
    tokens_per_step_raw = regime_budget.get("tokens_per_step")
    tokens_seen = None if tokens_seen_raw is None else int(tokens_seen_raw)
    tokens_per_step = None if tokens_per_step_raw is None else float(tokens_per_step_raw)
    model_spec = checkpoint_model_build_spec_from_mappings(
        task=str(raw_task),
        primary=model_cfg,
        state_dict=state_dict,
    )
    return compute_accounting_from_model_spec(
        model_spec,
        training_shape_summary=training_shape_summary_from_telemetry(telemetry_payload),
        tokens_seen=tokens_seen,
        tokens_per_step=tokens_per_step,
    )


def resolve_posthoc_checkpoint_path(run_dir: Path) -> Path | None:
    resolved_run_dir = run_dir.expanduser().resolve()
    best_path = resolved_run_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        return best_path
    latest_path = resolve_latest_checkpoint_path(resolved_run_dir)
    return latest_path if latest_path is not None and latest_path.exists() else None


def load_checkpoint_config_and_state(
    checkpoint_path: Path,
    *,
    strict: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
    import torch

    resolved_path = checkpoint_path.expanduser().resolve()
    payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        _strict_or_none(f"checkpoint payload must be a mapping: {resolved_path}", strict=strict)
        return None
    raw_cfg = payload.get("config")
    if not isinstance(raw_cfg, Mapping):
        _strict_or_none(f"checkpoint config must be a mapping: {resolved_path}", strict=strict)
        return None
    raw_state_dict = payload.get("model")
    state_dict = None
    if raw_state_dict is not None:
        if not isinstance(raw_state_dict, Mapping):
            _strict_or_none(
                f"checkpoint model state_dict must be a mapping: {resolved_path}",
                strict=strict,
            )
            return None
        state_dict = normalize_checkpoint_model_state_dict(
            raw_state_dict,
            checkpoint_path=resolved_path,
        )
    return ({str(key): value for key, value in raw_cfg.items()}, state_dict)


def derive_compute_accounting_for_run(
    run_dir: Path,
    *,
    telemetry_payload: Mapping[str, Any] | None,
    strict: bool = False,
) -> dict[str, Any] | None:
    checkpoint_path = resolve_posthoc_checkpoint_path(run_dir)
    if checkpoint_path is None:
        _strict_or_none(
            f"no best/latest checkpoint found for posthoc accounting under {run_dir}",
            strict=strict,
        )
        return None
    loaded = load_checkpoint_config_and_state(checkpoint_path, strict=strict)
    if loaded is None:
        return None
    raw_cfg, state_dict = loaded
    return compute_accounting_from_checkpoint_config(
        raw_cfg,
        state_dict=state_dict,
        telemetry_payload=telemetry_payload,
        strict=strict,
    )

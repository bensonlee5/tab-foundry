"""Shared Weights & Biases helpers for training entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
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

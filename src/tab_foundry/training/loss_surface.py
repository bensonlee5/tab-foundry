"""Training loss-surface resolution helpers."""

from __future__ import annotations

from typing import Any, Mapping

from tab_foundry.model.spec import ModelBuildSpec, SANDWICH_MODEL_ARCH


CLASSIFICATION_LOSS_SURFACE = "classification"
CELL_BPC_LOSS_SURFACE = "cell_bpc"
SUPPORTED_TRAINING_LOSS_SURFACES = (
    CLASSIFICATION_LOSS_SURFACE,
    CELL_BPC_LOSS_SURFACE,
)


def normalize_training_loss_surface(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized not in SUPPORTED_TRAINING_LOSS_SURFACES:
        raise ValueError(
            "training.loss_surface must be one of "
            f"{SUPPORTED_TRAINING_LOSS_SURFACES}, got {value!r}"
        )
    return normalized


def resolve_training_loss_surface(
    training_cfg: Mapping[str, Any] | Any,
    *,
    model_spec: ModelBuildSpec,
    backend: str | None = None,
) -> str:
    raw_value = None
    if isinstance(training_cfg, Mapping):
        raw_value = training_cfg.get("loss_surface")
    elif training_cfg is not None:
        raw_value = getattr(training_cfg, "loss_surface", None)
    explicit = normalize_training_loss_surface(raw_value)
    if explicit is not None:
        return explicit
    if str(model_spec.arch).strip().lower() == SANDWICH_MODEL_ARCH and backend == "legacy_prior":
        return CELL_BPC_LOSS_SURFACE
    return CLASSIFICATION_LOSS_SURFACE


def configure_model_loss_surface(model: Any, *, loss_surface: str) -> None:
    resolved = normalize_training_loss_surface(loss_surface)
    if resolved is None:
        raise ValueError("loss_surface must resolve to one supported value")
    setter = getattr(model, "set_loss_surface", None)
    if callable(setter):
        setter(resolved)
        return
    if resolved != CLASSIFICATION_LOSS_SURFACE:
        raise RuntimeError(
            "requested non-classification loss surface for a model without set_loss_surface(): "
            f"{resolved!r}"
        )

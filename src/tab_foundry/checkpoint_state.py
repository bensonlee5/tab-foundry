"""Shared checkpoint state-dict normalization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def normalize_checkpoint_model_state_dict(
    model_state: Mapping[str, Any],
    *,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Strip compile-wrapper prefixes from checkpoint state-dict keys."""

    normalized: dict[str, Any] = {}
    normalized_sources: dict[str, str] = {}
    for raw_key, value in model_state.items():
        source_key = str(raw_key)
        normalized_key = source_key
        while normalized_key.startswith("_orig_mod."):
            normalized_key = normalized_key.removeprefix("_orig_mod.")
        existing_source = normalized_sources.get(normalized_key)
        if existing_source is not None and existing_source != source_key:
            location = "" if checkpoint_path is None else f" in checkpoint {checkpoint_path}"
            raise RuntimeError(
                "compiled checkpoint state_dict normalization produced duplicate key "
                f"{normalized_key!r} from {existing_source!r} and {source_key!r}{location}"
            )
        normalized[normalized_key] = value
        normalized_sources[normalized_key] = source_key
    return normalized

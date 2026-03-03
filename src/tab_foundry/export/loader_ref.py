"""Reference bundle loader for contract tests and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file
from torch import nn

from tab_foundry.model.factory import build_model

from .contracts import ValidatedBundle
from .exporter import validate_export_bundle


@dataclass(slots=True)
class LoadedExportBundle:
    validated: ValidatedBundle
    model: nn.Module


def load_export_bundle(bundle_dir: Path) -> LoadedExportBundle:
    """Load and validate an exported bundle into a model instance."""

    validated = validate_export_bundle(bundle_dir)
    manifest = validated.manifest

    model = build_model(
        task=manifest.task,
        d_col=manifest.model.d_col,
        d_icl=manifest.model.d_icl,
        feature_group_size=manifest.model.feature_group_size,
        many_class_train_mode=validated.inference_config.many_class_inference_mode,
        max_mixed_radix_digits=manifest.model.max_mixed_radix_digits,
    )
    weights_path = bundle_dir.expanduser().resolve() / manifest.files.weights
    state_dict = load_file(str(weights_path))
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Failed to load exported weights strictly: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return LoadedExportBundle(validated=validated, model=model)

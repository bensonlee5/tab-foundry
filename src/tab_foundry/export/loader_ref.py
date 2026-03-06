"""Reference bundle loader for contract tests and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file
from torch import nn

from tab_foundry.model.factory import ModelBuildSpec, build_model_from_spec

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

    model_spec = ModelBuildSpec(
        task=manifest.task,
        d_col=manifest.model.d_col,
        d_icl=manifest.model.d_icl,
        feature_group_size=manifest.model.feature_group_size,
        many_class_train_mode=manifest.model.many_class_train_mode,
        max_mixed_radix_digits=manifest.model.max_mixed_radix_digits,
        tfcol_n_heads=manifest.model.tfcol_n_heads,
        tfcol_n_layers=manifest.model.tfcol_n_layers,
        tfcol_n_inducing=manifest.model.tfcol_n_inducing,
        tfrow_n_heads=manifest.model.tfrow_n_heads,
        tfrow_n_layers=manifest.model.tfrow_n_layers,
        tfrow_cls_tokens=manifest.model.tfrow_cls_tokens,
        tficl_n_heads=manifest.model.tficl_n_heads,
        tficl_n_layers=manifest.model.tficl_n_layers,
        tficl_ff_expansion=manifest.model.tficl_ff_expansion,
        many_class_base=manifest.model.many_class_base,
        head_hidden_dim=manifest.model.head_hidden_dim,
        use_digit_position_embed=manifest.model.use_digit_position_embed,
    )
    model = build_model_from_spec(model_spec)
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

# Inference Contract

This repository is the training-side producer of inference artifacts. Runtime inference is expected to live in a separate repository.

Related docs:

- quickstart: `README.md`
- workflow runbooks: `docs/workflows.md`
- design decisions and repo structure: `docs/development/design-decisions.md`
- architecture reference: `docs/development/model-architecture.md`
- canonical roadmap: `docs/development/roadmap.md`

Planning and architecture rationale live under `docs/development/`.
This file stays top-level because it is the stable export and validation
contract.

## Schema Version

Current version: `tab-foundry-export-v2`

Any breaking change must increment the schema version.

## Bundle Layout

An export bundle is a directory containing:

- `manifest.json`
- `weights.safetensors`
- `inference_config.json`
- `preprocessor_state.json`

## manifest.json

Required keys:

- `schema_version`
- `producer`: `{name, version, git_sha}` (`git_sha` may be `null`)
- `task`: `classification | regression`
- `model`: `{arch, d_col, d_icl, feature_group_size, many_class_train_mode, max_mixed_radix_digits}`
  - Exporter also emits architecture reconstruction fields:
    `{tfcol_n_heads, tfcol_n_layers, tfcol_n_inducing, tfrow_n_heads, tfrow_n_layers, tfrow_cls_tokens, tficl_n_heads, tficl_n_layers, tficl_ff_expansion, many_class_base, head_hidden_dim, use_digit_position_embed}`.
  - Validators accept manifests that omit these extra fields and apply the current model defaults.
- `files`: `{weights, inference_config, preprocessor_state}`
- `checksums`: sha256 for `weights`, `inference_config`, `preprocessor_state`
- `created_at_utc`: ISO8601 UTC timestamp

## inference_config.json

Required keys:

- `task`
- `model_arch` (`tabfoundry`)
- `group_shifts` (`[0, 1, 3]`)
- `feature_group_size`
- `many_class_threshold` (`10`)
- `many_class_inference_mode` (`full_probs`)

Regression-only additional key:

- `quantile_levels` (length 999)

## preprocessor_state.json

Required keys:

- `feature_order_policy` (`lexicographic_f_columns`)
- `missing_value_policy` (`strategy=train_mean`, `all_nan_fill=0.0`)
- `classification_label_policy` (`mapping=train_only_remap`, `unseen_test_label=filter`)
- `dtype_policy` (`features=float32`, `classification_labels=int64`, `regression_targets=float32`)

## Many-Class Modes

- `manifest.model.many_class_train_mode` configures the training-time branch behavior (`path_nll` vs `full_probs`) and is used when reconstructing the model from an export bundle.
- `inference_config.many_class_inference_mode` is an inference-runtime contract field and is currently fixed to `full_probs`.

## Producer Commands

Export from checkpoint:

```bash
uv run tab-foundry export \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  --out-dir outputs/exports/cls_smoke_v2
```

Validate bundle:

```bash
uv run tab-foundry validate-export --bundle-dir outputs/exports/cls_smoke_v2
```

## Compatibility Policy

- Training checkpoints remain unchanged and are still used for resume/training workflows.
- Export bundles are the cross-repo inference handoff contract.
- `tab-foundry-export-v1` bundles are intentionally unsupported after the `tabfoundry` family rename and must be regenerated as `tab-foundry-export-v2`.
- If a future version changes schema in a non-backward-compatible way, it must use a new `schema_version`.

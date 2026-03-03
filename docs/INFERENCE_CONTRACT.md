# Inference Export Contract

This repository is the training-side producer of inference artifacts. Runtime inference is expected to live in a separate repository.

## Schema Version

Current version: `tab-foundry-export-v1`

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
- `files`: `{weights, inference_config, preprocessor_state}`
- `checksums`: sha256 for `weights`, `inference_config`, `preprocessor_state`
- `created_at_utc`: ISO8601 UTC timestamp

## inference_config.json

Required keys:

- `task`
- `model_arch` (`tabiclv2`)
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
- `classification_label_policy` (`mapping=train_only_remap`, `unseen_test_label=error`)
- `dtype_policy` (`features=float32`, `classification_labels=int64`, `regression_targets=float32`)

## Producer Commands

Export from checkpoint:

```bash
uv run tab-foundry export \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  --out-dir outputs/exports/cls_smoke_v1
```

Validate bundle:

```bash
uv run tab-foundry validate-export --bundle-dir outputs/exports/cls_smoke_v1
```

## Compatibility Policy

- Training checkpoints remain unchanged and are still used for resume/training workflows.
- Export bundles are additive and are the cross-repo inference handoff contract.
- If a future version changes schema in a non-backward-compatible way, it must use a new `schema_version`.

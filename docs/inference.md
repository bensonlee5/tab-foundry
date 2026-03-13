# Inference Contract

This repository is the training-side producer of inference artifacts. Runtime
inference is expected to live in a separate repository.

Related docs:

- quickstart: `README.md`
- workflow runbooks: `docs/workflows.md`
- design decisions and repo structure: `docs/development/design-decisions.md`
- architecture reference: `docs/development/model-architecture.md`
- model config reference: `docs/development/model-config.md`
- canonical roadmap: `docs/development/roadmap.md`

Planning and architecture rationale live under `docs/development/`.
This file stays top-level because it is the stable export and validation
contract.

## Schema Version

Current version: `tab-foundry-export-v3`

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
- `model`: `{arch, d_col, d_icl, input_normalization, feature_group_size, many_class_train_mode, max_mixed_radix_digits}`
  - Exporter also emits architecture reconstruction fields:
    `{tfcol_n_heads, tfcol_n_layers, tfcol_n_inducing, tfrow_n_heads, tfrow_n_layers, tfrow_cls_tokens, tficl_n_heads, tficl_n_layers, tficl_ff_expansion, many_class_base, head_hidden_dim, use_digit_position_embed}`.
  - Validators accept manifests that omit the optional reconstruction fields and
    apply the current model defaults.
  - See `docs/development/model-config.md` for the meaning of each model field
    and the current canonical defaults.
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

- `quantile_levels` (length `999`)

## preprocessor_state.json

### v3 executable contract

Required keys:

- `feature_order_policy` (`positional_feature_ids`)
- `feature_ids` (ordered positional integer ids)
- `missing_value_policy`
  - `strategy=train_mean`
  - `all_nan_fill=0.0`
  - `fill_values` (one persisted train-mean fill value per feature)
- `classification_label_policy`
  - classification bundles: `{mapping=train_only_remap, unseen_test_label=filter, label_values=[...] }`
  - regression bundles: `null`
- `dtype_policy` (`features=float32`, `classification_labels=int64`, `regression_targets=float32`)

### v2 read-only compatibility

`tab-foundry-export-v2` remains validator-readable during migration, but it is
not the executable reference-consumer contract. Its `preprocessor_state.json`
contains only policy metadata:

- `feature_order_policy` (`lexicographic_f_columns`)
- `missing_value_policy` (`strategy=train_mean`, `all_nan_fill=0.0`)
- `classification_label_policy` (`mapping=train_only_remap`, `unseen_test_label=filter`)
- `dtype_policy` (`features=float32`, `classification_labels=int64`, `regression_targets=float32`)

## Many-Class Modes

- `manifest.model.many_class_train_mode` configures the training-time branch
  behavior (`path_nll` vs `full_probs`) and is used when reconstructing the
  model from an export bundle.
- `inference_config.many_class_inference_mode` is an inference-runtime contract
  field and is currently fixed to `full_probs`.

## Producer Commands

Build fitted preprocessing state for one manifest dataset:

```bash
uv run tab-foundry build-preprocessor-state \
  --manifest-path outputs/manifests/cls_smoke.parquet \
  --dataset-id root_deadbeef0000/shard_00000/dataset_000000_deadbeef0000 \
  --task classification \
  --out-path outputs/exports/cls_smoke_preprocessor_state.json
```

Export from checkpoint:

```bash
uv run tab-foundry export \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  --preprocessor-state outputs/exports/cls_smoke_preprocessor_state.json \
  --out-dir outputs/exports/cls_smoke_v3
```

Validate bundle:

```bash
uv run tab-foundry validate-export \
  --bundle-dir outputs/exports/cls_smoke_v3
```

## Reference Consumer

This repo includes a reference-only executable consumer in
`tab_foundry.export.loader_ref`.

Scope:

- dense numeric matrices in
- persisted checksum and schema validation
- persisted preprocessing application
- model-native outputs out (`class_probs` for classification,
  `quantiles` plus `quantile_levels` for regression)

Out of scope here:

- serving APIs
- dataframe adapters
- long-lived runtime ownership
- generalized production inference policy beyond the separate-repo handoff

## Compatibility Policy

- Training checkpoints remain unchanged and are still used for resume/training
  workflows.
- Export bundles are the cross-repo inference handoff contract.
- `tab-foundry-export-v3` is now the default export format and the only
  executable reference-consumer contract in this repo.
- `tab-foundry-export-v3` export now requires an explicit task-scoped
  `--preprocessor-state` JSON. Training checkpoints do not persist fitted task
  preprocessing state.
- `tab-foundry-export-v2` bundles remain validator-readable during migration.
- `tab-foundry-export-v1` bundles are intentionally unsupported after the
  `tabfoundry` family rename and must be regenerated.
- Checkpoint export/load now treats omitted `feature_group_size` as `1`. Legacy
  grouped-token checkpoints that omitted that field are intentionally rejected
  and must be regenerated or loaded with an explicit `feature_group_size`
  override before export.
- Because the v3 schema persists fitted preprocessing state and
  `manifest.model.input_normalization`, this is a user-facing artifact-contract
  change. Bundles produced under v2 should be regenerated if they need
  executable reference consumption.

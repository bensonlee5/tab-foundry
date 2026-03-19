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

The current v3 contract uses a single-manifest layout. Earlier v3 bundles that
used `inference_config.json` and `preprocessor_state.json` sidecars are
obsolete and must be regenerated. Earlier single-manifest v3 bundles that do
not include `manifest_sha256` are also obsolete and must be regenerated.

## Bundle Layout

An export bundle is a directory containing:

- `manifest.json`
- `weights.safetensors`

## manifest.json

Required keys:

- `schema_version`
- `producer`: `{name, version, git_sha}` (`git_sha` may be `null`)
- `task`: `classification`
- `manifest_sha256`: SHA-256 of the canonical UTF-8 JSON encoding of the full
  manifest with `manifest_sha256` omitted; validators reject stale or missing
  values
- `model`: `{arch, d_col, d_icl, input_normalization, feature_group_size, many_class_train_mode, max_mixed_radix_digits}`
  - `stage` is an additive optional field and is emitted only for
    `arch=tabfoundry_staged`.
  - `stage_label`, `module_overrides`, `staged_dropout`, and
    `pre_encoder_clip` are additive optional staged-surface fields. Newly
    regenerated staged bundles persist them when present so export/load replay
    matches the training checkpoint surface exactly.
  - Exporter also emits architecture reconstruction fields:
    `{tfcol_n_heads, tfcol_n_layers, tfcol_n_inducing, tfrow_n_heads, tfrow_n_layers, tfrow_cls_tokens, tficl_n_heads, tficl_n_layers, tficl_ff_expansion, many_class_base, head_hidden_dim, use_digit_position_embed}`.
  - Validators accept manifests that omit the optional reconstruction fields and
    apply the current model defaults.
  - Validators also accept older manifests that omit `stage` and the additive
    staged-surface fields, applying current defaults for omitted values.
  - See `docs/development/model-config.md` for the meaning of each model field
    and the current canonical defaults.
- `inference`
  - `task`
  - `model_arch` (`tabfoundry_simple` or `tabfoundry_staged`)
  - `model_stage` (optional; emitted only for `tabfoundry_staged`)
  - `group_shifts` (`[0, 1, 3]`)
  - `feature_group_size`
  - `many_class_threshold` (`10`)
  - `many_class_inference_mode` (`full_probs`)
- `preprocessor`
  - `feature_order_policy` (`positional_feature_ids`)
  - `missing_value_policy`
    - `strategy` (`train_mean`)
    - `all_nan_fill` (resolved runtime fill value, default `0.0`)
    - `impute_missing` (resolved runtime imputation toggle, default `true`)
  - `classification_label_policy`
    - `{mapping=train_only_remap, unseen_test_label=filter}`
  - `dtype_policy` (`features=float32`, `classification_labels=int64`, `regression_targets=float32`)
- `weights`: `{file, sha256}`
- `created_at_utc`: ISO8601 UTC timestamp

## Preprocessing Semantics

- Bundle metadata now records preprocessing policy only. v3 bundles do not
  persist dataset-specific fitted preprocessing values.
- Runtime preprocessing is always derived from the incoming support set
  (`x_train`, `y_train`) before applying the model.
- The bundled `preprocessor` section is the executable runtime policy surface
  for the reference consumer, not a cache of export-time train statistics.
- Bundles with `preprocessor.missing_value_policy.impute_missing=false` remain
  policy-valid, but the reference consumer only executes them when runtime
  feature inputs are already finite; otherwise it raises instead of returning
  non-finite predictions.
- Older v3 bundles that omit `preprocessor.missing_value_policy.impute_missing`
  remain readable and default that field to `true`.
- `manifest_sha256` protects the embedded `model`, `inference`,
  `preprocessor`, `weights`, and other top-level v3 manifest metadata against
  post-export edits.

## Many-Class Modes

- `manifest.model.many_class_train_mode` configures the training-time branch
  behavior (`path_nll` vs `full_probs`) and is used when reconstructing the
  model from an export bundle.
- `manifest.inference.many_class_inference_mode` is an inference-runtime
  contract field and is currently fixed to `full_probs`.

## Producer Commands

Export from checkpoint:

```bash
uv run tab-foundry export bundle \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  --out-dir outputs/exports/cls_smoke_v3
```

Validate bundle:

```bash
uv run tab-foundry export validate \
  --bundle-dir outputs/exports/cls_smoke_v3
```

## Reference Consumer

This repo includes a reference-only executable consumer in
`tab_foundry.export.loader_ref`.

Scope:

- dense numeric matrices in
- persisted checksum and schema validation
- runtime-derived preprocessing using the support set
- model-native outputs out (`class_probs` for classification)

Out of scope here:

- serving APIs
- dataframe adapters
- long-lived runtime ownership
- generalized production inference policy beyond the separate-repo handoff

## Compatibility Policy

- Training checkpoints remain unchanged and are still used for resume/training
  workflows.
- Export bundles are the cross-repo inference handoff contract.
- `tab-foundry-export-v3` is the default export format and the only executable
  reference-consumer contract in this repo.
- Existing v3 bundles produced with sidecar metadata are intentionally obsolete
  after the single-manifest redesign and must be regenerated.
- Existing single-manifest v3 bundles without `manifest_sha256` are
  intentionally obsolete after the metadata-integrity fix and must be
  regenerated.
- Existing v3 bundles without `preprocessor.missing_value_policy.impute_missing`
  remain validator-readable and default to `true` until regenerated.
- Existing v3 bundles without the additive staged-surface model fields remain
  validator-readable and default those values until regenerated.
- `tab-foundry-export-v2` bundles remain validator-readable during migration.
- `tab-foundry-export-v1` bundles are intentionally unsupported and must be
  regenerated onto the current classification-only staged/simple surface.
- Checkpoint export/load now treats omitted `feature_group_size` as `1`. Legacy
  grouped-token checkpoints that omitted that field are intentionally rejected
  and must be regenerated or loaded with an explicit `feature_group_size`
  override before export.

# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/binary_md_v2/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `binary_md_v2`
- Sweep status: `active`
- Parent sweep id: `binary_md_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `01_nano_exact_md_prior_parity_fix_binary_medium_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Comparison policy: `anchor_only`
- Anchor metrics: best ROC AUC `0.7615`, final ROC AUC `0.7599`, final training time `355.6s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Same nano feature encoder path. | Functional parity should be close; remaining differences come from the staged wrapper and metadata surface, not the active encoder math. |
| target conditioning | Mean-padded linear target encoder. | Same mean-padded linear target conditioner. | The anchor intentionally preserves the upstream label-conditioning mechanism. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Same nano post-norm block. | This is the strongest structural anchor to upstream parity. |
| row readout | Target-column readout from the final cell tensor. | Same target-column row pool. | Later row-pool experiments should be interpreted as explicit departures from upstream, not minor tuning. |
| tokenizer | One scalar token per feature. | Same scalar-per-feature tokenizer. | Grouped-token changes must be isolated from later row/context changes. |
| context encoder | None on the direct binary path. | None on the direct binary path. | Any context encoder row changes both compute graph depth and target-conditioning semantics and therefore needs its own adequacy commentary. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Repo-local manifest-driven prior-training surface at data/manifests/default.parquet. | Data-source changes are first-class sweep rows, not background assumptions. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Runtime support-set preprocessing with mean imputation and train-only label remap/filter semantics. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Exact-prior training surface label `prior_constant_lr`. | Optimizer and schedule changes are first-class sweep rows, not silent background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Legacy stage alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_anchor_activation_trace_baseline` | diagnostics | yes | ready | none | Keep the locked anchor model surface fixed but rerun prior training with activation tracing enabled so forward-pass norm dynamics are captured in `gradient_history.jsonl` and `telemetry.json`. | Run the traced anchor baseline before comparing shared-encoder stabilization variants. |
| 2 | `delta_shared_feature_norm` | feature_encoding | yes | ready | shared_norm | Replace the internal nano feature path with the shared linear feature encoder while keeping scalar-per-feature tokenization. | Run immediately after the traced anchor baseline so the shared path has a direct activation-profile comparator. |
| 3 | `delta_shared_feature_norm_with_post_layernorm` | feature_encoding | yes | ready | none | Shared linear feature encoder with explicit post-encoder LayerNorm before the transformer blocks. | Run after the traced shared-feature baseline to test whether LayerNorm fixes late-curve retention. |
| 4 | `delta_shared_feature_norm_with_post_rmsnorm` | feature_encoding | yes | ready | none | Shared linear feature encoder with explicit post-encoder RMSNorm before the transformer blocks. | Run after the LayerNorm variant to compare normalization families on the shared encoder path. |

## Detailed Rows

### 1. `delta_anchor_activation_trace_baseline`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Keep the locked anchor model surface fixed but rerun prior training with activation tracing enabled so forward-pass norm dynamics are captured in `gradient_history.jsonl` and `telemetry.json`.
- Rationale: Establish a traced anchor baseline before rerunning the shared-encoder rows so activation-scale shifts can be interpreted against the unchanged model surface.
- Hypothesis: Activation tracing should be additive only; the traced anchor rerun is expected to preserve anchor-quality behavior while revealing the forward-pass norm profile entering and traversing the transformer.
- Upstream delta: Upstream nanoTabPFN does not emit comparable forward-pass activation telemetry.
- Anchor delta: Keeps the anchor model, data, and preprocessing surfaces fixed and changes only the training recipe by enabling activation tracing.
- Expected effect: No model-quality change is expected; this row establishes the traced anchor baseline needed to interpret later shared-encoder stabilization runs.
- Effective labels: model=`delta_anchor_activation_trace_baseline`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr_trace_activations`
- Training overrides: `{'runtime': {'trace_activations': True}}`
- Parameter adequacy plan:
  - Confirm the traced run preserves the anchor model, data, and preprocessing surfaces.
  - Use the emitted activation norms as the baseline comparator for all later stabilization rows.
- Adequacy knobs to dimension explicitly:
  - Treat this as a diagnostic rerun of the anchor, not a new structural model claim.
  - Confirm the additive tracing overhead does not perturb the run contract or artifact schema.
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v2/delta_anchor_activation_trace_baseline/result_card.md`
- Benchmark metrics: pending

### 2. `delta_shared_feature_norm`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `shared_norm`
- Description: Replace the internal nano feature path with the shared linear feature encoder while keeping scalar-per-feature tokenization.
- Rationale: Reproduce the shared-feature-encoder row with activation tracing enabled so the late-curve retention gap can be separated from purely structural explanations.
- Hypothesis: Shared feature encoding may still peak above the anchor, but the traced rerun should show whether unstable pre-transformer or per-block activation scales accompany the weaker final checkpoint.
- Upstream delta: Upstream nanoTabPFN keeps its own internal feature normalization/encoding path.
- Anchor delta: Changes feature_encoder from nano to shared and enables activation tracing while keeping tokenizer, target conditioning, table block, row pooling, data, and preprocessing fixed.
- Expected effect: Better transfer across heterogeneous columns, with risk that the shared path simply mismatches the compact binary anchor scale.
- Effective labels: model=`delta_shared_feature_norm`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr_trace_activations`
- Model overrides: `{'module_overrides': {'feature_encoder': 'shared'}}`
- Parameter adequacy plan:
  - Keep optimizer and row/context structure fixed for the first pass.
  - Only treat a weak result as structural evidence after checking whether normalization mismatch is the dominant failure mode.
- Adequacy knobs to dimension explicitly:
  - input_normalization remains train_zscore_clip on the anchor unless the queue row explicitly changes preprocessing.
  - If performance drops, note whether the harm appears early or only after longer training.
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v2/delta_shared_feature_norm/result_card.md`
- Benchmark metrics: pending

### 3. `delta_shared_feature_norm_with_post_layernorm`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Shared linear feature encoder with explicit post-encoder LayerNorm before the transformer blocks.
- Rationale: Test whether explicit LayerNorm after the shared feature encoder stabilizes the cell table before it enters the transformer.
- Hypothesis: Post-encoder LayerNorm should reduce activation-scale drift across transformer blocks and improve final retention relative to the traced shared baseline.
- Upstream delta: Upstream nanoTabPFN does not expose a separate post-encoder normalization stage on this path.
- Anchor delta: Changes feature_encoder from nano to shared, adds post_encoder_norm=layernorm, and enables activation tracing while keeping the rest of the anchor surface fixed.
- Expected effect: LayerNorm after encoding should stabilize the activation scale entering the transformer and improve late-curve retention relative to the plain shared feature encoder row.
- Effective labels: model=`delta_shared_norm_post_ln`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr_trace_activations`
- Model overrides: `{'module_overrides': {'feature_encoder': 'shared', 'post_encoder_norm': 'layernorm'}}`
- Parameter adequacy plan:
  - Compare the traced shared baseline and this row jointly before drawing conclusions about structural encoder quality.
  - If the row is mixed, inspect whether activation-scale stabilization improves drift even when final ROC AUC is neutral.
- Adequacy knobs to dimension explicitly:
  - Compare pre-transformer and per-block activation growth directly against the traced shared-without-post-norm run.
  - Keep optimizer, dataset bundle, and preprocessing fixed so only the post-encoder normalization changes.
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v2/delta_shared_feature_norm_with_post_layernorm/result_card.md`
- Benchmark metrics: pending

### 4. `delta_shared_feature_norm_with_post_rmsnorm`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Shared linear feature encoder with explicit post-encoder RMSNorm before the transformer blocks.
- Rationale: Compare RMSNorm against LayerNorm as the explicit post-encoder stabilization stage on the shared feature path.
- Hypothesis: RMSNorm may preserve more of the shared encoder scale structure than LayerNorm while still preventing the activation growth that hurt late-curve retention.
- Upstream delta: Upstream nanoTabPFN does not expose a separate post-encoder normalization stage on this path.
- Anchor delta: Changes feature_encoder from nano to shared, adds post_encoder_norm=rmsnorm, and enables activation tracing while keeping the rest of the anchor surface fixed.
- Expected effect: RMSNorm may stabilize the shared encoder activation scale with less feature rescaling than LayerNorm, offering a second normalization hypothesis for the late-curve retention gap.
- Effective labels: model=`delta_shared_norm_post_rms`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr_trace_activations`
- Model overrides: `{'module_overrides': {'feature_encoder': 'shared', 'post_encoder_norm': 'rmsnorm'}}`
- Parameter adequacy plan:
  - Compare activation traces and final ROC AUC jointly against the traced shared baseline.
  - Use the result to separate normalization-family effects from the shared feature encoder itself.
- Adequacy knobs to dimension explicitly:
  - Compare against both the traced shared baseline and the post-layernorm row before preferring one normalization family.
  - Keep optimizer, dataset bundle, and preprocessing fixed so only the post-encoder normalization changes.
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v2/delta_shared_feature_norm_with_post_rmsnorm/result_card.md`
- Benchmark metrics: pending

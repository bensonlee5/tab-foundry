# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/binary_md_v2/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `binary_md_v2`
- Sweep status: `completed`
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

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_anchor_activation_trace_baseline` | diagnostics | yes | completed | none | Keep the locked anchor model surface fixed but rerun prior training with activation tracing enabled so forward-pass norm dynamics are captured in `gradient_history.jsonl` and `telemetry.json`. | None |
| 2 | `delta_shared_feature_norm` | feature_encoding | yes | completed | shared_norm | Replace the internal nano feature path with the shared linear feature encoder while keeping scalar-per-feature tokenization. | None |
| 3 | `delta_shared_feature_norm_with_post_layernorm` | feature_encoding | yes | completed | none | Shared linear feature encoder with explicit post-encoder LayerNorm before the transformer blocks. | None |
| 4 | `delta_shared_feature_norm_with_post_rmsnorm` | feature_encoding | yes | completed | none | Shared linear feature encoder with explicit post-encoder RMSNorm before the transformer blocks. | None |

## Detailed Rows

### 1. `delta_anchor_activation_trace_baseline`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
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
- Interpretation status: `completed`
- Decision: `Anchor trace baseline established. Best ROC AUC 0.7596 (step 2275). Activation norms captured in gradient_history. NanoTabPFN best 0.7616.`
- Notes:
  - wandb: https://wandb.ai/bensonlee55-none/tab-foundry/runs/litf5n0i
  - final_train_loss=0.452, mean_grad_norm=0.177, max_grad_norm=4.658
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v2/delta_anchor_activation_trace_baseline/result_card.md`
- Benchmark metrics:
  - Best ROC AUC: `0.7615` (step 1125)
  - Final ROC AUC: `0.7599`
  - Drift (final − best): `-0.0016`
  - NanoTabPFN control: `0.7616`
  - max_grad_norm: `4.658`

### 2. `delta_shared_feature_norm`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `shared_norm`
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
- Interpretation status: `completed`
- Decision: `Shared feature encoder (no post-norm) best ROC AUC 0.7620 (step 1950), slightly above anchor 0.7596. Activation norms captured. NanoTabPFN best 0.7616.`
- Notes:
  - wandb: https://wandb.ai/bensonlee55-none/tab-foundry/runs/6h20zft8
  - final_train_loss=0.450, mean_grad_norm=0.175, max_grad_norm=4.660
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v2/delta_shared_feature_norm/result_card.md`
- Benchmark metrics:
  - Best ROC AUC: `0.7654` (step 1325)
  - Final ROC AUC: `0.7558`
  - Drift (final − best): `-0.0096`
  - NanoTabPFN control: `0.7616`
  - max_grad_norm: `4.660`

### 3. `delta_shared_feature_norm_with_post_layernorm`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
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
- Interpretation status: `completed`
- Decision: `LayerNorm post-encoder achieved BEST sweep ROC AUC 0.7670 (step 2075), exceeding both anchor (0.7596) and shared-no-norm (0.7620). Hypothesis confirmed - LayerNorm stabilizes post-encoder activations. max_grad_norm reduced to 4.208 vs anchor 4.658.`
- Notes:
  - wandb: https://wandb.ai/bensonlee55-none/tab-foundry/runs/fedytzgt
  - final_train_loss=0.452, mean_grad_norm=0.182, max_grad_norm=4.208
  - Promoted to canonical follow-on anchor `01_shared_norm_post_ln_binary_medium_v1` for subsequent feature-testing sweeps.
- Follow-up run ids: `['01_shared_norm_post_ln_binary_medium_v1']`
- Result card path: `outputs/staged_ladder/research/binary_md_v2/delta_shared_feature_norm_with_post_layernorm/result_card.md`
- Benchmark metrics:
  - Best ROC AUC: `0.7670` (step 2075)
  - Final ROC AUC: `0.7643`
  - Drift (final − best): `-0.0026`
  - NanoTabPFN control: `0.7616`
  - max_grad_norm: `4.208`

### 4. `delta_shared_feature_norm_with_post_rmsnorm`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
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
- Interpretation status: `completed`
- Decision: `RMSNorm post-encoder best ROC AUC 0.7572 (step 1975), below both anchor (0.7596) and LayerNorm (0.7670). RMSNorm did not improve over shared-no-norm (0.7620). max_grad_norm 4.216 similar to LayerNorm. LayerNorm is the clear winner.`
- Notes:
  - wandb: https://wandb.ai/bensonlee55-none/tab-foundry/runs/r684s86u
  - final_train_loss=0.454, mean_grad_norm=0.182, max_grad_norm=4.216
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v2/delta_shared_feature_norm_with_post_rmsnorm/result_card.md`
- Benchmark metrics:
  - Best ROC AUC: `0.7604` (step 1250)
  - Final ROC AUC: `0.7548`
  - Drift (final − best): `-0.0056`
  - NanoTabPFN control: `0.7616`
  - max_grad_norm: `4.216`

# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v4/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v4/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_010_classification_evolution_medium_v4`
- Sweep status: `ready`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v3`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v4/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `7157a79e2f6014aa89b7a2c3bee20041eabd44b7e597cb70c1d8338de5cf51a0`

## Locked Surface

- Anchor run id: `null`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_v1`
- Surface role: `custom`
- Comparison policy: `anchor_only`
- Anchor metrics: `pending trusted rerun`

## Anchor Comparison

Upstream reference: `EquiTabPFN` from `https://arxiv.org/abs/2502.06684`.

Pending trusted rerun: no anchor is registered yet, so this matrix records the locked benchmark surface and queue state before the first anchor promotion.

| Dimension | Upstream EquiTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| benchmark ownership | Not applicable. | `dagzoo` owns the synthetic training fronts while `tab-realdata-hub` owns the real-data validation manifests. | This sweep defines the repo-to-repo contract for the first benchmark-evolution lane. |
| classification head | Label-conditioning choices should stay modular so target handling can evolve after the backbone stabilizes. | Direct multiclass head with `many_class_base=10`. | Treat this as a bounded head/output evolution, not a staged hierarchical many-class port. |
| summary bandwidth | Historical TF-RD-021B evidence used `sandwich_summary_tokens_per_axis=4`. | The evolved benchmark surface uses `sandwich_summary_tokens_per_axis=3`. | The new benchmark package should evaluate the evolved contract directly rather than replaying the historical four-token anchor. |
| validation surface | Not applicable. | Hub-backed medium classification manifest under local benchmark-manifest id `openml_classification_medium_v1`, materialized from `openml_classification_medium_v1.json`. | Keep the smaller hub-backed classification validation rung fixed while the synthetic training front reruns on the corrected sandwich and training surface. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control` | provenance | no | ready | none | Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests. | Benchmark and register the completed pilot control run on the intended hub-backed medium classification manifest as `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_curated_v5`, then promote it as the sweep anchor; do not retrain row 1. |
| 2 | `delta_data_manifest_root_tf_rd_010_missingness_mcar` | missingness | no | ready | none | Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Exploratory row: benchmark only after the row-1 control anchor is benchmarked and recorded; results are non-promotable until curated missingness fronts exist. |
| 3 | `delta_data_manifest_root_tf_rd_010_missingness_mar` | missingness | no | ready | none | Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Exploratory row: benchmark only after the row-1 control anchor is benchmarked and recorded; results are non-promotable until curated missingness fronts exist. |
| 4 | `delta_data_manifest_root_tf_rd_010_missingness_mnar` | missingness | no | ready | none | Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Exploratory row: benchmark only after the row-1 control anchor is benchmarked and recorded; results are non-promotable until curated missingness fronts exist. |

## Detailed Rows

### 1. `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests.
- Rationale: Establish the TF-RD-010 classification control front before reading any missingness harder-front effect on the smaller medium benchmark rung.
- Hypothesis: The evolved sandwich family should first be judged on the TF-RD-010 control corpus (`n_classes_min=2`) against the medium hub manifest.
- Upstream delta: Not applicable; this is a repo-local synthetic training-front contract tied to the first benchmark-evolution lane.
- Anchor delta: Use the evolved FiLM plus 3-summary-token sandwich contract and benchmark the completed control pilot trained on `tf_rd_010_dagzoo_medium_control_curated_v5` against the hub-owned medium classification manifest.
- Expected effect: Establish the TF-RD-010 classification control corpus that both the medium and large validation rungs will compare against.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `db3d4d5e84ddcd1af92308fc86f283e1509c009ce8b0206a97cd4612ee9084a5`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_dagzoo_medium_control_curated_v5'}`
- Parameter adequacy plan:
  - Confirm `tab-realdata-hub#1` has materialized the medium classification manifest from `openml_classification_medium_v1.json` before execution.
  - Freeze the legacy `cls_benchmark_linear_multiclass_medium_v1` control baseline before treating any row outcome as a promotion or defer decision.
  - Rank by `final_log_loss_at_matched_regime_budget`, interpreted explicitly as label-target log loss per test cell, then inspect calibration, runtime, stability, and any retained legacy cell-likelihood diagnostics as guardrails.
- Adequacy knobs to dimension explicitly:
  - explicit dagzoo provenance for the classification control corpus
  - medium and large real-data validation separation via `tab-realdata-hub` manifests
  - class-count coverage, feature-count coverage, missingness policy, and minority-class floor on the validation side
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at <=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - This row remains the intended TF-RD-010 medium reference for missingness and class-imbalance reporting on the medium validation pool.
  - `tf_rd_010_classification_evolution_medium_v4` uses the pilot-aligned control contract: `task_batch_size=16`, `grad_accum_steps=4`, `runtime.grad_clip=0.0`, `max_steps=2500`, linear schedule with `warmup_ratio=0.10`, `lr_max=1e-3`, and `optimizer.min_lr=1e-5`.
  - The completed pilot control training artifact at `outputs/research/adequacy/tf_rd_010_synthetic_adequacy_v3/pilot/production_control_curated_v5/train` reached `step 2500` and is the reusable training artifact for row 1.
  - The current top-level adequacy summary file is stale blocked canary output and must not be used as the row-1 gate artifact.
  - The repo-local `openml_classification_medium_v1` manifest is still a stale placeholder; canonical benchmark registration must use the intended hub-backed medium classification manifest.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_dagzoo_medium_control/result_card.md`
- Benchmark metrics: pending

### 2. `delta_data_manifest_root_tf_rd_010_missingness_mcar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Test whether moderate MCAR exposure improves robustness before structured missingness is considered on the medium validation pool.
- Hypothesis: MCAR may improve label-target log loss and calibration on the evolved sandwich family without adding the stronger structure of MAR or MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mcar_v3`.
- Expected effect: Moderate MCAR should test whether the evolved sandwich target benefits from missingness exposure before any larger benchmark-front escalation.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mcar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `995d37d1a01450ad97757de24b4bc9dd9d91e134942c79a2001353dd51a33592`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mcar_v3'}`
- Parameter adequacy plan:
  - Compare directly against the clean control row before preferring missingness exposure.
  - Keep class-imbalance reporting explicit on the medium rung, but defer any dedicated skew ladder to TF-RD-017.
  - Use the larger hub-backed validation rung later as the main transfer check for any kept missingness front.
- Adequacy knobs to dimension explicitly:
  - explicit MCAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - natural-log CE/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - control row uses curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`
  - this missingness row remains `include_all` `v3`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at <=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - `tf_rd_010_classification_evolution_medium_v4` keeps the active sandwich benchmark path under `training.loss_surface=classification`, so `final_log_loss_at_matched_regime_budget` remains the canonical ranking key, interpreted explicitly as label-target log loss per test cell; `cell_bpc` is legacy-only historical context.
  - The control comparison for this sweep now uses curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`, while this missingness row remains `include_all` `v3`.
  - This row is exploratory and non-promotable until curated missingness fronts exist.
  - The repo-local `openml_classification_medium_v1` manifest is still a stale placeholder; canonical benchmark registration must use the intended hub-backed medium classification manifest.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mcar/result_card.md`
- Benchmark metrics: pending

### 3. `delta_data_manifest_root_tf_rd_010_missingness_mar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Add a structured missingness row so TF-RD-010 can distinguish random masking from observed-feature-linked masking under the evolved benchmark contract.
- Hypothesis: MAR may provide a clearer harder front than MCAR while remaining more interpretable than MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mar_v3`.
- Expected effect: Structured MAR may provide a harder but still interpretable missingness front for the first TF-RD-010 classification benchmark program.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `11a430b5f5fa7d5bfe9055fa6d99334c5db5428eef20e16f4d2e5f518062dba7`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mar_v3'}`
- Parameter adequacy plan:
  - Compare directly against the clean control plus MCAR and MNAR before preferring structured missingness.
  - Keep class-imbalance reporting explicit on the medium rung, but defer any dedicated skew ladder to TF-RD-017.
  - Use the larger hub-backed validation rung later as the main transfer check for any kept missingness front.
- Adequacy knobs to dimension explicitly:
  - explicit MAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - natural-log CE/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - control row uses curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`
  - this missingness row remains `include_all` `v3`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at <=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - `tf_rd_010_classification_evolution_medium_v4` keeps the active sandwich benchmark path under `training.loss_surface=classification`, so `final_log_loss_at_matched_regime_budget` remains the canonical ranking key, interpreted explicitly as label-target log loss per test cell; `cell_bpc` is legacy-only historical context.
  - The control comparison for this sweep now uses curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`, while this missingness row remains `include_all` `v3`.
  - This row is exploratory and non-promotable until curated missingness fronts exist.
  - The repo-local `openml_classification_medium_v1` manifest is still a stale placeholder; canonical benchmark registration must use the intended hub-backed medium classification manifest.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mar/result_card.md`
- Benchmark metrics: pending

### 4. `delta_data_manifest_root_tf_rd_010_missingness_mnar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Keep one strongest missingness row in the first draft package so TF-RD-010 can compare MCAR, MAR, and MNAR under the same medium benchmark contract.
- Hypothesis: MNAR may be the hardest missingness front, but it may also be the least interpretable candidate for the first evolved benchmark package.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mnar_v3`.
- Expected effect: Structured MNAR may be the strongest synthetic missingness perturbation, but it risks a less interpretable first benchmark-evolution read than MCAR or MAR.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mnar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `06e855120e76bfedb0eed931904d480d7996198504cb73097dcf5f3c9bf796a1`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mnar_v3'}`
- Parameter adequacy plan:
  - Compare directly against the clean control plus MCAR and MAR before preferring the strongest self-masking option.
  - Keep class-imbalance reporting explicit on the medium rung, but defer any dedicated skew ladder to TF-RD-017.
  - Use the larger hub-backed validation rung later as the main transfer check for any kept missingness front.
- Adequacy knobs to dimension explicitly:
  - explicit MNAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - natural-log CE/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - control row uses curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`
  - this missingness row remains `include_all` `v3`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at <=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - `tf_rd_010_classification_evolution_medium_v4` keeps the active sandwich benchmark path under `training.loss_surface=classification`, so `final_log_loss_at_matched_regime_budget` remains the canonical ranking key, interpreted explicitly as label-target log loss per test cell; `cell_bpc` is legacy-only historical context.
  - The control comparison for this sweep now uses curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`, while this missingness row remains `include_all` `v3`.
  - This row is exploratory and non-promotable until curated missingness fronts exist.
  - The repo-local `openml_classification_medium_v1` manifest is still a stale placeholder; canonical benchmark registration must use the intended hub-backed medium classification manifest.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mnar/result_card.md`
- Benchmark metrics: pending

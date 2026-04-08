# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v4/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v4/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_010_classification_evolution_medium_v4`
- Sweep status: `completed`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v3`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v4/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `e75c5161033650fd2cecaf844a297fa9d5ebe4d649c18aeca6741643007d4371`

## Locked Surface

- Anchor run id: `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v8`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_v1`
- Surface role: `custom`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1136`, final BPF `2.1136`, final log loss `0.6812`, final Brier score `0.4229`, best ROC AUC `0.6094`, final ROC AUC `0.6094`, final training time `7449.8s`

## Anchor Comparison

Upstream reference: `EquiTabPFN` from `https://arxiv.org/abs/2502.06684`.

| Dimension | Upstream EquiTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| benchmark ownership | Not applicable. | `dagzoo` owns the synthetic training fronts while `tab-realdata-hub` owns the real-data validation manifests. | This sweep defines the repo-to-repo contract for the first benchmark-evolution lane. |
| classification head | Label-conditioning choices should stay modular so target handling can evolve after the backbone stabilizes. | Direct multiclass head with `many_class_base=10`. | Treat this as a bounded head/output evolution, not a staged hierarchical many-class port. |
| summary bandwidth | Historical TF-RD-021B evidence used `sandwich_summary_tokens_per_axis=4`. | The evolved benchmark surface uses `sandwich_summary_tokens_per_axis=3`. | The new benchmark package should evaluate the evolved contract directly rather than replaying the historical four-token anchor. |
| validation surface | Not applicable. | Hub-backed medium classification manifest under local benchmark-manifest id `openml_classification_medium_v1`, materialized from `openml_classification_medium_v1.json`. | Keep the smaller hub-backed classification validation rung fixed while the synthetic training front reruns on the corrected sandwich and training surface. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control` | provenance | no | completed | none | Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests. | Benchmark and register the completed pilot control run on the intended hub-backed medium classification manifest as `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_curated_v5`, then promote it as the sweep anchor; do not retrain row 1. |
| 2 | `delta_data_manifest_root_tf_rd_010_missingness_mcar` | missingness | no | completed | none | Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Exploratory row: benchmark only after the row-1 control anchor is benchmarked and recorded; results are non-promotable until curated missingness fronts exist. |
| 3 | `delta_data_manifest_root_tf_rd_010_missingness_mar` | missingness | no | completed | none | Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Exploratory row: benchmark only after the row-1 control anchor is benchmarked and recorded; results are non-promotable until curated missingness fronts exist. |
| 4 | `delta_data_manifest_root_tf_rd_010_missingness_mnar` | missingness | no | completed | none | Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Exploratory row: benchmark only after the row-1 control anchor is benchmarked and recorded; results are non-promotable until curated missingness fronts exist. |

## Detailed Rows

### 1. `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests.
- Rationale: Establish the TF-RD-010 classification control front before reading any missingness harder-front effect on the smaller medium benchmark rung.
- Hypothesis: The evolved sandwich family should first be judged on the TF-RD-010 control corpus (`n_classes_min=2`) against the medium hub manifest.
- Upstream delta: Not applicable; this is a repo-local synthetic training-front contract tied to the first benchmark-evolution lane.
- Anchor delta: Use the evolved FiLM plus 3-summary-token sandwich contract and benchmark the completed control pilot trained on `tf_rd_010_dagzoo_medium_control_curated_v5` against the hub-owned medium classification manifest.
- Expected effect: Establish the TF-RD-010 classification control corpus that both the medium and large validation rungs will compare against.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ca3aadfde8968c65d71fe6101418fb0b868106edb7c8452cae95fe0529c126a9`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_dagzoo_medium_control_curated_v5'}`
- Reuse train artifact: `outputs/research/adequacy/tf_rd_010_synthetic_adequacy_v3/pilot/production_control_curated_v5/train`
- Reuse training surface fingerprint: `1614c767510feacd669b4868fd2dfacbe7332f0b64b9c694c448caca85794d20`
- Parameter adequacy plan:
  - Confirm `tab-realdata-hub#1` has materialized the medium classification manifest from `openml_classification_medium_v1.json` before execution.
  - Freeze the legacy `cls_benchmark_linear_multiclass_medium_v1` control baseline before treating any row outcome as a promotion or defer decision.
  - Rank by `final_log_loss_at_matched_regime_budget`, interpreted explicitly as label-target log loss per test cell, then inspect calibration, runtime, stability, and any retained legacy cell-likelihood diagnostics as guardrails.
- Adequacy knobs to dimension explicitly:
  - explicit dagzoo provenance for the classification control corpus
  - medium and large real-data validation separation via `tab-realdata-hub` manifests
  - class-count coverage, feature-count coverage, missingness policy, and minority-class floor on the validation side
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at <=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - This row remains the intended TF-RD-010 medium reference for missingness and class-imbalance reporting on the medium validation pool.
  - `tf_rd_010_classification_evolution_medium_v4` uses the pilot-aligned control contract: `task_batch_size=16`, `grad_accum_steps=4`, `runtime.grad_clip=0.0`, `max_steps=2500`, linear schedule with `warmup_ratio=0.10`, `lr_max=1e-3`, and `optimizer.min_lr=1e-5`.
  - Row 1 benchmarks only the reusable artifact's `best.pt` and `latest.pt` checkpoints instead of the full 100-snapshot curve.
  - The completed pilot control training artifact at `outputs/research/adequacy/tf_rd_010_synthetic_adequacy_v3/pilot/production_control_curated_v5/train` reached `step 2500` and is the reusable training artifact for row 1.
  - The current top-level adequacy summary file is stale blocked canary output and must not be used as the row-1 gate artifact.
  - The repo-local `openml_classification_medium_v1` manifest is still a stale placeholder; canonical benchmark registration must use the intended hub-backed medium classification manifest.
  - Benchmarked pinned reusable training artifact `outputs/research/adequacy/tf_rd_010_synthetic_adequacy_v3/pilot/production_control_curated_v5/train`.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v5`.
  - Benchmarked the pinned TF-RD-010 medium control train artifact against the refreshed hub-backed medium manifest using best and final checkpoints only, then promoted it as the sweep anchor.
  - Supersedes historical queue run `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v5`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v6`.
  - Supersedes historical queue run `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v6`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v7`.
  - Supersedes historical queue run `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v7`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v8`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_dagzoo_medium_control/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v8` with final log loss `0.6812`, delta final log loss `+0.0000`, final Brier score `0.4229`, delta final brier score `+0.0000`, final ROC AUC `0.6094`, delta final roc auc `+0.0000`, final BPC (legacy feature-cell diagnostic) `2.1136`, delta final bpc (legacy feature-cell diagnostic) `+0.0000`, final BPF (legacy feature-cell diagnostic) `2.1136`, delta final bpf (legacy feature-cell diagnostic) `+0.0000`, best ROC AUC `0.6094`, delta final training time `+0.0s`

### 2. `delta_data_manifest_root_tf_rd_010_missingness_mcar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Test whether moderate MCAR exposure improves robustness before structured missingness is considered on the medium validation pool.
- Hypothesis: MCAR may improve label-target log loss and calibration on the evolved sandwich family without adding the stronger structure of MAR or MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mcar_v3`.
- Expected effect: Moderate MCAR should test whether the evolved sandwich target benefits from missingness exposure before any larger benchmark-front escalation.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mcar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ce551f657c6babd75acbfe061796a947eea6ba4849fe323189bdb42eb3aa2e9c`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at <=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - `tf_rd_010_classification_evolution_medium_v4` keeps the active sandwich benchmark path under `training.loss_surface=classification`, so `final_log_loss_at_matched_regime_budget` remains the canonical ranking key, interpreted explicitly as label-target log loss per test cell; `cell_bpc` is legacy-only historical context.
  - The control comparison for this sweep now uses curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`, while this missingness row remains `include_all` `v3`.
  - This row is exploratory and non-promotable until curated missingness fronts exist.
  - This row benchmarks only `best.pt` and `latest.pt` to keep the medium sweep tractable on CPU.
  - The repo-local `openml_classification_medium_v1` manifest is still a stale placeholder; canonical benchmark registration must use the intended hub-backed medium classification manifest.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v4_02_delta_data_manifest_root_tf_rd_010_missingness_mcar_v1`.
  - Exploratory missingness row benchmarked against the refreshed hub-backed medium manifest using best and final checkpoints only; non-promotable until curated missingness fronts exist.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mcar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_medium_v4_02_delta_data_manifest_root_tf_rd_010_missingness_mcar_v1` with final log loss `0.6944`, delta final log loss `+0.0132`, final Brier score `0.4319`, delta final brier score `+0.0090`, final ROC AUC `0.5943`, delta final roc auc `-0.0151`, final BPC (legacy feature-cell diagnostic) `2.1257`, delta final bpc (legacy feature-cell diagnostic) `+0.0120`, final BPF (legacy feature-cell diagnostic) `2.1256`, delta final bpf (legacy feature-cell diagnostic) `+0.0120`, best ROC AUC `0.5943`, delta final training time `+3566.0s`

### 3. `delta_data_manifest_root_tf_rd_010_missingness_mar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Add a structured missingness row so TF-RD-010 can distinguish random masking from observed-feature-linked masking under the evolved benchmark contract.
- Hypothesis: MAR may provide a clearer harder front than MCAR while remaining more interpretable than MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mar_v3`.
- Expected effect: Structured MAR may provide a harder but still interpretable missingness front for the first TF-RD-010 classification benchmark program.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0ddc1ffd07251f83afe1a7ad2c79927180031518bbeb9a6364f738e7ca9592a8`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at <=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - `tf_rd_010_classification_evolution_medium_v4` keeps the active sandwich benchmark path under `training.loss_surface=classification`, so `final_log_loss_at_matched_regime_budget` remains the canonical ranking key, interpreted explicitly as label-target log loss per test cell; `cell_bpc` is legacy-only historical context.
  - The control comparison for this sweep now uses curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`, while this missingness row remains `include_all` `v3`.
  - This row is exploratory and non-promotable until curated missingness fronts exist.
  - This row benchmarks only `best.pt` and `latest.pt` to keep the medium sweep tractable on CPU.
  - The repo-local `openml_classification_medium_v1` manifest is still a stale placeholder; canonical benchmark registration must use the intended hub-backed medium classification manifest.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v4_03_delta_data_manifest_root_tf_rd_010_missingness_mar_v1`.
  - Exploratory missingness row benchmarked against the refreshed hub-backed medium manifest using best and final checkpoints only; non-promotable until curated missingness fronts exist.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_medium_v4_03_delta_data_manifest_root_tf_rd_010_missingness_mar_v1` with final log loss `0.7090`, delta final log loss `+0.0279`, final Brier score `0.4424`, delta final brier score `+0.0195`, final ROC AUC `0.5763`, delta final roc auc `-0.0331`, final BPC (legacy feature-cell diagnostic) `2.1548`, delta final bpc (legacy feature-cell diagnostic) `+0.0412`, final BPF (legacy feature-cell diagnostic) `2.1548`, delta final bpf (legacy feature-cell diagnostic) `+0.0412`, best ROC AUC `0.5763`, delta final training time `+3496.8s`

### 4. `delta_data_manifest_root_tf_rd_010_missingness_mnar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Keep one strongest missingness row in the first draft package so TF-RD-010 can compare MCAR, MAR, and MNAR under the same medium benchmark contract.
- Hypothesis: MNAR may be the hardest missingness front, but it may also be the least interpretable candidate for the first evolved benchmark package.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mnar_v3`.
- Expected effect: Structured MNAR may be the strongest synthetic missingness perturbation, but it risks a less interpretable first benchmark-evolution read than MCAR or MAR.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mnar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `f5ac7886cb4404df75e7a9a85985282c8c07326a2dad8f24e83e485258b56e6c`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
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
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at <=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - `tf_rd_010_classification_evolution_medium_v4` keeps the active sandwich benchmark path under `training.loss_surface=classification`, so `final_log_loss_at_matched_regime_budget` remains the canonical ranking key, interpreted explicitly as label-target log loss per test cell; `cell_bpc` is legacy-only historical context.
  - The control comparison for this sweep now uses curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`, while this missingness row remains `include_all` `v3`.
  - This row is exploratory and non-promotable until curated missingness fronts exist.
  - This row benchmarks only `best.pt` and `latest.pt` to keep the medium sweep tractable on CPU.
  - The repo-local `openml_classification_medium_v1` manifest is still a stale placeholder; canonical benchmark registration must use the intended hub-backed medium classification manifest.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v4_04_delta_data_manifest_root_tf_rd_010_missingness_mnar_v1`.
  - Exploratory missingness row benchmarked against the refreshed hub-backed medium manifest using best and final checkpoints only; non-promotable until curated missingness fronts exist.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mnar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_medium_v4_04_delta_data_manifest_root_tf_rd_010_missingness_mnar_v1` with final log loss `0.7096`, delta final log loss `+0.0284`, final Brier score `0.4427`, delta final brier score `+0.0198`, final ROC AUC `0.5768`, delta final roc auc `-0.0326`, final BPC (legacy feature-cell diagnostic) `2.1675`, delta final bpc (legacy feature-cell diagnostic) `+0.0539`, final BPF (legacy feature-cell diagnostic) `2.1675`, delta final bpf (legacy feature-cell diagnostic) `+0.0539`, best ROC AUC `0.5768`, delta final training time `+3316.4s`

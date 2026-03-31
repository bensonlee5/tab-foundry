# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v4/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v4/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_010_classification_evolution_medium_v4`
- Sweep status: `blocked_on_synthetic_adequacy`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v3`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v4/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `9ee6aeaf6bf3694968b626603d2e9bade8b34bfe4d5360dcec0c86bf4600d157`

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
| 1 | `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control` | provenance | no | blocked_on_synthetic_adequacy | none | Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests. | Block this row pending `tf_rd_010_synthetic_adequacy_v1` and the factorization-correct `v3` corpus family under issue `#205`; do not retune or rerun `medium_v4` until the adequacy interpretation artifact is written. |
| 2 | `delta_data_manifest_root_tf_rd_010_missingness_mcar` | missingness | no | blocked_on_synthetic_adequacy | none | Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Block this row pending `tf_rd_010_synthetic_adequacy_v1` under issue `#205`; do not run the missingness ladder until the adequacy interpretation says the factorization-correct data is learnable. |
| 3 | `delta_data_manifest_root_tf_rd_010_missingness_mar` | missingness | no | blocked_on_synthetic_adequacy | none | Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Block this row pending `tf_rd_010_synthetic_adequacy_v1` under issue `#205`; do not run the missingness ladder until the adequacy interpretation says the factorization-correct data is learnable. |
| 4 | `delta_data_manifest_root_tf_rd_010_missingness_mnar` | missingness | no | blocked_on_synthetic_adequacy | none | Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Block this row pending `tf_rd_010_synthetic_adequacy_v1` under issue `#205`; do not run the missingness ladder until the adequacy interpretation says the factorization-correct data is learnable. |

## Detailed Rows

### 1. `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control`

- Dimension family: `data`
- Status: `blocked_on_synthetic_adequacy`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests.
- Rationale: Establish the TF-RD-010 classification control front before reading any missingness harder-front effect on the smaller medium benchmark rung.
- Hypothesis: The evolved sandwich family should first be judged on the TF-RD-010 control corpus (`n_classes_min=2`) against the medium hub manifest.
- Upstream delta: Not applicable; this is a repo-local synthetic training-front contract tied to the first benchmark-evolution lane.
- Anchor delta: Use the evolved FiLM plus 3-summary-token sandwich contract and train on `tf_rd_010_dagzoo_medium_control_v2` while validating on the hub-owned medium classification manifest.
- Expected effect: Establish the TF-RD-010 classification control corpus that both the medium and large validation rungs will compare against.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `a02e6bcf7dde4adcc2f522a89a8eaaae58136931f26a783d05d10c2067030121`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_dagzoo_medium_control_v2'}`
- Parameter adequacy plan:
  - Confirm `tab-realdata-hub#1` has materialized the medium classification manifest from `openml_classification_medium_v1.json` before execution.
  - Freeze the legacy `cls_benchmark_linear_multiclass_medium_v1` control baseline before treating any row outcome as a promotion or defer decision.
  - Rank by `final_log_loss_at_matched_regime_budget`, interpreted explicitly as label-target log loss per test cell, then inspect calibration, runtime, stability, and any retained legacy cell-likelihood diagnostics as guardrails.
- Adequacy knobs to dimension explicitly:
  - explicit dagzoo provenance for the classification control corpus
  - medium and large real-data validation separation via `tab-realdata-hub` manifests
  - class-count coverage, feature-count coverage, missingness policy, and minority-class floor on the validation side
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at \<=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - This row remains the intended TF-RD-010 medium reference for missingness and class-imbalance reporting on the medium validation pool.
  - Historical 400-step TF-RD-010 executions, the completed 3-step reset-contract rerun, and the completed clipped `tf_rd_010_classification_evolution_medium_v2` rerun remain historical context only.
  - Trusted rerun work now flows through issues `#202`, `#205`, and `#204`.
  - `tf_rd_010_classification_evolution_medium_v4` preserves the blocked medium rerun contract: exact-shape-compatible manifest tasks are batched with `task_batch_size=16`, `grad_accum_steps=4`, `64` tasks per optimizer update, `runtime.grad_clip=0.0`, linear schedule with `warmup_ratio=0.10`, `lr_max=1e-3`, and `optimizer.min_lr=1e-5`.
  - `medium_v4` keeps the active sandwich benchmark path under `training.loss_surface=classification`, so `final_log_loss_at_matched_regime_budget` remains the canonical ranking key, interpreted explicitly as label-target log loss per test cell; `cell_bpc` is legacy-only historical context.
  - `tf_rd_010_classification_evolution_medium_v3` is preserved historical no-clipping evidence only: rows 1-3 reached very early best benchmark steps and then drifted badly, and row 4 was intentionally stopped rather than extended as canonical evidence.
  - The first CE control-row CPU pilot (`...medium_control_v3`) was intentionally stopped at step `1324`; sampled checkpoint benchmarking across steps `400/800/1050/1125/1200/1275` found the current best medium-manifest log loss at `step_001200.pt` (`0.9988591380293615`), but that pilot is historical operational context only because the `dagzoo` factorization changed.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_dagzoo_medium_control/result_card.md`
- Benchmark metrics: pending

### 2. `delta_data_manifest_root_tf_rd_010_missingness_mcar`

- Dimension family: `data`
- Status: `blocked_on_synthetic_adequacy`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Test whether moderate MCAR exposure improves robustness before structured missingness is considered on the medium validation pool.
- Hypothesis: MCAR may improve label-target log loss and calibration on the evolved sandwich family without adding the stronger structure of MAR or MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mcar_v2`.
- Expected effect: Moderate MCAR should test whether the evolved sandwich target benefits from missingness exposure before any larger benchmark-front escalation.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mcar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d779279310449bc473ec3e3de7f0c2c715db8075748bcd8bae253e48532f5e75`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mcar_v2'}`
- Parameter adequacy plan:
  - Compare directly against the clean control row before preferring missingness exposure.
  - Keep class-imbalance reporting explicit on the medium rung, but defer any dedicated skew ladder to TF-RD-017.
  - Use the larger hub-backed validation rung later as the main transfer check for any kept missingness front.
- Adequacy knobs to dimension explicitly:
  - explicit MCAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - natural-log CE/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at \<=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - Historical 400-step TF-RD-010 executions, the completed 3-step reset-contract rerun, and the completed clipped `tf_rd_010_classification_evolution_medium_v2` rerun remain historical context only.
  - Trusted rerun work now flows through issues `#202`, `#205`, and `#204`.
  - `tf_rd_010_classification_evolution_medium_v4` preserves the blocked medium rerun contract: exact-shape-compatible manifest tasks are batched with `task_batch_size=16`, `grad_accum_steps=4`, `64` tasks per optimizer update, `runtime.grad_clip=0.0`, linear schedule with `warmup_ratio=0.10`, `lr_max=1e-3`, and `optimizer.min_lr=1e-5`.
  - `medium_v4` keeps the active sandwich benchmark path under `training.loss_surface=classification`, so `final_log_loss_at_matched_regime_budget` remains the canonical ranking key, interpreted explicitly as label-target log loss per test cell; `cell_bpc` is legacy-only historical context.
  - `tf_rd_010_classification_evolution_medium_v3` is preserved historical no-clipping evidence only: rows 1-3 reached very early best benchmark steps and then drifted badly, and row 4 was intentionally stopped rather than extended as canonical evidence.
  - This row is blocked until `tf_rd_010_synthetic_adequacy_v1` is interpreted on the factorization-correct `v3` corpus family.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mcar/result_card.md`
- Benchmark metrics: pending

### 3. `delta_data_manifest_root_tf_rd_010_missingness_mar`

- Dimension family: `data`
- Status: `blocked_on_synthetic_adequacy`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Add a structured missingness row so TF-RD-010 can distinguish random masking from observed-feature-linked masking under the evolved benchmark contract.
- Hypothesis: MAR may provide a clearer harder front than MCAR while remaining more interpretable than MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mar_v2`.
- Expected effect: Structured MAR may provide a harder but still interpretable missingness front for the first TF-RD-010 classification benchmark program.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `7a8b820cd263269478c6fa2502b65814f35a2a702a3ac0d3d1764224fb5c3bd9`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mar_v2'}`
- Parameter adequacy plan:
  - Compare directly against the clean control plus MCAR and MNAR before preferring structured missingness.
  - Keep class-imbalance reporting explicit on the medium rung, but defer any dedicated skew ladder to TF-RD-017.
  - Use the larger hub-backed validation rung later as the main transfer check for any kept missingness front.
- Adequacy knobs to dimension explicitly:
  - explicit MAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - natural-log CE/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at \<=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - Historical 400-step TF-RD-010 executions, the completed 3-step reset-contract rerun, and the completed clipped `tf_rd_010_classification_evolution_medium_v2` rerun remain historical context only.
  - Trusted rerun work now flows through issues `#202`, `#205`, and `#204`.
  - `tf_rd_010_classification_evolution_medium_v4` preserves the blocked medium rerun contract: exact-shape-compatible manifest tasks are batched with `task_batch_size=16`, `grad_accum_steps=4`, `64` tasks per optimizer update, `runtime.grad_clip=0.0`, linear schedule with `warmup_ratio=0.10`, `lr_max=1e-3`, and `optimizer.min_lr=1e-5`.
  - `medium_v4` keeps the active sandwich benchmark path under `training.loss_surface=classification`, so `final_log_loss_at_matched_regime_budget` remains the canonical ranking key, interpreted explicitly as label-target log loss per test cell; `cell_bpc` is legacy-only historical context.
  - `tf_rd_010_classification_evolution_medium_v3` is preserved historical no-clipping evidence only: rows 1-3 reached very early best benchmark steps and then drifted badly, and row 4 was intentionally stopped rather than extended as canonical evidence.
  - This row is blocked until `tf_rd_010_synthetic_adequacy_v1` is interpreted on the factorization-correct `v3` corpus family.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mar/result_card.md`
- Benchmark metrics: pending

### 4. `delta_data_manifest_root_tf_rd_010_missingness_mnar`

- Dimension family: `data`
- Status: `blocked_on_synthetic_adequacy`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Keep one strongest missingness row in the first draft package so TF-RD-010 can compare MCAR, MAR, and MNAR under the same medium benchmark contract.
- Hypothesis: MNAR may be the hardest missingness front, but it may also be the least interpretable candidate for the first evolved benchmark package.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mnar_v2`.
- Expected effect: Structured MNAR may be the strongest synthetic missingness perturbation, but it risks a less interpretable first benchmark-evolution read than MCAR or MAR.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mnar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e9ba8a06b260362eeb84e93cba50bf8bdd5b6778915720c7e53d7a4c2c7d13cb`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mnar_v2'}`
- Parameter adequacy plan:
  - Compare directly against the clean control plus MCAR and MAR before preferring the strongest self-masking option.
  - Keep class-imbalance reporting explicit on the medium rung, but defer any dedicated skew ladder to TF-RD-017.
  - Use the larger hub-backed validation rung later as the main transfer check for any kept missingness front.
- Adequacy knobs to dimension explicitly:
  - explicit MNAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - natural-log CE/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at \<=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - Historical 400-step TF-RD-010 executions, the completed 3-step reset-contract rerun, and the completed clipped `tf_rd_010_classification_evolution_medium_v2` rerun remain historical context only.
  - Trusted rerun work now flows through issues `#202`, `#205`, and `#204`.
  - `tf_rd_010_classification_evolution_medium_v4` preserves the blocked medium rerun contract: exact-shape-compatible manifest tasks are batched with `task_batch_size=16`, `grad_accum_steps=4`, `64` tasks per optimizer update, `runtime.grad_clip=0.0`, linear schedule with `warmup_ratio=0.10`, `lr_max=1e-3`, and `optimizer.min_lr=1e-5`.
  - `medium_v4` keeps the active sandwich benchmark path under `training.loss_surface=classification`, so `final_log_loss_at_matched_regime_budget` remains the canonical ranking key, interpreted explicitly as label-target log loss per test cell; `cell_bpc` is legacy-only historical context.
  - `tf_rd_010_classification_evolution_medium_v3` is preserved historical no-clipping evidence only: rows 1-3 reached very early best benchmark steps and then drifted badly, and row 4 was intentionally stopped rather than extended as canonical evidence.
  - This row is blocked until `tf_rd_010_synthetic_adequacy_v1` is interpreted on the factorization-correct `v3` corpus family.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mnar/result_card.md`
- Benchmark metrics: pending

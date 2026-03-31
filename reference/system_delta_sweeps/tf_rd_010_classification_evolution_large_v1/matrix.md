# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_large_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_010_classification_evolution_large_v1`
- Sweep status: `superseded`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v1`
- Complexity level: `classification_lg`

## Locked Surface

- Anchor run id: `null`
- Benchmark manifest: local benchmark-manifest id `openml_classification_large_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_large_v1`
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
| benchmark ownership | Not applicable. | `dagzoo` owns the synthetic training fronts while `tab-realdata-hub` owns the real-data validation manifests. | This sweep defines the harder validation rung for the repo-to-repo benchmark contract. |
| classification head | Label-conditioning choices should stay modular so target handling can evolve after the backbone stabilizes. | Direct multiclass head with `many_class_base=10`. | Treat this as a bounded head/output evolution, not a staged hierarchical many-class port. |
| summary bandwidth | Historical TF-RD-021B evidence used `sandwich_summary_tokens_per_axis=4`. | The evolved benchmark surface uses `sandwich_summary_tokens_per_axis=3`. | The new benchmark package should evaluate the evolved contract directly rather than replaying the historical four-token anchor. |
| validation surface | Not applicable. | Hub-backed large classification manifest under local benchmark-manifest id `openml_classification_large_v1`, materialized from `openml_classification_large_v1.json`. | Keep the larger hub-backed classification validation rung fixed while the synthetic training front reruns on the corrected sandwich and training surface. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control` | provenance | no | ready | none | Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests. | Do not execute this preserved reset-contract row for new canonical evidence. Use the matching row in `tf_rd_010_classification_evolution_large_v2` under issue `#203` instead. |
| 2 | `delta_data_manifest_root_tf_rd_010_missingness_mcar` | missingness | no | ready | none | Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Do not execute this preserved reset-contract row for new canonical evidence. Use the matching row in `tf_rd_010_classification_evolution_large_v2` under issue `#203` instead. |
| 3 | `delta_data_manifest_root_tf_rd_010_missingness_mar` | missingness | no | ready | none | Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Do not execute this preserved reset-contract row for new canonical evidence. Use the matching row in `tf_rd_010_classification_evolution_large_v2` under issue `#203` instead. |
| 4 | `delta_data_manifest_root_tf_rd_010_missingness_mnar` | missingness | no | ready | none | Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Do not execute this preserved reset-contract row for new canonical evidence. Use the matching row in `tf_rd_010_classification_evolution_large_v2` under issue `#203` instead. |

## Detailed Rows

### 1. `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests.
- Rationale: Establish the TF-RD-010 classification control front before reading any missingness harder-front effect on the larger benchmark rung.
- Hypothesis: The evolved sandwich family should first be judged on the TF-RD-010 control corpus (`n_classes_min=2`) against the larger hub validation manifest.
- Upstream delta: Not applicable; this is a repo-local synthetic training-front contract tied to the first benchmark-evolution lane.
- Anchor delta: Use the evolved FiLM plus 3-summary-token sandwich contract and train on `tf_rd_010_dagzoo_medium_control_v1` while validating on the hub-owned large classification manifest.
- Expected effect: Establish the TF-RD-010 classification control corpus that both the medium and large validation rungs will compare against.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `6ba9205fc920ba48faab6382df71be653bc5f736cbbb6aec20d32104ed3f8159`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 3}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_dagzoo_medium_control_v1'}`
- Parameter adequacy plan:
  - Confirm `tab-realdata-hub#1` has materialized the large classification manifest from `openml_classification_large_v1.json` before execution.
  - Freeze the legacy `cls_benchmark_linear_multiclass_large_v1` control baseline before treating any row outcome as a promotion or defer decision.
  - Rank by `final_bpc_at_matched_regime_budget`, then inspect raw log loss, calibration, runtime, and stability as guardrails.
- Adequacy knobs to dimension explicitly:
  - explicit dagzoo provenance for the classification control corpus
  - medium and large real-data validation separation via `tab-realdata-hub` manifests
  - class-count coverage, feature-count coverage, missingness policy, and minority-class floor on the validation side
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus is now a balanced 144-task DAGZoo grid with row totals `128/256/512/1024`, feature counts `6/10/14/20`, and explicit class coverage for every integer from `2` through `10`.
  - Trusted reruns use a single synthetic epoch only: one pass over 144 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `3` optimizer steps with an allowed short final batch.
  - This row remains the intended TF-RD-010 large reference for missingness transfer and class-imbalance reporting on the large validation pool.
  - Historical 400-step TF-RD-010 executions are invalidated and retained only as non-canonical artifacts.
  - Trusted rerun work now flows through issues `#202`, `#203`, and `#204`.
  - Superseded for active execution by `tf_rd_010_classification_evolution_large_v2`; keep this row only as the preserved 3-step reset-contract reference.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_large_v1/delta_data_manifest_root_tf_rd_010_dagzoo_medium_control/result_card.md`
- Benchmark metrics: pending

### 2. `delta_data_manifest_root_tf_rd_010_missingness_mcar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Test whether moderate MCAR exposure improves robustness on the larger benchmark rung before structured missingness is considered.
- Hypothesis: MCAR may improve BPC and log-loss behavior on the evolved sandwich family under the larger hub validation pool without adding the stronger structure of MAR or MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mcar_v1`.
- Expected effect: Moderate MCAR should test whether the evolved sandwich target benefits from missingness exposure before any larger benchmark-front escalation.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mcar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ddb562bfb9879f20f9a7807bbbdd38096010eea3d277df889cc00b48a71fabb7`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 3}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mcar_v1'}`
- Parameter adequacy plan:
  - Compare directly against the clean control row before preferring missingness exposure.
  - Treat the larger hub-backed validation rung as the primary benchmark context for missingness transfer.
  - Keep class-imbalance reporting explicit on the large rung, but defer any dedicated skew ladder to TF-RD-017.
- Adequacy knobs to dimension explicitly:
  - explicit MCAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - BPC/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus is now a balanced 144-task DAGZoo grid with row totals `128/256/512/1024`, feature counts `6/10/14/20`, and explicit class coverage for every integer from `2` through `10`.
  - Trusted reruns use a single synthetic epoch only: one pass over 144 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `3` optimizer steps with an allowed short final batch.
  - The large validation pool follows the same hub bundle policy as the medium rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`, while offering the larger task set.
  - Historical 400-step TF-RD-010 executions are invalidated and retained only as non-canonical artifacts.
  - Trusted rerun work now flows through issues `#202`, `#203`, and `#204`.
  - Superseded for active execution by `tf_rd_010_classification_evolution_large_v2`; keep this row only as the preserved 3-step reset-contract reference.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_large_v1/delta_data_manifest_root_tf_rd_010_missingness_mcar/result_card.md`
- Benchmark metrics: pending

### 3. `delta_data_manifest_root_tf_rd_010_missingness_mar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Add a structured missingness row so TF-RD-010 can distinguish random masking from observed-feature-linked masking under the larger benchmark contract.
- Hypothesis: MAR may provide a clearer harder front than MCAR while remaining more interpretable than MNAR on the large validation pool.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mar_v1`.
- Expected effect: Structured MAR may provide a harder but still interpretable missingness front for the first TF-RD-010 classification benchmark program.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b4a68c18caa4818de5868b57f0195e6b1b54482bcb4db3e0c7e0fa9cf1bf2d52`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 3}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mar_v1'}`
- Parameter adequacy plan:
  - Compare directly against the clean control plus MCAR and MNAR before preferring structured missingness.
  - Treat the larger hub-backed validation rung as the primary benchmark context for missingness transfer.
  - Keep class-imbalance reporting explicit on the large rung, but defer any dedicated skew ladder to TF-RD-017.
- Adequacy knobs to dimension explicitly:
  - explicit MAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - BPC/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus is now a balanced 144-task DAGZoo grid with row totals `128/256/512/1024`, feature counts `6/10/14/20`, and explicit class coverage for every integer from `2` through `10`.
  - Trusted reruns use a single synthetic epoch only: one pass over 144 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `3` optimizer steps with an allowed short final batch.
  - The large validation pool follows the same hub bundle policy as the medium rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`, while offering the larger task set.
  - Historical 400-step TF-RD-010 executions are invalidated and retained only as non-canonical artifacts.
  - Trusted rerun work now flows through issues `#202`, `#203`, and `#204`.
  - Superseded for active execution by `tf_rd_010_classification_evolution_large_v2`; keep this row only as the preserved 3-step reset-contract reference.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_large_v1/delta_data_manifest_root_tf_rd_010_missingness_mar/result_card.md`
- Benchmark metrics: pending

### 4. `delta_data_manifest_root_tf_rd_010_missingness_mnar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Keep one strongest missingness row in the first draft package so TF-RD-010 can compare MCAR, MAR, and MNAR under the same larger benchmark contract.
- Hypothesis: MNAR may be the hardest missingness front, but it may also be the least interpretable candidate for the first evolved benchmark package.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mnar_v1`.
- Expected effect: Structured MNAR may be the strongest synthetic missingness perturbation, but it risks a less interpretable first benchmark-evolution read than MCAR or MAR.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mnar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `a41df0d6cebb383fb16f27ee0d466367a32f935dcbf83efc23e28a94a2c65a8a`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 3}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mnar_v1'}`
- Parameter adequacy plan:
  - Compare directly against the clean control plus MCAR and MAR before preferring the strongest self-masking option.
  - Treat the larger hub-backed validation rung as the primary benchmark context for missingness transfer.
  - Keep class-imbalance reporting explicit on the large rung, but defer any dedicated skew ladder to TF-RD-017.
- Adequacy knobs to dimension explicitly:
  - explicit MNAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - BPC/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus is now a balanced 144-task DAGZoo grid with row totals `128/256/512/1024`, feature counts `6/10/14/20`, and explicit class coverage for every integer from `2` through `10`.
  - Trusted reruns use a single synthetic epoch only: one pass over 144 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `3` optimizer steps with an allowed short final batch.
  - The large validation pool follows the same hub bundle policy as the medium rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`, while offering the larger task set.
  - Historical 400-step TF-RD-010 executions are invalidated and retained only as non-canonical artifacts.
  - Trusted rerun work now flows through issues `#202`, `#203`, and `#204`.
  - Superseded for active execution by `tf_rd_010_classification_evolution_large_v2`; keep this row only as the preserved 3-step reset-contract reference.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_large_v1/delta_data_manifest_root_tf_rd_010_missingness_mnar/result_card.md`
- Benchmark metrics: pending

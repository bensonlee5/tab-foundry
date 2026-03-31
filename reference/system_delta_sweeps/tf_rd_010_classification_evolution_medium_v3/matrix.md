# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v3/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v3/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_010_classification_evolution_medium_v3`
- Sweep status: `superseded`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v2`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v3/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `6963243a231e892f50bfcf537a5a052c960d87bc210b5a84180a9f452a3b0e1d`

## Locked Surface

- Anchor run id: `sd_tf_rd_010_classification_evolution_medium_v3_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1`
- Benchmark manifest: `data/manifests/bench/nanotabpfn_openml_classification_medium_v1/manifest.parquet`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_v1`
- Surface role: `custom`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `42342.9611`, final BPF `42342.9611`, final log loss `1.0897`, final Brier score `0.6623`, best ROC AUC `0.4874`, final ROC AUC `0.4780`, final training time `1384.3s`

## Anchor Comparison

Upstream reference: `EquiTabPFN` from `https://arxiv.org/abs/2502.06684`.

| Dimension | Upstream EquiTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| benchmark ownership | Not applicable. | `dagzoo` owns the synthetic training fronts while `tab-realdata-hub` owns the real-data validation manifests. | This sweep defines the repo-to-repo contract for the first benchmark-evolution lane. |
| classification head | Label-conditioning choices should stay modular so target handling can evolve after the backbone stabilizes. | Direct multiclass head with `many_class_base=10`. | Treat this as a bounded head/output evolution, not a staged hierarchical many-class port. |
| summary bandwidth | Historical TF-RD-021B evidence used `sandwich_summary_tokens_per_axis=4`. | The evolved benchmark surface uses `sandwich_summary_tokens_per_axis=3`. | The new benchmark package should evaluate the evolved contract directly rather than replaying the historical four-token anchor. |
| validation surface | Not applicable. | Hub-backed medium classification manifest at `data/manifests/bench/nanotabpfn_openml_classification_medium_v1/manifest.parquet`, materialized from `openml_classification_medium_v1.json`. | Keep the smaller hub-backed classification validation rung fixed while the synthetic training front reruns on the corrected sandwich and training surface. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control` | provenance | no | completed | none | Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests. | Treat this row as historical no-clipping overfit evidence only, then compare `medium_v4` row 1 against it before authorizing any further medium rerun work under issue `#205`. |
| 2 | `delta_data_manifest_root_tf_rd_010_missingness_mcar` | missingness | no | completed | none | Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Do not extend this package; keep this row as historical overfit evidence and gate any further medium execution on the `medium_v4` row 1 pilot review under issue `#205`. |
| 3 | `delta_data_manifest_root_tf_rd_010_missingness_mar` | missingness | no | completed | none | Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Do not extend this package; keep this row as historical overfit evidence and gate any further medium execution on the `medium_v4` row 1 pilot review under issue `#205`. |
| 4 | `delta_data_manifest_root_tf_rd_010_missingness_mnar` | missingness | no | ready | none | Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Do not execute this row under `medium_v3`; it was intentionally stopped after rows 1-3 showed severe early-best-step drift. Use the `medium_v4` row 1 pilot under issue `#205` instead. |

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
- Anchor delta: Use the evolved FiLM plus 3-summary-token sandwich contract and train on `tf_rd_010_dagzoo_medium_control_v2` while validating on the hub-owned medium classification manifest.
- Expected effect: Establish the TF-RD-010 classification control corpus that both the medium and large validation rungs will compare against.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `65061f420f2f22bacbfae3f837d7a71ea20b8fb0d297ebf56cff327c4726299b`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_dagzoo_medium_control_v2'}`
- Parameter adequacy plan:
  - Confirm `tab-realdata-hub#1` has materialized the medium classification manifest from `openml_classification_medium_v1.json` before execution.
  - Freeze the legacy `cls_benchmark_linear_multiclass_medium_v1` control baseline before treating any row outcome as a promotion or defer decision.
  - Rank by `final_bpc_at_matched_regime_budget`, then inspect raw log loss, calibration, runtime, and stability as guardrails.
- Adequacy knobs to dimension explicitly:
  - explicit dagzoo provenance for the classification control corpus
  - medium and large real-data validation separation via `tab-realdata-hub` manifests
  - class-count coverage, feature-count coverage, missingness policy, and minority-class floor on the validation side
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at \<=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - This row remains the intended TF-RD-010 medium reference for missingness and class-imbalance reporting on the medium validation pool.
  - Historical 400-step TF-RD-010 executions, the completed 3-step reset-contract rerun, and the completed clipped `tf_rd_010_classification_evolution_medium_v2` rerun remain historical context only.
  - Trusted rerun work now flows through issues `#202`, `#205`, and `#204`.
  - `tf_rd_010_classification_evolution_medium_v3` is preserved historical no-clipping evidence only: rows 1-3 hit very early best benchmark steps and then drifted badly, and row 4 was intentionally stopped rather than extended as canonical evidence. `tf_rd_010_classification_evolution_medium_v4` now owns the active accumulation/LR pilot.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v3_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v3/delta_data_manifest_root_tf_rd_010_dagzoo_medium_control/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_medium_v3_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1` with final BPC `42342.9611`, delta final BPC `+0.0000`, final BPF `42342.9611`, delta final BPF `+0.0000`, final log loss `1.0897`, delta final log loss `+0.0000`, final Brier score `0.6623`, delta final Brier score `+0.0000`, best ROC AUC `0.4874`, final ROC AUC `0.4780`, final-minus-best `+40609.0700`, delta final ROC AUC `+0.0000`, delta drift `+0.0000`, delta final training time `+0.0s`

### 2. `delta_data_manifest_root_tf_rd_010_missingness_mcar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Test whether moderate MCAR exposure improves robustness before structured missingness is considered on the medium validation pool.
- Hypothesis: MCAR may improve BPC and log-loss behavior on the evolved sandwich family without adding the stronger structure of MAR or MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mcar_v2`.
- Expected effect: Moderate MCAR should test whether the evolved sandwich target benefits from missingness exposure before any larger benchmark-front escalation.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mcar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0ddfec95f0b74003bc826f55d8c74642b4c18f845d213d8cd2c94a4b22ed305e`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mcar_v2'}`
- Parameter adequacy plan:
  - Compare directly against the clean control row before preferring missingness exposure.
  - Keep class-imbalance reporting explicit on the medium rung, but defer any dedicated skew ladder to TF-RD-017.
  - Use the larger hub-backed validation rung later as the main transfer check for any kept missingness front.
- Adequacy knobs to dimension explicitly:
  - explicit MCAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - BPC/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at \<=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - Historical 400-step TF-RD-010 executions, the completed 3-step reset-contract rerun, and the completed clipped `tf_rd_010_classification_evolution_medium_v2` rerun remain historical context only.
  - Trusted rerun work now flows through issues `#202`, `#205`, and `#204`.
  - `tf_rd_010_classification_evolution_medium_v3` is preserved historical no-clipping evidence only: rows 1-3 hit very early best benchmark steps and then drifted badly, and row 4 was intentionally stopped rather than extended as canonical evidence. `tf_rd_010_classification_evolution_medium_v4` now owns the active accumulation/LR pilot.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v3_02_delta_data_manifest_root_tf_rd_010_missingness_mcar_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v3/delta_data_manifest_root_tf_rd_010_missingness_mcar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_medium_v3_02_delta_data_manifest_root_tf_rd_010_missingness_mcar_v1` with final BPC `44073.4618`, delta final BPC `+1730.5007`, final BPF `44073.4612`, delta final BPF `+1730.5001`, final log loss `1.0993`, delta final log loss `+0.0096`, final Brier score `0.6680`, delta final Brier score `+0.0057`, best ROC AUC `0.4957`, final ROC AUC `0.5077`, final-minus-best `+41757.4350`, delta final ROC AUC `+0.0297`, delta drift `+1148.3650`, delta final training time `-7.9s`

### 3. `delta_data_manifest_root_tf_rd_010_missingness_mar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Add a structured missingness row so TF-RD-010 can distinguish random masking from observed-feature-linked masking under the evolved benchmark contract.
- Hypothesis: MAR may provide a clearer harder front than MCAR while remaining more interpretable than MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mar_v2`.
- Expected effect: Structured MAR may provide a harder but still interpretable missingness front for the first TF-RD-010 classification benchmark program.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b44fe74951a9a9a5d4c86540d6444816d87813f4985544e29cc5180e99113841`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mar_v2'}`
- Parameter adequacy plan:
  - Compare directly against the clean control plus MCAR and MNAR before preferring structured missingness.
  - Keep class-imbalance reporting explicit on the medium rung, but defer any dedicated skew ladder to TF-RD-017.
  - Use the larger hub-backed validation rung later as the main transfer check for any kept missingness front.
- Adequacy knobs to dimension explicitly:
  - explicit MAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - BPC/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at \<=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - Historical 400-step TF-RD-010 executions, the completed 3-step reset-contract rerun, and the completed clipped `tf_rd_010_classification_evolution_medium_v2` rerun remain historical context only.
  - Trusted rerun work now flows through issues `#202`, `#205`, and `#204`.
  - `tf_rd_010_classification_evolution_medium_v3` is preserved historical no-clipping evidence only: rows 1-3 hit very early best benchmark steps and then drifted badly, and row 4 was intentionally stopped rather than extended as canonical evidence. `tf_rd_010_classification_evolution_medium_v4` now owns the active accumulation/LR pilot.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v3_03_delta_data_manifest_root_tf_rd_010_missingness_mar_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v3/delta_data_manifest_root_tf_rd_010_missingness_mar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_medium_v3_03_delta_data_manifest_root_tf_rd_010_missingness_mar_v1` with final BPC `33422.2219`, delta final BPC `-8920.7393`, final BPF `33422.2219`, delta final BPF `-8920.7393`, final log loss `1.0972`, delta final log loss `+0.0075`, final Brier score `0.6669`, delta final Brier score `+0.0046`, best ROC AUC `0.4871`, final ROC AUC `0.5093`, final-minus-best `+31165.1757`, delta final ROC AUC `+0.0313`, delta drift `-9443.8943`, delta final training time `+78.7s`

### 4. `delta_data_manifest_root_tf_rd_010_missingness_mnar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Keep one strongest missingness row in the first draft package so TF-RD-010 can compare MCAR, MAR, and MNAR under the same medium benchmark contract.
- Hypothesis: MNAR may be the hardest missingness front, but it may also be the least interpretable candidate for the first evolved benchmark package.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mnar_v2`.
- Expected effect: Structured MNAR may be the strongest synthetic missingness perturbation, but it risks a less interpretable first benchmark-evolution read than MCAR or MAR.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mnar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ea0e050b642dd63199dbd8a0fec0f7d61dbccdfa74bb10fe85445a376c126029`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 1, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mnar_v2'}`
- Parameter adequacy plan:
  - Compare directly against the clean control plus MCAR and MAR before preferring the strongest self-masking option.
  - Keep class-imbalance reporting explicit on the medium rung, but defer any dedicated skew ladder to TF-RD-017.
  - Use the larger hub-backed validation rung later as the main transfer check for any kept missingness front.
- Adequacy knobs to dimension explicitly:
  - explicit MNAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - BPC/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at \<=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - Historical 400-step TF-RD-010 executions, the completed 3-step reset-contract rerun, and the completed clipped `tf_rd_010_classification_evolution_medium_v2` rerun remain historical context only.
  - Trusted rerun work now flows through issues `#202`, `#205`, and `#204`.
  - `tf_rd_010_classification_evolution_medium_v3` is preserved historical no-clipping evidence only: rows 1-3 hit very early best benchmark steps and then drifted badly, and row 4 was intentionally stopped rather than extended as canonical evidence. `tf_rd_010_classification_evolution_medium_v4` now owns the active accumulation/LR pilot.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v3/delta_data_manifest_root_tf_rd_010_missingness_mnar/result_card.md`
- Benchmark metrics: pending

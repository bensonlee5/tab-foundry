# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v2/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v2/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_010_classification_evolution_medium_v2`
- Sweep status: `ready`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v2/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `734e6888bd088b21268c5c36c53f695e5df9f7b0eab61b18e313299192a3ca6c`

## Locked Surface

- Anchor run id: `sd_tf_rd_010_classification_evolution_medium_v2_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_v1`
- Surface role: `custom`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `52841.8197`, final BPF `52841.8178`, final log loss `1.0928`, final Brier score `0.6643`, best ROC AUC `0.4746`, final ROC AUC `0.4821`, final training time `142.8s`

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
| 1 | `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control` | provenance | no | completed | none | Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests. | Execute this row under issue `#205` as the active medium anchor candidate for the expanded one-epoch 2500-step TF-RD-010 successor package. |
| 2 | `delta_data_manifest_root_tf_rd_010_missingness_mcar` | missingness | no | completed | none | Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Execute this row under issue `#205` after the active medium anchor is established, then compare it directly against that 2500-step successor anchor before carrying the front forward. |
| 3 | `delta_data_manifest_root_tf_rd_010_missingness_mar` | missingness | no | completed | none | Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Execute this row under issue `#205` after the active medium anchor is established, then compare it directly against that 2500-step successor anchor before carrying the front forward. |
| 4 | `delta_data_manifest_root_tf_rd_010_missingness_mnar` | missingness | no | completed | none | Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Execute this row under issue `#205` after the active medium anchor is established, then compare it directly against that 2500-step successor anchor before carrying the front forward. |

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
  - Historical 400-step TF-RD-010 executions are invalidated and retained only as non-canonical artifacts.
  - Trusted rerun work now flows through issues `#202`, `#205`, and `#204`.
  - The completed March 30, 2026 `tf_rd_010_classification_evolution_medium_v1` sweep remains the historical medium evidence; `tf_rd_010_classification_evolution_medium_v2` is the active medium successor regime.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v2_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1`.
  - Pending review for TF-RD-010 medium_v2 2500-step successor regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v2/delta_data_manifest_root_tf_rd_010_dagzoo_medium_control/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_medium_v2_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1` with final BPC `52841.8197`, delta final BPC `+0.0000`, final BPF `52841.8178`, delta final BPF `+0.0000`, final log loss `1.0928`, delta final log loss `+0.0000`, final Brier score `0.6643`, delta final Brier score `+0.0000`, best ROC AUC `0.4746`, final ROC AUC `0.4821`, final-minus-best `+51266.7673`, delta final ROC AUC `+0.0000`, delta drift `+0.0000`, delta final training time `+0.0s`

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
  - Historical 400-step TF-RD-010 executions are invalidated and retained only as non-canonical artifacts.
  - Trusted rerun work now flows through issues `#202`, `#205`, and `#204`.
  - The completed March 30, 2026 `tf_rd_010_classification_evolution_medium_v1` sweep remains the historical medium evidence; `tf_rd_010_classification_evolution_medium_v2` is the active medium successor regime.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v2_02_delta_data_manifest_root_tf_rd_010_missingness_mcar_v1`.
  - Pending review for TF-RD-010 medium_v2 2500-step successor regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v2/delta_data_manifest_root_tf_rd_010_missingness_mcar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_medium_v2_02_delta_data_manifest_root_tf_rd_010_missingness_mcar_v1` with final BPC `46448.7082`, delta final BPC `-6393.1115`, final BPF `46448.7088`, delta final BPF `-6393.1090`, final log loss `1.1091`, delta final log loss `+0.0163`, final Brier score `0.6748`, delta final Brier score `+0.0105`, best ROC AUC `0.4831`, final ROC AUC `0.4944`, final-minus-best `+44210.7182`, delta final ROC AUC `+0.0123`, delta drift `-7056.0491`, delta final training time `-1.3s`

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
  - Historical 400-step TF-RD-010 executions are invalidated and retained only as non-canonical artifacts.
  - Trusted rerun work now flows through issues `#202`, `#205`, and `#204`.
  - The completed March 30, 2026 `tf_rd_010_classification_evolution_medium_v1` sweep remains the historical medium evidence; `tf_rd_010_classification_evolution_medium_v2` is the active medium successor regime.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v2_03_delta_data_manifest_root_tf_rd_010_missingness_mar_v3`.
  - Pending review for TF-RD-010 medium_v2 2500-step successor regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v2/delta_data_manifest_root_tf_rd_010_missingness_mar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_medium_v2_03_delta_data_manifest_root_tf_rd_010_missingness_mar_v3` with final BPC `49182.1258`, delta final BPC `-3659.6939`, final BPF `49182.1245`, delta final BPF `-3659.6933`, final log loss `1.0880`, delta final log loss `-0.0048`, final Brier score `0.6607`, delta final Brier score `-0.0036`, best ROC AUC `0.4775`, final ROC AUC `0.5043`, final-minus-best `+46587.2124`, delta final ROC AUC `+0.0222`, delta drift `-4679.5549`, delta final training time `-4.6s`

### 4. `delta_data_manifest_root_tf_rd_010_missingness_mnar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Keep one strongest missingness row in the first draft package so TF-RD-010 can compare MCAR, MAR, and MNAR under the same benchmark contract.
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
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to 159984 corpus manifest records/tasks: 144 invocation cells x 1111 datasets, still capped at \<=1024 total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over 159984 corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - The medium validation pool follows the same hub bundle policy as the large rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`.
  - Historical 400-step TF-RD-010 executions are invalidated and retained only as non-canonical artifacts.
  - Trusted rerun work now flows through issues `#202`, `#205`, and `#204`.
  - The completed March 30, 2026 `tf_rd_010_classification_evolution_medium_v1` sweep remains the historical medium evidence; `tf_rd_010_classification_evolution_medium_v2` is the active medium successor regime.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v2_04_delta_data_manifest_root_tf_rd_010_missingness_mnar_v1`.
  - Pending review for TF-RD-010 medium_v2 2500-step successor regime.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v2/delta_data_manifest_root_tf_rd_010_missingness_mnar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_medium_v2_04_delta_data_manifest_root_tf_rd_010_missingness_mnar_v1` with final BPC `21138.8335`, delta final BPC `-31702.9862`, final BPF `21138.8345`, delta final BPF `-31702.9833`, final log loss `1.0966`, delta final log loss `+0.0038`, final Brier score `0.6657`, delta final Brier score `+0.0014`, best ROC AUC `0.5116`, final ROC AUC `0.4946`, final-minus-best `+18718.6642`, delta final ROC AUC `+0.0125`, delta drift `-32548.1031`, delta final training time `-11.0s`

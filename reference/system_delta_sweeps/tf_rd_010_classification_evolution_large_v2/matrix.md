# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_large_v2/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_large_v2/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_010_classification_evolution_large_v2`
- Sweep status: `completed`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v4`
- Complexity level: `classification_lg`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_010_classification_evolution_large_v2/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `d85d8f8690f3586e6e579f701c0bd418c43ceb7ced09cf0352373a35fc5efcad`

## Locked Surface

- Anchor run id: `sd_tf_rd_010_classification_evolution_large_v2_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_large_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_large_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_v1`
- Surface role: `custom`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.0860`, final BPF `2.0860`, final log loss `0.8974`, final Brier score `0.5465`, best ROC AUC `0.6324`, final ROC AUC `0.6324`, final training time `7449.8s`

## Anchor Comparison

Upstream reference: `EquiTabPFN` from `https://arxiv.org/abs/2502.06684`.

| Dimension | Upstream EquiTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| benchmark ownership | Not applicable. | `dagzoo` owns the synthetic training fronts while `tab-realdata-hub` owns the real-data validation manifests. | This sweep defines the harder validation rung for the repo-to-repo benchmark contract. |
| classification head | Label-conditioning choices should stay modular so target handling can evolve after the backbone stabilizes. | Direct multiclass head with `many_class_base=10`. | Treat this as a bounded head/output evolution, not a staged hierarchical many-class port. |
| summary bandwidth | Historical TF-RD-021B evidence used `sandwich_summary_tokens_per_axis=4`. | The evolved benchmark surface uses `sandwich_summary_tokens_per_axis=3`. | The new benchmark package should evaluate the evolved contract directly rather than replaying the historical four-token anchor. |
| validation surface | Not applicable. | Hub-backed large classification manifest under local benchmark-manifest id `openml_classification_large_v1`, materialized from `openml_classification_large_v1.json`. | Keep the larger hub-backed classification validation rung fixed while the synthetic training front reruns on the corrected sandwich and training surface. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control` | provenance | no | completed | none | Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests. | Keep the original `medium_v4` control as the TF-RD-010 carried comparator, and preserve this completed large control row as the harder-rung reference; do not reopen `large_v2` within TF-RD-010. |
| 2 | `delta_data_manifest_root_tf_rd_010_missingness_mcar` | missingness | no | completed | none | Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Keep this row deferred: it did not beat the carried large control row, so TF-RD-010 does not promote missingness exposure from this benchmark-only transfer. |
| 3 | `delta_data_manifest_root_tf_rd_010_missingness_mar` | missingness | no | completed | none | Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Keep this row deferred: it did not beat the carried large control row, so TF-RD-010 does not promote structured missingness from this benchmark-only transfer. |
| 4 | `delta_data_manifest_root_tf_rd_010_missingness_mnar` | missingness | no | completed | none | Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Keep this row deferred: it did not beat the carried large control row, so TF-RD-010 does not promote strongest-missingness exposure from this benchmark-only transfer. |

## Detailed Rows

### 1. `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests.
- Rationale: Establish the TF-RD-010 classification control front on the larger benchmark rung before reading any missingness harder-front effect.
- Hypothesis: The evolved sandwich family should first be judged on the TF-RD-010 control corpus against the frozen local large validation bundle, then promoted as the same-invocation anchor for rows 2 through 4.
- Upstream delta: Not applicable; this is a repo-local synthetic training-front contract tied to the first benchmark-evolution lane.
- Anchor delta: Use the evolved FiLM plus 3-summary-token sandwich contract and benchmark the completed control pilot trained on `tf_rd_010_dagzoo_medium_control_curated_v5` against the local large classification manifest without retraining.
- Expected effect: Establish the TF-RD-010 classification control corpus that both the medium and large validation rungs will compare against.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ca3aadfde8968c65d71fe6101418fb0b868106edb7c8452cae95fe0529c126a9`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_dagzoo_medium_control_curated_v5'}`
- Reuse train artifact: `outputs/research/adequacy/tf_rd_010_synthetic_adequacy_v3/pilot/production_control_curated_v5/train`
- Reuse training surface fingerprint: `1614c767510feacd669b4868fd2dfacbe7332f0b64b9c694c448caca85794d20`
- Parameter adequacy plan:
  - Confirm the local `openml_classification_large_v1` manifest still matches frozen baseline task ids `[363685, 363699, 363707]` before execution.
  - Freeze the legacy `cls_benchmark_linear_multiclass_large_v1` control baseline before treating any row outcome as a promotion or defer decision.
  - Rank by `final_log_loss_at_matched_regime_budget`, interpreted explicitly as label-target log loss per test cell, then inspect calibration, runtime, and stability as guardrails.
- Adequacy knobs to dimension explicitly:
  - explicit dagzoo provenance for the classification control corpus
  - medium and large real-data validation separation via `tab-realdata-hub` manifests
  - class-count coverage, feature-count coverage, missingness policy, and minority-class floor on the validation side
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest lineage.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to `159984` corpus manifest records/tasks: `144` invocation cells x `1111` datasets, still capped at `<=1024` total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over `159984` corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - This row remains the intended TF-RD-010 large reference for missingness transfer and class-imbalance reporting on the large validation pool.
  - `tf_rd_010_classification_evolution_large_v2` shares the pilot-aligned training surface with the refreshed medium package: `task_batch_size=16`, `grad_accum_steps=4`, `runtime.grad_clip=0.0`, `max_steps=2500`, linear schedule with `warmup_ratio=0.10`, `lr_max=1e-3`, and `optimizer.min_lr=1e-5`.
  - Reuse the completed pilot control training artifact at `outputs/research/adequacy/tf_rd_010_synthetic_adequacy_v3/pilot/production_control_curated_v5/train`; do not retrain the control row for `large_v2`.
  - The local large benchmark bundle is preflighted against the frozen `cls_benchmark_linear_multiclass_large_v1` contract with task ids `[363685, 363699, 363707]`.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_large_v2_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1`.
  - With the original `medium_v4` control kept over the worse `medium_v5` replay, this completed large control row remains the best large-rung result and the carried harder-rung reference.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_large_v2/delta_data_manifest_root_tf_rd_010_dagzoo_medium_control/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_large_v2_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1` with final log loss `0.8974`, delta final log loss `+0.0000`, final Brier score `0.5465`, delta final brier score `+0.0000`, final ROC AUC `0.6324`, delta final roc auc `+0.0000`, final BPC (legacy feature-cell diagnostic) `2.0860`, delta final bpc (legacy feature-cell diagnostic) `+0.0000`, final BPF (legacy feature-cell diagnostic) `2.0860`, delta final bpf (legacy feature-cell diagnostic) `+0.0000`, best ROC AUC `0.6324`, delta final training time `+0.0s`

### 2. `delta_data_manifest_root_tf_rd_010_missingness_mcar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Test whether moderate MCAR exposure improves robustness on the larger benchmark rung before structured missingness is considered.
- Hypothesis: MCAR may improve label-target log loss per test cell on the evolved sandwich family under the frozen local large validation pool without adding the stronger structure of MAR or MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and benchmark the completed `tf_rd_010_missingness_mcar_v3` training artifact on the local large classification manifest without retraining.
- Expected effect: Moderate MCAR should test whether the evolved sandwich target benefits from missingness exposure before any larger benchmark-front escalation.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mcar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ce551f657c6babd75acbfe061796a947eea6ba4849fe323189bdb42eb3aa2e9c`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mcar_v3'}`
- Reuse train artifact: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mcar/sd_tf_rd_010_classification_evolution_medium_v4_02_delta_data_manifest_root_tf_rd_010_missingness_mcar_v1/train`
- Reuse training surface fingerprint: `60f35937e0c9701505f061f4c886e3ee2027c5376fcb492fb32c097b15b73fa7`
- Parameter adequacy plan:
  - Compare directly against the clean control row before preferring missingness exposure.
  - Treat the larger local large validation rung as the primary benchmark context for missingness transfer.
  - Keep class-imbalance reporting explicit on the large rung, but defer any dedicated skew ladder to TF-RD-017.
- Adequacy knobs to dimension explicitly:
  - explicit MCAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - natural-log CE/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest lineage.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to `159984` corpus manifest records/tasks: `144` invocation cells x `1111` datasets, still capped at `<=1024` total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over `159984` corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - Reuse the completed `medium_v4` MCAR training artifact; do not retrain this row for `large_v2`.
  - The control comparison for this sweep still uses curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`, while this missingness row remains `include_all` `v3`.
  - TF-RD-010 keeps the original `medium_v4` control as the carried comparator, and this missingness row remains worse than the completed large control row.
  - The local large benchmark bundle is preflighted against the frozen `cls_benchmark_linear_multiclass_large_v1` contract with task ids `[363685, 363699, 363707]`.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_large_v2_02_delta_data_manifest_root_tf_rd_010_missingness_mcar_v1`.
  - This row stays deferred on the large rung and does not promote missingness exposure from the first TF-RD-010 medium/large package.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_large_v2/delta_data_manifest_root_tf_rd_010_missingness_mcar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_large_v2_02_delta_data_manifest_root_tf_rd_010_missingness_mcar_v1` with final log loss `0.9155`, delta final log loss `+0.0181`, final Brier score `0.5563`, delta final brier score `+0.0098`, final ROC AUC `0.6076`, delta final roc auc `-0.0248`, final BPC (legacy feature-cell diagnostic) `2.0972`, delta final bpc (legacy feature-cell diagnostic) `+0.0111`, final BPF (legacy feature-cell diagnostic) `2.0972`, delta final bpf (legacy feature-cell diagnostic) `+0.0111`, best ROC AUC `0.6076`, delta final training time `+3566.0s`

### 3. `delta_data_manifest_root_tf_rd_010_missingness_mar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Add a structured missingness row so TF-RD-010 can distinguish random masking from observed-feature-linked masking under the larger benchmark contract.
- Hypothesis: MAR may provide a clearer harder front than MCAR while remaining more interpretable than MNAR on the local large validation pool.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and benchmark the completed `tf_rd_010_missingness_mar_v3` training artifact on the local large classification manifest without retraining.
- Expected effect: Structured MAR may provide a harder but still interpretable missingness front for the first TF-RD-010 classification benchmark program.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `0ddc1ffd07251f83afe1a7ad2c79927180031518bbeb9a6364f738e7ca9592a8`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mar_v3'}`
- Reuse train artifact: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mar/sd_tf_rd_010_classification_evolution_medium_v4_03_delta_data_manifest_root_tf_rd_010_missingness_mar_v1/train`
- Reuse training surface fingerprint: `71fd6c814bcd7c0a1799c31746551df915b0915a752da53124df3be6f1f128ee`
- Parameter adequacy plan:
  - Compare directly against the clean control plus MCAR and MNAR before preferring structured missingness.
  - Treat the larger local large validation rung as the primary benchmark context for missingness transfer.
  - Keep class-imbalance reporting explicit on the large rung, but defer any dedicated skew ladder to TF-RD-017.
- Adequacy knobs to dimension explicitly:
  - explicit MAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - natural-log CE/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest lineage.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to `159984` corpus manifest records/tasks: `144` invocation cells x `1111` datasets, still capped at `<=1024` total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over `159984` corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - Reuse the completed `medium_v4` MAR training artifact; do not retrain this row for `large_v2`.
  - The control comparison for this sweep still uses curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`, while this missingness row remains `include_all` `v3`.
  - TF-RD-010 keeps the original `medium_v4` control as the carried comparator, and this structured-missingness row remains worse than the completed large control row.
  - The local large benchmark bundle is preflighted against the frozen `cls_benchmark_linear_multiclass_large_v1` contract with task ids `[363685, 363699, 363707]`.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_large_v2_03_delta_data_manifest_root_tf_rd_010_missingness_mar_v1`.
  - This row stays deferred on the large rung and does not promote structured missingness from the first TF-RD-010 medium/large package.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_large_v2/delta_data_manifest_root_tf_rd_010_missingness_mar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_large_v2_03_delta_data_manifest_root_tf_rd_010_missingness_mar_v1` with final log loss `0.9419`, delta final log loss `+0.0444`, final Brier score `0.5724`, delta final brier score `+0.0258`, final ROC AUC `0.5890`, delta final roc auc `-0.0433`, final BPC (legacy feature-cell diagnostic) `2.1179`, delta final bpc (legacy feature-cell diagnostic) `+0.0318`, final BPF (legacy feature-cell diagnostic) `2.1179`, delta final bpf (legacy feature-cell diagnostic) `+0.0318`, best ROC AUC `0.5890`, delta final training time `+3496.8s`

### 4. `delta_data_manifest_root_tf_rd_010_missingness_mnar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Keep one strongest missingness row in the first draft package so TF-RD-010 can compare MCAR, MAR, and MNAR under the same larger benchmark contract.
- Hypothesis: MNAR may be the hardest missingness front, but it may also be the least interpretable candidate for the first evolved benchmark package on the local large validation pool.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and benchmark the completed `tf_rd_010_missingness_mnar_v3` training artifact on the local large classification manifest without retraining.
- Expected effect: Structured MNAR may be the strongest synthetic missingness perturbation, but it risks a less interpretable first benchmark-evolution read than MCAR or MAR.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mnar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `f5ac7886cb4404df75e7a9a85985282c8c07326a2dad8f24e83e485258b56e6c`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'loader_pin_memory': False, 'loader_persistent_workers': False, 'loader_prefetch_factor': None, 'non_blocking_device_transfer': False, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': False, 'compile_dynamic': False, 'compile_backend': 'inductor', 'compile_mode': 'max-autotune-no-cudagraphs', 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mnar_v3'}`
- Reuse train artifact: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v4/delta_data_manifest_root_tf_rd_010_missingness_mnar/sd_tf_rd_010_classification_evolution_medium_v4_04_delta_data_manifest_root_tf_rd_010_missingness_mnar_v1/train`
- Reuse training surface fingerprint: `5c2fe334e601ae78d310357633456e020bc186c4a8fffcb117ce9b048bd674f9`
- Parameter adequacy plan:
  - Compare directly against the clean control plus MCAR and MAR before preferring the strongest self-masking option.
  - Treat the larger local large validation rung as the primary benchmark context for missingness transfer.
  - Keep class-imbalance reporting explicit on the large rung, but defer any dedicated skew ladder to TF-RD-017.
- Adequacy knobs to dimension explicitly:
  - explicit MNAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - natural-log CE/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest lineage.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to `159984` corpus manifest records/tasks: `144` invocation cells x `1111` datasets, still capped at `<=1024` total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over `159984` corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - Reuse the completed `medium_v4` MNAR training artifact; do not retrain this row for `large_v2`.
  - The control comparison for this sweep still uses curated `accepted_only` `tf_rd_010_dagzoo_medium_control_curated_v5`, while this missingness row remains `include_all` `v3`.
  - TF-RD-010 keeps the original `medium_v4` control as the carried comparator, and this strongest-missingness row remains worse than the completed large control row.
  - The local large benchmark bundle is preflighted against the frozen `cls_benchmark_linear_multiclass_large_v1` contract with task ids `[363685, 363699, 363707]`.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_large_v2_04_delta_data_manifest_root_tf_rd_010_missingness_mnar_v1`.
  - This row stays deferred on the large rung and does not promote strongest missingness from the first TF-RD-010 medium/large package.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_large_v2/delta_data_manifest_root_tf_rd_010_missingness_mnar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_large_v2_04_delta_data_manifest_root_tf_rd_010_missingness_mnar_v1` with final log loss `0.9412`, delta final log loss `+0.0437`, final Brier score `0.5726`, delta final brier score `+0.0260`, final ROC AUC `0.6008`, delta final roc auc `-0.0316`, final BPC (legacy feature-cell diagnostic) `2.1310`, delta final bpc (legacy feature-cell diagnostic) `+0.0450`, final BPF (legacy feature-cell diagnostic) `2.1310`, delta final bpf (legacy feature-cell diagnostic) `+0.0450`, best ROC AUC `0.6008`, delta final training time `+3316.4s`

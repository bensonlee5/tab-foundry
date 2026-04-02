# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v5/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v5/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_010_classification_evolution_medium_v5`
- Sweep status: `completed`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v4`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_010_classification_evolution_medium_v5/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `9f3ff7d433e8154ce9f58fb760aa52e39f96de9e31d66b358765d24c184bd6e6`

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
| benchmark ownership | Not applicable. | `dagzoo` owns the synthetic training fronts while `tab-realdata-hub` owns the real-data validation manifests. | This sweep preserves the repo-to-repo contract from `medium_v4` while isolating the row-order confound on the control row. |
| classification head | Label-conditioning choices should stay modular so target handling can evolve after the backbone stabilizes. | Direct multiclass head with `many_class_base=10`. | Treat this as the same bounded head/output evolution rather than reopening staged or hierarchical many-class variants. |
| summary bandwidth | Historical TF-RD-021B evidence used `sandwich_summary_tokens_per_axis=4`. | The evolved benchmark surface uses `sandwich_summary_tokens_per_axis=3`. | `medium_v5` keeps the same evolved sandwich contract and changes only the control-row training provenance. |
| validation surface | Not applicable. | Hub-backed medium classification manifest under local benchmark-manifest id `openml_classification_medium_v1`, materialized from `openml_classification_medium_v1.json`. | Keep the smaller hub-backed classification validation rung fixed while the control row is retrained under the sorted-order code path. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control` | provenance | no | completed | none | Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests. | Preserve this sorted-order replay as negative evidence only: keep the original `medium_v4` control as the TF-RD-010 carried comparator for both medium and large interpretation, and do not reopen the missingness rows inside TF-RD-010. |

## Detailed Rows

### 1. `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests.
- Rationale: Remove the row-order confound between `tf_rd_010_classification_evolution_medium_v4` row 1 and rows 2 through 4 before promoting any TF-RD-010 missingness recommendation on the medium benchmark rung.
- Hypothesis: A fresh sorted-order retrain on `tf_rd_010_dagzoo_medium_control_curated_v5` should provide the canonical medium control comparator for `medium_v4` rows 2 through 4 without changing the benchmark contract.
- Upstream delta: Not applicable; this is a repo-local synthetic training-front contract tied to the first benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and retrain `tf_rd_010_dagzoo_medium_control_curated_v5` under the current sorted-order code path before benchmarking on the hub-owned medium classification manifest.
- Expected effect: Establish the TF-RD-010 classification control corpus that both the medium and large validation rungs will compare against.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `72b828ff3f0ff3295d3758e4a4d194a152569d0f83f8b1bec2d1da70d39be60e`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'no', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_dagzoo_medium_control_curated_v5'}`
- Parameter adequacy plan:
  - Confirm `tab-realdata-hub#1` has materialized the medium classification manifest from `openml_classification_medium_v1.json` before execution.
  - Freeze the legacy `cls_benchmark_linear_multiclass_medium_v1` control baseline before treating any row outcome as a promotion or defer decision.
  - Rank by `final_log_loss_at_matched_regime_budget`, interpreted explicitly as label-target log loss per test cell, then inspect calibration, runtime, stability, and any retained legacy cell-likelihood diagnostics as guardrails.
  - Use this completed comparison to resolve the control-order confound explicitly: keep the original `medium_v4` control if this replay does not beat `0.6811727401`.
- Adequacy knobs to dimension explicitly:
  - explicit dagzoo provenance for the classification control corpus
  - medium and large real-data validation separation via `tab-realdata-hub` manifests
  - class-count coverage, feature-count coverage, missingness policy, and minority-class floor on the validation side
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This corpus keeps the same balanced 144-cell DAGZoo front shape but expands it to `159984` corpus manifest records/tasks: `144` invocation cells x `1111` datasets, still capped at `<=1024` total rows per synthetic dataset.
  - Trusted executions use a single synthetic epoch only: one pass over `159984` corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `2500` optimizer steps with an allowed short final batch.
  - `tf_rd_010_classification_evolution_medium_v4` remains completed historical evidence: row 1 reused an older unsorted control train artifact while rows 2 through 4 were trained after dataset sorting was enabled.
  - `tf_rd_010_classification_evolution_medium_v5` is intentionally a single-row follow-up that reopens only the control anchor to remove that order confound.
  - `tf_rd_010_classification_evolution_medium_v5` uses the pilot-aligned control contract: `task_batch_size=16`, `grad_accum_steps=4`, `runtime.grad_clip=0.0`, `max_steps=2500`, linear schedule with `warmup_ratio=0.10`, `lr_max=1e-3`, and `optimizer.min_lr=1e-5`.
  - This row benchmarks only `best.pt` and `latest.pt` after the fresh sorted-order retrain so the follow-up stays scoped to the control confound.
  - This row must be trained fresh under the current sorted-order code path; do not reuse `outputs/research/adequacy/tf_rd_010_synthetic_adequacy_v3/pilot/production_control_curated_v5/train` or any earlier unsorted control artifact.
  - The repo-local `openml_classification_medium_v1` manifest is still a stale placeholder; canonical benchmark registration must use the intended hub-backed medium classification manifest.
  - `final_log_loss=0.6849303354` is worse than the kept `medium_v4` control at `0.6811727401`, so TF-RD-010 keeps the original `medium_v4` control as the carried comparator.
  - That carry-forward decision leaves `large_v2` interpreted against the original `medium_v4` control, with no missingness promotion from this control-order follow-up.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_medium_v5_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_medium_v5/delta_data_manifest_root_tf_rd_010_dagzoo_medium_control/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_medium_v5_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v2` with final log loss `0.6849`, delta final log loss `+0.0038`, final Brier score `0.4246`, delta final brier score `+0.0017`, final ROC AUC `0.6044`, delta final roc auc `-0.0050`, final BPC (legacy feature-cell diagnostic) `2.1154`, delta final bpc (legacy feature-cell diagnostic) `+0.0017`, final BPF (legacy feature-cell diagnostic) `2.1154`, delta final bpf (legacy feature-cell diagnostic) `+0.0017`, best ROC AUC `0.6044`, delta final training time `+391.0s`

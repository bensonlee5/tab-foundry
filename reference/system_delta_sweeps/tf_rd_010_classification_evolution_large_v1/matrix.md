# System Delta Matrix

This file reflects the reset TF-RD-010 queue state in
`reference/system_delta_sweeps/tf_rd_010_classification_evolution_large_v1/queue.yaml`
after invalidating the previously recorded large executions as canonical
evidence.

## Sweep

- Sweep id: `tf_rd_010_classification_evolution_large_v1`
- Sweep status: `ready`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v1`
- Complexity level: `classification_lg`

## Locked Surface

- Anchor run id: `null`
- Benchmark manifest: legacy local benchmark id `nanotabpfn_openml_classification_large_v1`, materialized from upstream bundle `openml_classification_large_v1.json`
- Control baseline id: `cls_benchmark_linear_multiclass_large_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_v1`
- Surface role: `custom`
- Comparison policy: `anchor_only`
- Anchor metrics: pending trusted rerun

## Anchor Comparison

Upstream reference: `EquiTabPFN` from `https://arxiv.org/abs/2502.06684`.

| Dimension | Upstream EquiTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| benchmark ownership | Not applicable. | `dagzoo` owns the synthetic training fronts while `tab-realdata-hub` owns the real-data validation manifests. | This sweep defines the harder validation rung for the repo-to-repo benchmark contract. |
| classification head | Label-conditioning choices should stay modular so target handling can evolve after the backbone stabilizes. | direct multiclass head with `many_class_base=10`. | Treat this as a bounded head/output evolution, not a staged hierarchical many-class port. |
| summary bandwidth | Historical TF-RD-021B evidence used `sandwich_summary_tokens_per_axis=4`. | The evolved benchmark surface uses `sandwich_summary_tokens_per_axis=3`. | The new benchmark package should evaluate the evolved contract directly rather than replaying the historical four-token anchor. |
| validation surface | Not applicable. | Hub-backed large classification manifest under the local benchmark-manifest output root, using the legacy local id `nanotabpfn_openml_classification_large_v1`, materialized from `openml_classification_large_v1.json`. | Keep the larger hub-backed classification validation rung fixed while the synthetic training front reruns on the corrected sandwich and training surface. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control` | provenance | no | ready | Point training at the TF-RD-010 control corpus: a balanced 144-task DAGZoo grid capped at `<=1024` rows per dataset, with row totals `128/256/512/1024`, feature counts `6/10/14/20`, and class coverage `2..10`. | Rerun this row after issues `#204` and `#205` land, then use it as the trusted large anchor for the remaining large-rung comparisons under issue `#203`. |
| 2 | `delta_data_manifest_root_tf_rd_010_missingness_mcar` | missingness | no | ready | Point training at the TF-RD-010 MCAR corpus on the same balanced 144-task row/feature/class grid while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Rerun this row after the trusted large anchor is re-established, then compare it directly against the anchor before carrying any missingness front forward under issue `#203`. |
| 3 | `delta_data_manifest_root_tf_rd_010_missingness_mar` | missingness | no | ready | Point training at the TF-RD-010 MAR corpus on the same balanced 144-task row/feature/class grid while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Rerun this row after the trusted large anchor is re-established, then compare it against the corrected control and MCAR reads under issue `#203`. |
| 4 | `delta_data_manifest_root_tf_rd_010_missingness_mnar` | missingness | no | ready | Point training at the TF-RD-010 MNAR corpus on the same balanced 144-task row/feature/class grid while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Rerun this row after the trusted large anchor is re-established, then compare it against the corrected control, MCAR, and MAR reads under issue `#203`. |

## Reset Notes

- The TF-RD-010 benchmark contract remains active and still ranks rows by `final_bpc_at_matched_regime_budget`.
- `tab-realdata-hub` remains the owner of the medium and large validation bundles with `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`, while the large rung offers the larger task set.
- TF-RD-010 synthetic reruns now use a balanced 144-task DAGZoo grid with every dataset capped at `<=1024` total rows and explicit class coverage `2..10`.
- Trusted reruns use a single synthetic epoch only: one pass over corpus manifest records/tasks at `prior_dump_batch_size=64`, which resolves to `3` optimizer steps here.
- The previous 400-step large results are retained only as historical artifacts; they are not canonical evidence.
- Trusted rerun work now flows through issues `#202`, `#203`, and `#204`.

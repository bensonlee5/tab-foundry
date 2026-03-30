# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_010_classification_evolution_large_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_010_classification_evolution_large_v1`
- Sweep status: `completed`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v1`
- Complexity level: `classification_lg`

## Locked Surface

- Anchor run id: `sd_tf_rd_010_classification_evolution_large_v1_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1`
- Benchmark manifest: `data/manifests/bench/nanotabpfn_openml_classification_large_v1/manifest.parquet`
- Control baseline id: `cls_benchmark_linear_multiclass_large_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_v1`
- Surface role: `custom`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `115233.1950`, final BPF `115233.1963`, final log loss `1.0783`, final Brier score `0.6531`, best ROC AUC `0.5396`, final ROC AUC `0.5396`, final training time `111.3s`

## Anchor Comparison

Upstream reference: `EquiTabPFN` from `https://arxiv.org/abs/2502.06684`.

| Dimension | Upstream EquiTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| benchmark ownership | Not applicable. | `dagzoo` owns the synthetic training fronts while `tab-realdata-hub` owns the real-data validation manifests. | This sweep defines the harder validation rung for the repo-to-repo benchmark contract. |
| classification head | Label-conditioning choices should stay modular so target handling can evolve after the backbone stabilizes. | Direct multiclass head with `many_class_base=10`. | Treat this as a bounded head/output evolution, not a staged hierarchical many-class port. |
| summary bandwidth | Historical TF-RD-021B evidence used `sandwich_summary_tokens_per_axis=4`. | The evolved benchmark surface uses `sandwich_summary_tokens_per_axis=3`. | The new benchmark package should evaluate the evolved contract directly rather than replaying the historical four-token anchor. |
| validation surface | Not applicable. | Hub-backed large classification manifest at `data/manifests/bench/nanotabpfn_openml_classification_large_v1/manifest.parquet`, materialized from `openml_classification_large_v1.json`. | Keep the larger hub-backed classification validation rung fixed while the synthetic training front changes across rows. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control` | provenance | no | completed | none | Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests. | Completed as the locked large control anchor; keep this row as the fixed TF-RD-010 reference and move any improvement attempts to TF-RD-021 or later higher-budget follow-up instead of reopening baseline definition. |
| 2 | `delta_data_manifest_root_tf_rd_010_missingness_mcar` | missingness | no | completed | none | Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Completed as mixed negative evidence; although final BPC improved sharply versus the control anchor, this row failed stability guardrails and degraded final ROC AUC, so keep the clean control anchor and do not promote MCAR from this short-run pass. |
| 3 | `delta_data_manifest_root_tf_rd_010_missingness_mar` | missingness | no | completed | none | Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Completed as mixed negative evidence; although final BPC improved versus the control anchor, this row failed stability guardrails and degraded final ROC AUC, so keep the clean control anchor and do not promote MAR on the large rung. |
| 4 | `delta_data_manifest_root_tf_rd_010_missingness_mnar` | missingness | no | completed | none | Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Completed as negative evidence; this row produced the worst large-rung BPC outcome and the largest drift signal, so keep the clean control anchor and do not promote MNAR on the large rung. |

## Detailed Rows

### 1. `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests.
- Rationale: Establish the TF-RD-010 classification control front before reading any missingness harder-front effect on the larger benchmark rung.
- Hypothesis: The evolved sandwich family should first be judged on the TF-RD-010 control corpus (`n_classes_min=2`) against the larger hub validation manifest.
- Upstream delta: Not applicable; this is a repo-local synthetic training-front contract tied to the first benchmark-evolution lane.
- Anchor delta: Use the evolved FiLM plus 3-summary-token sandwich contract and train on `tf_rd_010_dagzoo_medium_control_v1` while validating on the hub-owned large classification manifest.
- Expected effect: Establish the TF-RD-010 classification control corpus that both the medium and large validation rungs will compare against.
- Effective labels: model=`cls_benchmark_sandwich_classification_evolution_v1`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
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
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - This row is the fixed TF-RD-010 large reference for missingness transfer and class-imbalance reporting on the large validation pool.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_large_v1_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Sweep summary marked this row `stability=fail` at the 400-step short-run budget; treat it as the fixed short-screen anchor rather than promotion evidence.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_large_v1/delta_data_manifest_root_tf_rd_010_dagzoo_medium_control/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_large_v1_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1` with final BPC `115233.1950`, delta final BPC `+0.0000`, final BPF `115233.1963`, delta final BPF `+0.0000`, final log loss `1.0783`, delta final log loss `+0.0000`, final Brier score `0.6531`, delta final Brier score `+0.0000`, best ROC AUC `0.5396`, final ROC AUC `0.5396`, final-minus-best `+0.0000`, delta final ROC AUC `+0.0000`, delta drift `+0.0000`, delta final training time `+0.0s`

### 2. `delta_data_manifest_root_tf_rd_010_missingness_mcar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Test whether moderate MCAR exposure improves robustness on the larger benchmark rung before structured missingness is considered.
- Hypothesis: MCAR may improve BPC/log-loss behavior on the evolved sandwich family under the larger hub validation pool without adding the stronger structure of MAR or MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mcar_v1`.
- Expected effect: Moderate MCAR should test whether the evolved sandwich target benefits from missingness exposure before any larger benchmark-front escalation.
- Effective labels: model=`cls_benchmark_sandwich_classification_evolution_v1`, data=`tf_rd_010_missingness_mcar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
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
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - The large validation pool follows the same hub bundle policy as the medium rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`, while offering the larger task set.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_large_v1_02_delta_data_manifest_root_tf_rd_010_missingness_mcar_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Sweep summary marked this row `stability=fail` at the 400-step short-run budget; the better BPC read did not clear the promotion guardrails.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_large_v1/delta_data_manifest_root_tf_rd_010_missingness_mcar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_large_v1_02_delta_data_manifest_root_tf_rd_010_missingness_mcar_v1` with final BPC `52508.1309`, delta final BPC `-62725.0640`, final BPF `52508.1296`, delta final BPF `-62725.0667`, final log loss `1.0877`, delta final log loss `+0.0094`, final Brier score `0.6593`, delta final Brier score `+0.0062`, best ROC AUC `0.4999`, final ROC AUC `0.4999`, final-minus-best `+0.0000`, delta final ROC AUC `-0.0397`, delta drift `+0.0000`, delta final training time `+0.2s`

### 3. `delta_data_manifest_root_tf_rd_010_missingness_mar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Add a structured missingness row so TF-RD-010 can distinguish random masking from observed-feature-linked masking under the larger benchmark contract.
- Hypothesis: MAR may provide a clearer harder front than MCAR while remaining more interpretable than MNAR on the large validation pool.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mar_v1`.
- Expected effect: Structured MAR may provide a harder but still interpretable missingness front for the first TF-RD-010 classification benchmark program.
- Effective labels: model=`cls_benchmark_sandwich_classification_evolution_v1`, data=`tf_rd_010_missingness_mar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
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
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - The large validation pool follows the same hub bundle policy as the medium rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`, while offering the larger task set.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_large_v1_03_delta_data_manifest_root_tf_rd_010_missingness_mar_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Sweep summary marked this row `stability=fail` at the 400-step short-run budget.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_large_v1/delta_data_manifest_root_tf_rd_010_missingness_mar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_large_v1_03_delta_data_manifest_root_tf_rd_010_missingness_mar_v1` with final BPC `79920.0386`, delta final BPC `-35313.1563`, final BPF `79920.0333`, delta final BPF `-35313.1629`, final log loss `1.0942`, delta final log loss `+0.0159`, final Brier score `0.6636`, delta final Brier score `+0.0105`, best ROC AUC `0.4953`, final ROC AUC `0.4953`, final-minus-best `+0.0000`, delta final ROC AUC `-0.0443`, delta drift `+0.0000`, delta final training time `-4.2s`

### 4. `delta_data_manifest_root_tf_rd_010_missingness_mnar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Keep one strongest missingness row in the first draft package so TF-RD-010 can compare MCAR, MAR, and MNAR under the same larger benchmark contract.
- Hypothesis: MNAR may be the hardest missingness front, but it may also be the least interpretable candidate for the first evolved benchmark package.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the evolved FiLM plus 3-summary-token sandwich contract fixed and replace the control corpus with `tf_rd_010_missingness_mnar_v1`.
- Expected effect: Structured MNAR may be the strongest synthetic missingness perturbation, but it risks a less interpretable first benchmark-evolution read than MCAR or MAR.
- Effective labels: model=`cls_benchmark_sandwich_classification_evolution_v1`, data=`tf_rd_010_missingness_mnar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
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
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - `dagzoo` owns this synthetic training front; `tab-realdata-hub` owns the validation manifest.
  - The large validation pool follows the same hub bundle policy as the medium rung: `min_classes=2`, `max_classes=10`, and `max_missing_pct=20.0`, while offering the larger task set.
  - Canonical rerun registered as `sd_tf_rd_010_classification_evolution_large_v1_04_delta_data_manifest_root_tf_rd_010_missingness_mnar_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Sweep summary marked this row `stability=fail` at the 400-step short-run budget.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_010_classification_evolution_large_v1/delta_data_manifest_root_tf_rd_010_missingness_mnar/result_card.md`
- Registered run: `sd_tf_rd_010_classification_evolution_large_v1_04_delta_data_manifest_root_tf_rd_010_missingness_mnar_v1` with final BPC `158968.8233`, delta final BPC `+43735.6283`, final BPF `158968.8286`, delta final BPF `+43735.6323`, final log loss `1.0890`, delta final log loss `+0.0107`, final Brier score `0.6602`, delta final Brier score `+0.0071`, best ROC AUC `0.4811`, final ROC AUC `0.4836`, final-minus-best `+87752.6605`, delta final ROC AUC `-0.0560`, delta drift `+87752.6605`, delta final training time `-14.7s`

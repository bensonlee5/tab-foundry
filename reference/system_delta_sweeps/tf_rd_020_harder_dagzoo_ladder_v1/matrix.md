# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_020_harder_dagzoo_ladder_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_020_harder_dagzoo_ladder_v1`
- Sweep status: `completed`
- Parent sweep id: `row_first_training_adequacy_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_row_first_training_adequacy_v1_01_delta_training_task_batch4_v1`
- Benchmark bundle: `data/manifests/bench/openml_classification_medium_v1/manifest.parquet`
- Control baseline id: `cls_benchmark_linear_v2`
- External benchmarks: `nanotabpfn`
- Training experiment: `cls_benchmark_staged_corpus`
- Training config profile: `cls_benchmark_staged_corpus`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `4.4473`, final Brier score `0.6465`, best ROC AUC `0.5958`, final ROC AUC `0.5746`, final training time `699.4s`

## Anchor Comparison

Upstream reference: `TabICLv2` from `https://arxiv.org/abs/2602.11139`.

| Dimension | Upstream TabICLv2 | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| promoted anchor | TabICLv2 motivates synthetic pretraining but does not define this repo-local row-first promoted-anchor contract. | The kept `row_cls + qass + no tfcol` anchor trained on the settled `task_batch_size=4` manifest-batched surface from TF-RD-018 issue `#109`. | TF-RD-020 should change only the synthetic corpus front while keeping the promoted row-first training recipe fixed, and interpret all completed rows as sharing the inherited `2500`-step schedule horizon on a `400`-step runtime. |
| harder dagzoo corpus front | No dedicated upstream reference defines this exact dagzoo harder-front ladder. | `tf_rd_013_dagzoo_shape_aware_size_medium_v1` is the carried medium control surface. | Rows nominate missingness, shift/drift, and mechanism-diversity or noise candidates that can become the next synthetic harder carry-forward front. |
| filtering sequence | Not applicable. | The carried medium control remains unfiltered. | Keep all issue `#147` rows pre-filter; later dagzoo filtering policy remains outside this sweep. |
| nomination rubric | Not applicable. | No harder-front nomination rubric existed before this sweep. | Rank rows by final log loss first, final Brier score second, final ROC AUC third, and best-to-final ROC delta fourth while treating runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior as guardrails. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_020_missingness_mcar` | missingness | yes | completed | none | Point training at the TF-RD-020 MCAR harder-front manifest while keeping the settled four-task row-first recipe fixed. | Keep as the TF-RD-020 missingness winner for issue `#148`; carry this synthetic-only harder-front evidence back into TF-RD-018 continuation alongside the kept shift or drift and mechanism or noise winners. |
| 2 | `delta_data_manifest_root_tf_rd_020_missingness_mar` | missingness | yes | completed | none | Point training at the TF-RD-020 MAR harder-front manifest while keeping the settled four-task row-first recipe fixed. | Defer behind the kept MCAR row; retain as the structured-missingness near miss for any later larger-corpus follow-up. |
| 3 | `delta_data_manifest_root_tf_rd_020_missingness_mnar` | missingness | yes | completed | none | Point training at the TF-RD-020 MNAR harder-front manifest while keeping the settled four-task row-first recipe fixed. | Defer behind the kept MCAR row; it remained the closest missingness runner-up but still lost on final log loss and final Brier score. |
| 4 | `delta_data_manifest_root_tf_rd_020_shift_graph_drift` | shift | yes | completed | none | Point training at the TF-RD-020 graph-drift harder-front manifest while keeping the settled four-task row-first recipe fixed. | Defer behind the kept noise-drift row; retain as the closest shift near miss because it lost the family decision on final log loss and final Brier score. |
| 5 | `delta_data_manifest_root_tf_rd_020_shift_mechanism_drift` | shift | yes | completed | none | Point training at the TF-RD-020 mechanism-drift harder-front manifest while keeping the settled four-task row-first recipe fixed. | Defer behind the kept noise-drift row; this row regressed after its best checkpoint and is not the preferred shift carry-forward front. |
| 6 | `delta_data_manifest_root_tf_rd_020_shift_noise_drift` | shift | yes | completed | none | Point training at the TF-RD-020 noise-drift harder-front manifest while keeping the settled four-task row-first recipe fixed. | Keep as the TF-RD-020 shift or drift winner for issue `#149`; carry this synthetic-only harder-front evidence back into TF-RD-018 continuation. |
| 7 | `delta_data_manifest_root_tf_rd_020_shift_mixed` | shift | yes | completed | none | Point training at the TF-RD-020 mixed-drift harder-front manifest while keeping the settled four-task row-first recipe fixed. | Defer behind the kept single-axis shift rows; the mixed drift surface lost on final log loss, final Brier score, and best-to-final ROC stability. |
| 8 | `delta_data_manifest_root_tf_rd_020_mechanism_piecewise` | mechanism_diversity | yes | completed | none | Point training at the TF-RD-020 piecewise-mechanism harder-front manifest while keeping the settled four-task row-first recipe fixed. | Defer behind the kept noise-mixture row; the piecewise control degraded sharply after its early best checkpoint. |
| 9 | `delta_data_manifest_root_tf_rd_020_mechanism_gp` | mechanism_diversity | yes | completed | none | Point training at the TF-RD-020 GP-only mechanism harder-front manifest while keeping the settled four-task row-first recipe fixed. | Defer behind the kept noise-mixture row; the GP-only mechanism mix did not recover enough final log loss or Brier score to stay competitive. |
| 10 | `delta_data_manifest_root_tf_rd_020_noise_laplace` | noise | yes | completed | none | Point training at the TF-RD-020 Laplace-noise harder-front manifest while keeping the settled four-task row-first recipe fixed. | Defer behind the kept noise-mixture row; this is the closest mechanism or noise runner-up, but it still loses the family decision on final log loss and final Brier score. |
| 11 | `delta_data_manifest_root_tf_rd_020_noise_mixture` | noise | yes | completed | none | Point training at the TF-RD-020 mixture-noise harder-front manifest while keeping the settled four-task row-first recipe fixed. | Keep as the TF-RD-020 mechanism or noise winner for issue `#150`; carry this synthetic-only harder-front evidence back into TF-RD-018 continuation. |

## Detailed Rows

### 1. `delta_data_manifest_root_tf_rd_020_missingness_mcar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the TF-RD-020 MCAR harder-front manifest while keeping the settled four-task row-first recipe fixed.
- Rationale: Open issue `#148` with the lowest-assumption missingness row so TF-RD-020 can test whether moderate MCAR alone is enough to create a harder synthetic front.
- Hypothesis: A `25%` MCAR variant may increase sample difficulty without introducing the label-correlated structure of MAR or MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic-data harder-front axis.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_missingness_mcar_v1`.
- Expected effect: Moderate MCAR may make the synthetic training surface harder without introducing mechanism-linked missingness structure.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_missingness_mcar`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0168`); context (grad `0.0213`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix before reading benchmark output.
  - Compare final log loss first, final Brier score second, final ROC AUC third, and best-to-final ROC delta fourth against the carried medium control and the other missingness rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#148` winner.
- Adequacy knobs to dimension explicitly:
  - explicit `missing_rate` and `missing_mechanism` resolution across the carried `8/28/4` invocation mix
  - manifest-contract deltas versus the kept TF-RD-013 medium control
  - benchmark-facing generalization under the settled four-task row-first recipe
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - This row remains pre-filter by design; do not reopen small-shot ease filter tuning here.
  - Supersedes historical queue run `sd_tf_rd_020_harder_dagzoo_ladder_v1_01_delta_data_manifest_root_tf_rd_020_missingness_mcar_v5`; that registry entry is retained as history only.
  - Supersedes historical queue run `sd_tf_rd_020_harder_dagzoo_ladder_v1_01_delta_data_manifest_root_tf_rd_020_missingness_mcar_v6`; that registry entry is retained as history only.
  - Canonical harmonized `400`-step rerun registered as `sd_tf_rd_020_harder_dagzoo_ladder_v1_01_delta_data_manifest_root_tf_rd_020_missingness_mcar_v7`.
  - This row keeps issue `#148` on final log loss, final Brier score, and final ROC AUC against the MAR and MNAR alternatives.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_020_harder_dagzoo_ladder_v1/delta_data_manifest_root_tf_rd_020_missingness_mcar/result_card.md`
- Registered run: `sd_tf_rd_020_harder_dagzoo_ladder_v1_01_delta_data_manifest_root_tf_rd_020_missingness_mcar_v7` with final log loss `0.5865`, delta final log loss `-3.8607`, final Brier score `0.4027`, delta final Brier score `-0.2438`, best ROC AUC `0.5642`, final ROC AUC `0.5642`, final-minus-best `+0.0000`, delta final ROC AUC `-0.0104`, delta drift `+0.0213`, delta final training time `+949.2s`

### 2. `delta_data_manifest_root_tf_rd_020_missingness_mar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the TF-RD-020 MAR harder-front manifest while keeping the settled four-task row-first recipe fixed.
- Rationale: Add one structured missingness row with resolved MAR logits so the missingness lane can distinguish random masking from observed-feature-driven null structure.
- Hypothesis: MAR at `missing_rate=0.25`, `missing_mar_observed_fraction=0.6`, and `missing_mar_logit_scale=1.4` may create a clearer harder front than MCAR without the full self-masking bias of MNAR.
- Upstream delta: Not applicable; this is a repo-local synthetic-data harder-front axis.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_missingness_mar_v1`.
- Expected effect: Structured MAR missingness may create a clearer harder front than MCAR without the stronger self-masking bias of MNAR.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_missingness_mar`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0248`); context (grad `0.0407`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the MAR override fields in the per-invocation provenance.
  - Compare final log loss first, final Brier score second, final ROC AUC third, and best-to-final ROC delta fourth against the carried medium control and the other missingness rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#148` winner.
- Adequacy knobs to dimension explicitly:
  - explicit MAR override resolution across the carried `8/28/4` invocation mix
  - manifest-contract deltas versus the kept TF-RD-013 medium control
  - benchmark-facing generalization under the settled four-task row-first recipe
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row remains pre-filter by design; do not reopen small-shot ease filter tuning here.
  - Supersedes historical queue run `sd_tf_rd_020_harder_dagzoo_ladder_v1_02_delta_data_manifest_root_tf_rd_020_missingness_mar_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_020_harder_dagzoo_ladder_v1_02_delta_data_manifest_root_tf_rd_020_missingness_mar_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_020_harder_dagzoo_ladder_v1/delta_data_manifest_root_tf_rd_020_missingness_mar/result_card.md`
- Registered run: `sd_tf_rd_020_harder_dagzoo_ladder_v1_02_delta_data_manifest_root_tf_rd_020_missingness_mar_v2` with final log loss `0.5978`, delta final log loss `-3.8495`, final Brier score `0.4122`, delta final Brier score `-0.2343`, best ROC AUC `0.5298`, final ROC AUC `0.5298`, final-minus-best `+0.0000`, delta final ROC AUC `-0.0447`, delta drift `+0.0213`, delta final training time `+277.2s`

### 3. `delta_data_manifest_root_tf_rd_020_missingness_mnar`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the TF-RD-020 MNAR harder-front manifest while keeping the settled four-task row-first recipe fixed.
- Rationale: Close the missingness lane with one self-masking row so TF-RD-020 can read whether the harder front comes from mechanism structure rather than rate alone.
- Hypothesis: MNAR at `missing_rate=0.25` and `missing_mnar_logit_scale=1.6` may be the strongest missingness harder front, but it also risks producing a more confounded synthetic lane than MCAR or MAR.
- Upstream delta: Not applicable; this is a repo-local synthetic-data harder-front axis.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_missingness_mnar_v1`.
- Expected effect: Structured MNAR missingness may be the strongest missingness harder front, but it risks a less interpretable synthetic lane than MCAR or MAR.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_missingness_mnar`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0202`); context (grad `0.0234`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the MNAR override fields in the per-invocation provenance.
  - Compare final log loss first, final Brier score second, final ROC AUC third, and best-to-final ROC delta fourth against the carried medium control and the other missingness rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#148` winner.
- Adequacy knobs to dimension explicitly:
  - explicit MNAR override resolution across the carried `8/28/4` invocation mix
  - manifest-contract deltas versus the kept TF-RD-013 medium control
  - benchmark-facing generalization under the settled four-task row-first recipe
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row remains pre-filter by design; do not reopen small-shot ease filter tuning here.
  - Supersedes historical queue run `sd_tf_rd_020_harder_dagzoo_ladder_v1_03_delta_data_manifest_root_tf_rd_020_missingness_mnar_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_020_harder_dagzoo_ladder_v1_03_delta_data_manifest_root_tf_rd_020_missingness_mnar_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_020_harder_dagzoo_ladder_v1/delta_data_manifest_root_tf_rd_020_missingness_mnar/result_card.md`
- Registered run: `sd_tf_rd_020_harder_dagzoo_ladder_v1_03_delta_data_manifest_root_tf_rd_020_missingness_mnar_v2` with final log loss `0.5958`, delta final log loss `-3.8514`, final Brier score `0.4102`, delta final Brier score `-0.2363`, best ROC AUC `0.5551`, final ROC AUC `0.5551`, final-minus-best `+0.0000`, delta final ROC AUC `-0.0195`, delta drift `+0.0213`, delta final training time `+641.4s`

### 4. `delta_data_manifest_root_tf_rd_020_shift_graph_drift`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the TF-RD-020 graph-drift harder-front manifest while keeping the settled four-task row-first recipe fixed.
- Rationale: Open issue `#149` with the most structurally local shift row so TF-RD-020 can test whether moderate graph drift alone is enough to create a clearer harder front.
- Hypothesis: `graph_drift` at `graph_scale=0.5` may increase structural mismatch without immediately changing mechanism or noise complexity.
- Upstream delta: Not applicable; this is a repo-local synthetic-data harder-front axis.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_shift_graph_drift_v1`.
- Expected effect: Moderate graph drift may create a harder synthetic front without simultaneously changing mechanism or noise structure.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_graph_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0438`); context (grad `0.0729`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves `shift.mode=graph_drift` with `graph_scale=0.5` in effective config artifacts.
  - Compare final log loss first, final Brier score second, final ROC AUC third, and best-to-final ROC delta fourth against the carried medium control and the other shift rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#149` winner.
- Adequacy knobs to dimension explicitly:
  - explicit `shift.mode=graph_drift` resolution with `graph_scale=0.5` across the carried `8/28/4` invocation mix
  - manifest-contract deltas versus the kept TF-RD-013 medium control
  - benchmark-facing generalization under the settled four-task row-first recipe
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row remains pre-filter by design; do not reopen small-shot ease filter tuning here.
  - Supersedes historical queue run `sd_tf_rd_020_harder_dagzoo_ladder_v1_04_delta_data_manifest_root_tf_rd_020_shift_graph_drift_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_020_harder_dagzoo_ladder_v1_04_delta_data_manifest_root_tf_rd_020_shift_graph_drift_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_020_harder_dagzoo_ladder_v1/delta_data_manifest_root_tf_rd_020_shift_graph_drift/result_card.md`
- Registered run: `sd_tf_rd_020_harder_dagzoo_ladder_v1_04_delta_data_manifest_root_tf_rd_020_shift_graph_drift_v2` with final log loss `0.5551`, delta final log loss `-3.8921`, final Brier score `0.3751`, delta final Brier score `-0.2714`, best ROC AUC `0.6029`, final ROC AUC `0.6029`, final-minus-best `+0.0000`, delta final ROC AUC `+0.0283`, delta drift `+0.0213`, delta final training time `+873.1s`

### 5. `delta_data_manifest_root_tf_rd_020_shift_mechanism_drift`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the TF-RD-020 mechanism-drift harder-front manifest while keeping the settled four-task row-first recipe fixed.
- Rationale: Add the mechanism-tilt shift row so the shift lane can distinguish graph-structure changes from controlled increases in nonlinear mechanism mass.
- Hypothesis: `mechanism_drift` at `mechanism_scale=0.5` may create a stronger harder front than pure graph drift because it directly perturbs function-family mix between train and test.
- Upstream delta: Not applicable; this is a repo-local synthetic-data harder-front axis.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_shift_mechanism_drift_v1`.
- Expected effect: Moderate mechanism drift may create a clearer harder front than pure graph drift by changing function-family mass between train and test.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_mechanism_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0885`); context (grad `0.1328`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves `shift.mode=mechanism_drift` with `mechanism_scale=0.5` in effective config artifacts.
  - Compare final log loss first, final Brier score second, final ROC AUC third, and best-to-final ROC delta fourth against the carried medium control and the other shift rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#149` winner.
- Adequacy knobs to dimension explicitly:
  - explicit `shift.mode=mechanism_drift` resolution with `mechanism_scale=0.5` across the carried `8/28/4` invocation mix
  - manifest-contract deltas versus the kept TF-RD-013 medium control
  - benchmark-facing generalization under the settled four-task row-first recipe
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row remains pre-filter by design; do not reopen small-shot ease filter tuning here.
  - Supersedes historical queue run `sd_tf_rd_020_harder_dagzoo_ladder_v1_05_delta_data_manifest_root_tf_rd_020_shift_mechanism_drift_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_020_harder_dagzoo_ladder_v1_05_delta_data_manifest_root_tf_rd_020_shift_mechanism_drift_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_020_harder_dagzoo_ladder_v1/delta_data_manifest_root_tf_rd_020_shift_mechanism_drift/result_card.md`
- Registered run: `sd_tf_rd_020_harder_dagzoo_ladder_v1_05_delta_data_manifest_root_tf_rd_020_shift_mechanism_drift_v2` with final log loss `0.6387`, delta final log loss `-3.8086`, final Brier score `0.4214`, delta final Brier score `-0.2251`, best ROC AUC `0.6241`, final ROC AUC `0.5997`, final-minus-best `-0.0244`, delta final ROC AUC `+0.0251`, delta drift `-0.0031`, delta final training time `+1087.7s`

### 6. `delta_data_manifest_root_tf_rd_020_shift_noise_drift`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the TF-RD-020 noise-drift harder-front manifest while keeping the settled four-task row-first recipe fixed.
- Rationale: Add the variance-only shift row so the shift lane can isolate test-time noise inflation from graph or mechanism drift.
- Hypothesis: `noise_drift` at `variance_scale=0.5` may create a harder front if the current overfitting problem is partly a mismatch in stochasticity rather than structure.
- Upstream delta: Not applicable; this is a repo-local synthetic-data harder-front axis.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_shift_noise_drift_v1`.
- Expected effect: Moderate variance drift may create a harder front if the current overfitting problem is partly a mismatch in stochasticity rather than structure.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_noise_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0685`); context (grad `0.1265`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves `shift.mode=noise_drift` with `variance_scale=0.5` in effective config artifacts.
  - Compare final log loss first, final Brier score second, final ROC AUC third, and best-to-final ROC delta fourth against the carried medium control and the other shift rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#149` winner.
- Adequacy knobs to dimension explicitly:
  - explicit `shift.mode=noise_drift` resolution with `variance_scale=0.5` across the carried `8/28/4` invocation mix
  - manifest-contract deltas versus the kept TF-RD-013 medium control
  - benchmark-facing generalization under the settled four-task row-first recipe
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - This row remains pre-filter by design; do not reopen small-shot ease filter tuning here.
  - Supersedes historical queue run `sd_tf_rd_020_harder_dagzoo_ladder_v1_06_delta_data_manifest_root_tf_rd_020_shift_noise_drift_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_020_harder_dagzoo_ladder_v1_06_delta_data_manifest_root_tf_rd_020_shift_noise_drift_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_020_harder_dagzoo_ladder_v1/delta_data_manifest_root_tf_rd_020_shift_noise_drift/result_card.md`
- Registered run: `sd_tf_rd_020_harder_dagzoo_ladder_v1_06_delta_data_manifest_root_tf_rd_020_shift_noise_drift_v2` with final log loss `0.5501`, delta final log loss `-3.8972`, final Brier score `0.3740`, delta final Brier score `-0.2724`, best ROC AUC `0.5945`, final ROC AUC `0.5880`, final-minus-best `-0.0065`, delta final ROC AUC `+0.0134`, delta drift `+0.0147`, delta final training time `+528.7s`

### 7. `delta_data_manifest_root_tf_rd_020_shift_mixed`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the TF-RD-020 mixed-drift harder-front manifest while keeping the settled four-task row-first recipe fixed.
- Rationale: Close the shift lane with the combined drift row so TF-RD-020 can test whether one mixed surface is the clearest synthetic harder front before moving on to mechanism and noise comparisons.
- Hypothesis: `mixed` drift at `graph_scale=0.5`, `mechanism_scale=0.5`, and `variance_scale=0.5` may be the strongest shift candidate, but it risks being harder to interpret than the single-axis rows.
- Upstream delta: Not applicable; this is a repo-local synthetic-data harder-front axis.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_shift_mixed_v1`.
- Expected effect: Mixed drift may be the strongest shift candidate, but it also risks becoming less interpretable than the single-axis rows.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_mixed`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0751`); context (grad `0.1044`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves `shift.mode=mixed` with all three `0.5` drift scales in effective config artifacts.
  - Compare final log loss first, final Brier score second, final ROC AUC third, and best-to-final ROC delta fourth against the carried medium control and the other shift rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#149` winner.
- Adequacy knobs to dimension explicitly:
  - explicit `shift.mode=mixed` resolution with all three `0.5` scales across the carried `8/28/4` invocation mix
  - manifest-contract deltas versus the kept TF-RD-013 medium control
  - benchmark-facing generalization under the settled four-task row-first recipe
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row remains pre-filter by design; do not reopen small-shot ease filter tuning here.
  - Supersedes historical queue run `sd_tf_rd_020_harder_dagzoo_ladder_v1_07_delta_data_manifest_root_tf_rd_020_shift_mixed_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_020_harder_dagzoo_ladder_v1_07_delta_data_manifest_root_tf_rd_020_shift_mixed_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_020_harder_dagzoo_ladder_v1/delta_data_manifest_root_tf_rd_020_shift_mixed/result_card.md`
- Registered run: `sd_tf_rd_020_harder_dagzoo_ladder_v1_07_delta_data_manifest_root_tf_rd_020_shift_mixed_v2` with final log loss `0.6795`, delta final log loss `-3.7677`, final Brier score `0.4371`, delta final Brier score `-0.2094`, best ROC AUC `0.6152`, final ROC AUC `0.5350`, final-minus-best `-0.0802`, delta final ROC AUC `-0.0396`, delta drift `-0.0590`, delta final training time `+996.7s`

### 8. `delta_data_manifest_root_tf_rd_020_mechanism_piecewise`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the TF-RD-020 piecewise-mechanism harder-front manifest while keeping the settled four-task row-first recipe fixed.
- Rationale: Open issue `#150` with the shipped piecewise control so TF-RD-020 can test whether broader mechanism family structure is a cleaner harder front than explicit train/test drift.
- Hypothesis: A `piecewise=0.3`, `linear=0.7` family mix may create a more data-hungry synthetic front without changing evaluation-time shift assumptions.
- Upstream delta: Not applicable; this is a repo-local synthetic-data harder-front axis.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_mechanism_piecewise_v1`.
- Expected effect: The shipped piecewise-plus-linear family mix may create a more data-hungry synthetic front without introducing explicit train-test shift.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_mechanism_piecewise`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0629`); context (grad `0.0621`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the piecewise-plus-linear family mix in the per-invocation effective config artifacts.
  - Compare final log loss first, final Brier score second, final ROC AUC third, and best-to-final ROC delta fourth against the carried medium control and the other mechanism or noise rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#150` winner.
- Adequacy knobs to dimension explicitly:
  - explicit mechanism family-mix resolution across the carried `8/28/4` invocation mix
  - manifest-contract deltas versus the kept TF-RD-013 medium control
  - benchmark-facing generalization under the settled four-task row-first recipe
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row remains pre-filter by design; do not reopen small-shot ease filter tuning here.
  - Supersedes historical queue run `sd_tf_rd_020_harder_dagzoo_ladder_v1_08_delta_data_manifest_root_tf_rd_020_mechanism_piecewise_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_020_harder_dagzoo_ladder_v1_08_delta_data_manifest_root_tf_rd_020_mechanism_piecewise_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_020_harder_dagzoo_ladder_v1/delta_data_manifest_root_tf_rd_020_mechanism_piecewise/result_card.md`
- Registered run: `sd_tf_rd_020_harder_dagzoo_ladder_v1_08_delta_data_manifest_root_tf_rd_020_mechanism_piecewise_v2` with final log loss `0.6272`, delta final log loss `-3.8201`, final Brier score `0.4252`, delta final Brier score `-0.2213`, best ROC AUC `0.5584`, final ROC AUC `0.4705`, final-minus-best `-0.0880`, delta final ROC AUC `-0.1041`, delta drift `-0.0667`, delta final training time `+1085.5s`

### 9. `delta_data_manifest_root_tf_rd_020_mechanism_gp`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the TF-RD-020 GP-only mechanism harder-front manifest while keeping the settled four-task row-first recipe fixed.
- Rationale: Add the GP-only mechanism row so the mechanism lane can distinguish the shipped piecewise control from the widened GP family path.
- Hypothesis: `gp=1.0` may create a more sample-hungry but still interpretable harder front if the current corpus is too easy because it under-exercises smoother nonlinear families.
- Upstream delta: Not applicable; this is a repo-local synthetic-data harder-front axis.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_mechanism_gp_v1`.
- Expected effect: The widened GP family may create a more sample-hungry but still interpretable harder front if the current corpus is too easy.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_mechanism_gp`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0226`); context (grad `0.0346`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the GP-only family mix in the per-invocation effective config artifacts.
  - Compare final log loss first, final Brier score second, final ROC AUC third, and best-to-final ROC delta fourth against the carried medium control and the other mechanism or noise rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#150` winner.
- Adequacy knobs to dimension explicitly:
  - explicit GP-only family-mix resolution across the carried `8/28/4` invocation mix
  - manifest-contract deltas versus the kept TF-RD-013 medium control
  - benchmark-facing generalization under the settled four-task row-first recipe
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row remains pre-filter by design; do not reopen small-shot ease filter tuning here.
  - Supersedes historical queue run `sd_tf_rd_020_harder_dagzoo_ladder_v1_09_delta_data_manifest_root_tf_rd_020_mechanism_gp_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_020_harder_dagzoo_ladder_v1_09_delta_data_manifest_root_tf_rd_020_mechanism_gp_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_020_harder_dagzoo_ladder_v1/delta_data_manifest_root_tf_rd_020_mechanism_gp/result_card.md`
- Registered run: `sd_tf_rd_020_harder_dagzoo_ladder_v1_09_delta_data_manifest_root_tf_rd_020_mechanism_gp_v2` with final log loss `0.6272`, delta final log loss `-3.8201`, final Brier score `0.4302`, delta final Brier score `-0.2163`, best ROC AUC `0.4511`, final ROC AUC `0.4566`, final-minus-best `+0.0055`, delta final ROC AUC `-0.1180`, delta drift `+0.0267`, delta final training time `+944.7s`

### 10. `delta_data_manifest_root_tf_rd_020_noise_laplace`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the TF-RD-020 Laplace-noise harder-front manifest while keeping the settled four-task row-first recipe fixed.
- Rationale: Add the simplest non-Gaussian noise row so the mechanism or noise lane can test whether heavier tails alone create a clearer harder front than function-family changes.
- Hypothesis: Laplace noise at `base_scale=1.0` may increase stochastic difficulty without the broader regime ambiguity of the mixture row.
- Upstream delta: Not applicable; this is a repo-local synthetic-data harder-front axis.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_noise_laplace_v1`.
- Expected effect: Laplace noise may create a harder front through heavier tails without the broader ambiguity of the mixture-noise row.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_noise_laplace`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0232`); context (grad `0.0435`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the Laplace noise family in the per-invocation effective config artifacts.
  - Compare final log loss first, final Brier score second, final ROC AUC third, and best-to-final ROC delta fourth against the carried medium control and the other mechanism or noise rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#150` winner.
- Adequacy knobs to dimension explicitly:
  - explicit Laplace-noise resolution across the carried `8/28/4` invocation mix
  - manifest-contract deltas versus the kept TF-RD-013 medium control
  - benchmark-facing generalization under the settled four-task row-first recipe
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row remains pre-filter by design; do not reopen small-shot ease filter tuning here.
  - Supersedes historical queue run `sd_tf_rd_020_harder_dagzoo_ladder_v1_10_delta_data_manifest_root_tf_rd_020_noise_laplace_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_020_harder_dagzoo_ladder_v1_10_delta_data_manifest_root_tf_rd_020_noise_laplace_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_020_harder_dagzoo_ladder_v1/delta_data_manifest_root_tf_rd_020_noise_laplace/result_card.md`
- Registered run: `sd_tf_rd_020_harder_dagzoo_ladder_v1_10_delta_data_manifest_root_tf_rd_020_noise_laplace_v2` with final log loss `0.5959`, delta final log loss `-3.8514`, final Brier score `0.4113`, delta final Brier score `-0.2352`, best ROC AUC `0.5599`, final ROC AUC `0.5618`, final-minus-best `+0.0019`, delta final ROC AUC `-0.0127`, delta drift `+0.0232`, delta final training time `+944.4s`

### 11. `delta_data_manifest_root_tf_rd_020_noise_mixture`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the TF-RD-020 mixture-noise harder-front manifest while keeping the settled four-task row-first recipe fixed.
- Rationale: Close issue `#150` with the heaviest-tailed candidate so TF-RD-020 can test whether one mixed noise surface is the clearest non-mechanism harder front.
- Hypothesis: Mixture noise with `gaussian=0.5`, `laplace=0.3`, `student_t=0.2`, and `student_t_df=6.0` may be the strongest noise candidate, but it risks becoming a noisier and less interpretable front than the mechanism rows.
- Upstream delta: Not applicable; this is a repo-local synthetic-data harder-front axis.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_noise_mixture_v1`.
- Expected effect: Mixture noise may be the strongest heavier-tail noise candidate, but it also risks becoming noisier and less interpretable than the mechanism rows.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_noise_mixture`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.1773`); context (grad `0.3327`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the mixture-noise family plus mixture weights in the per-invocation effective config artifacts.
  - Compare final log loss first, final Brier score second, final ROC AUC third, and best-to-final ROC delta fourth against the carried medium control and the other mechanism or noise rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#150` winner.
- Adequacy knobs to dimension explicitly:
  - explicit mixture-noise resolution across the carried `8/28/4` invocation mix
  - manifest-contract deltas versus the kept TF-RD-013 medium control
  - benchmark-facing generalization under the settled four-task row-first recipe
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - This row remains pre-filter by design; do not reopen small-shot ease filter tuning here.
  - Supersedes historical queue run `sd_tf_rd_020_harder_dagzoo_ladder_v1_11_delta_data_manifest_root_tf_rd_020_noise_mixture_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tf_rd_020_harder_dagzoo_ladder_v1_11_delta_data_manifest_root_tf_rd_020_noise_mixture_v2`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_020_harder_dagzoo_ladder_v1/delta_data_manifest_root_tf_rd_020_noise_mixture/result_card.md`
- Registered run: `sd_tf_rd_020_harder_dagzoo_ladder_v1_11_delta_data_manifest_root_tf_rd_020_noise_mixture_v2` with final log loss `0.5737`, delta final log loss `-3.8736`, final Brier score `0.3917`, delta final Brier score `-0.2548`, best ROC AUC `0.5934`, final ROC AUC `0.5904`, final-minus-best `-0.0030`, delta final ROC AUC `+0.0159`, delta drift `+0.0183`, delta final training time `+700.4s`

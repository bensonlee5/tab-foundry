# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_020_harder_dagzoo_ladder_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_020_harder_dagzoo_ladder_v1`
- Sweep status: `draft`
- Parent sweep id: `row_first_training_adequacy_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_row_first_training_adequacy_v1_01_delta_training_task_batch4_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Training experiment: `cls_benchmark_staged_corpus`
- Training config profile: `cls_benchmark_staged_corpus`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `4.4473`, final Brier score `0.6465`, best ROC AUC `0.5958`, final ROC AUC `0.5746`, final training time `699.4s`

## Anchor Comparison

Upstream reference: `TabICLv2` from `https://arxiv.org/abs/2602.11139`.

| Dimension | Upstream TabICLv2 | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| promoted anchor | TabICLv2 motivates synthetic pretraining but does not define this repo-local row-first promoted-anchor contract. | The kept `row_cls + qass + no tfcol` anchor trained on the settled `task_batch_size=4` manifest-batched surface from TF-RD-018 issue `#109`. | TF-RD-020 changes only the synthetic corpus front while keeping the promoted row-first recipe fixed. |
| harder dagzoo corpus front | No dedicated upstream reference defines this exact dagzoo harder-front ladder. | `tf_rd_013_dagzoo_shape_aware_size_medium_v1` remains the carried medium control surface. | Rows nominate missingness, shift or drift, and mechanism-diversity or noise candidates that can become the next synthetic harder carry-forward front. |
| filtering sequence | Not applicable. | The carried medium control remains unfiltered. | Keep all issue `#147` rows pre-filter and defer the final small-shot ease filter decision to issue `#151`. |
| nomination rubric | Not applicable. | No harder-front nomination rubric existed before this sweep. | Rank rows by final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third while treating runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior as guardrails. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_020_missingness_mcar` | missingness | yes | ready | `tf_rd_020_missingness_mcar_v1` | Replace the carried medium corpus with the `25%` MCAR front while keeping the settled four-task row-first recipe fixed. | Run first under issue `#148`, then compare directly against the MAR and MNAR rows before nominating one missingness winner. |
| 2 | `delta_data_manifest_root_tf_rd_020_missingness_mar` | missingness | yes | ready | `tf_rd_020_missingness_mar_v1` | Replace the carried medium corpus with the MAR front at `missing_rate=0.25`, `missing_mar_observed_fraction=0.6`, and `missing_mar_logit_scale=1.4`. | Run second under issue `#148`, then compare directly against the MCAR and MNAR rows before nominating one missingness winner. |
| 3 | `delta_data_manifest_root_tf_rd_020_missingness_mnar` | missingness | yes | ready | `tf_rd_020_missingness_mnar_v1` | Replace the carried medium corpus with the MNAR front at `missing_rate=0.25` and `missing_mnar_logit_scale=1.6`. | Run third under issue `#148`, then select exactly one missingness nominee for cross-front comparison. |
| 4 | `delta_data_manifest_root_tf_rd_020_shift_graph_drift` | shift | yes | ready | `tf_rd_020_shift_graph_drift_v1` | Replace the carried medium corpus with the graph-drift front at `graph_scale=0.5`. | Run first under issue `#149`, then compare directly against the mechanism-drift, noise-drift, and mixed rows before nominating one shift winner. |
| 5 | `delta_data_manifest_root_tf_rd_020_shift_mechanism_drift` | shift | yes | ready | `tf_rd_020_shift_mechanism_drift_v1` | Replace the carried medium corpus with the mechanism-drift front at `mechanism_scale=0.5`. | Run second under issue `#149`, then compare directly against the graph-drift, noise-drift, and mixed rows before nominating one shift winner. |
| 6 | `delta_data_manifest_root_tf_rd_020_shift_noise_drift` | shift | yes | ready | `tf_rd_020_shift_noise_drift_v1` | Replace the carried medium corpus with the noise-drift front at `variance_scale=0.5`. | Run third under issue `#149`, then compare directly against the graph-drift, mechanism-drift, and mixed rows before nominating one shift winner. |
| 7 | `delta_data_manifest_root_tf_rd_020_shift_mixed` | shift | yes | ready | `tf_rd_020_shift_mixed_v1` | Replace the carried medium corpus with the mixed-drift front at `graph_scale=0.5`, `mechanism_scale=0.5`, and `variance_scale=0.5`. | Run fourth under issue `#149`, then select exactly one shift or drift nominee for cross-front comparison. |
| 8 | `delta_data_manifest_root_tf_rd_020_mechanism_piecewise` | mechanism_diversity | yes | ready | `tf_rd_020_mechanism_piecewise_v1` | Replace the carried medium corpus with the piecewise-plus-linear mechanism front. | Run first under issue `#150`, then compare directly against the GP-only and heavier-tail noise rows before nominating one mechanism or noise winner. |
| 9 | `delta_data_manifest_root_tf_rd_020_mechanism_gp` | mechanism_diversity | yes | ready | `tf_rd_020_mechanism_gp_v1` | Replace the carried medium corpus with the GP-only mechanism front. | Run second under issue `#150`, then compare directly against the piecewise and heavier-tail noise rows before nominating one mechanism or noise winner. |
| 10 | `delta_data_manifest_root_tf_rd_020_noise_laplace` | noise | yes | ready | `tf_rd_020_noise_laplace_v1` | Replace the carried medium corpus with the Laplace-noise front at `base_scale=1.0`. | Run third under issue `#150`, then compare directly against the mechanism rows and mixture noise row before nominating one mechanism or noise winner. |
| 11 | `delta_data_manifest_root_tf_rd_020_noise_mixture` | noise | yes | ready | `tf_rd_020_noise_mixture_v1` | Replace the carried medium corpus with the mixture-noise front at `base_scale=1.0`, `student_t_df=6.0`, and fixed Gaussian/Laplace/Student-t weights. | Run fourth under issue `#150`, then select exactly one mechanism-diversity or noise nominee for cross-front comparison. |

## Nomination Rules

- Issues `#148`, `#149`, and `#150` each nominate exactly one pre-filter winner from their own rows.
- The single cross-front winner becomes the selected harder front for issue `#151`.
- Rank rows by benchmark-facing final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third.
- Treat runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior as guardrails rather than primary selection metrics.
- Keep every row pre-filter; issue `#151` is the first place where the dagzoo small-shot ease filter re-enters the TF-RD-020 lane.

## Detailed Rows

### 1. `delta_data_manifest_root_tf_rd_020_missingness_mcar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `tf_rd_020_missingness_mcar_v1`
- Description: Replace the carried medium corpus with the `25%` MCAR front while keeping the settled four-task row-first recipe fixed.
- Rationale: Open issue `#148` with the lowest-assumption missingness row so TF-RD-020 can test whether moderate MCAR alone is enough to create a harder synthetic front.
- Hypothesis: A `25%` MCAR variant may increase sample difficulty without introducing the label-correlated structure of MAR or MNAR.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_missingness_mcar_v1`.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_missingness_mcar`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix before reading benchmark output.
  - Compare final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third against the carried medium control and the other missingness rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#148` winner.
- Interpretation status: `pending`
- Decision: `None`
- Benchmark metrics: pending

### 2. `delta_data_manifest_root_tf_rd_020_missingness_mar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `tf_rd_020_missingness_mar_v1`
- Description: Replace the carried medium corpus with the MAR front at `missing_rate=0.25`, `missing_mar_observed_fraction=0.6`, and `missing_mar_logit_scale=1.4`.
- Rationale: Add one structured missingness row with resolved MAR logits so the missingness lane can distinguish random masking from observed-feature-driven null structure.
- Hypothesis: MAR at `missing_rate=0.25`, `missing_mar_observed_fraction=0.6`, and `missing_mar_logit_scale=1.4` may create a clearer harder front than MCAR without the full self-masking bias of MNAR.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_missingness_mar_v1`.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_missingness_mar`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the MAR override fields in the per-invocation provenance.
  - Compare final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third against the carried medium control and the other missingness rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#148` winner.
- Interpretation status: `pending`
- Decision: `None`
- Benchmark metrics: pending

### 3. `delta_data_manifest_root_tf_rd_020_missingness_mnar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `tf_rd_020_missingness_mnar_v1`
- Description: Replace the carried medium corpus with the MNAR front at `missing_rate=0.25` and `missing_mnar_logit_scale=1.6`.
- Rationale: Close the missingness lane with one self-masking row so TF-RD-020 can read whether the harder front comes from mechanism structure rather than rate alone.
- Hypothesis: MNAR at `missing_rate=0.25` and `missing_mnar_logit_scale=1.6` may be the strongest missingness harder front, but it also risks producing a more confounded synthetic lane than MCAR or MAR.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_missingness_mnar_v1`.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_missingness_mnar`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the MNAR override fields in the per-invocation provenance.
  - Compare final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third against the carried medium control and the other missingness rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#148` winner.
- Interpretation status: `pending`
- Decision: `None`
- Benchmark metrics: pending

### 4. `delta_data_manifest_root_tf_rd_020_shift_graph_drift`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `tf_rd_020_shift_graph_drift_v1`
- Description: Replace the carried medium corpus with the graph-drift front at `graph_scale=0.5`.
- Rationale: Open issue `#149` with the most structurally local shift row so TF-RD-020 can test whether moderate graph drift alone is enough to create a clearer harder front.
- Hypothesis: `graph_drift` at `graph_scale=0.5` may increase structural mismatch without immediately changing mechanism or noise complexity.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_shift_graph_drift_v1`.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_graph_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves `shift.mode=graph_drift` with `graph_scale=0.5` in effective config artifacts.
  - Compare final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third against the carried medium control and the other shift rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#149` winner.
- Interpretation status: `pending`
- Decision: `None`
- Benchmark metrics: pending

### 5. `delta_data_manifest_root_tf_rd_020_shift_mechanism_drift`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `tf_rd_020_shift_mechanism_drift_v1`
- Description: Replace the carried medium corpus with the mechanism-drift front at `mechanism_scale=0.5`.
- Rationale: Add the mechanism-tilt shift row so the shift lane can distinguish graph-structure changes from controlled increases in nonlinear mechanism mass.
- Hypothesis: `mechanism_drift` at `mechanism_scale=0.5` may create a stronger harder front than pure graph drift because it directly perturbs function-family mix between train and test.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_shift_mechanism_drift_v1`.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_mechanism_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves `shift.mode=mechanism_drift` with `mechanism_scale=0.5` in effective config artifacts.
  - Compare final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third against the carried medium control and the other shift rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#149` winner.
- Interpretation status: `pending`
- Decision: `None`
- Benchmark metrics: pending

### 6. `delta_data_manifest_root_tf_rd_020_shift_noise_drift`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `tf_rd_020_shift_noise_drift_v1`
- Description: Replace the carried medium corpus with the noise-drift front at `variance_scale=0.5`.
- Rationale: Add the variance-only shift row so the shift lane can isolate test-time noise inflation from graph or mechanism drift.
- Hypothesis: `noise_drift` at `variance_scale=0.5` may create a harder front if the current overfitting problem is partly a mismatch in stochasticity rather than structure.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_shift_noise_drift_v1`.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_noise_drift`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves `shift.mode=noise_drift` with `variance_scale=0.5` in effective config artifacts.
  - Compare final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third against the carried medium control and the other shift rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#149` winner.
- Interpretation status: `pending`
- Decision: `None`
- Benchmark metrics: pending

### 7. `delta_data_manifest_root_tf_rd_020_shift_mixed`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `tf_rd_020_shift_mixed_v1`
- Description: Replace the carried medium corpus with the mixed-drift front at `graph_scale=0.5`, `mechanism_scale=0.5`, and `variance_scale=0.5`.
- Rationale: Close the shift lane with the combined drift row so TF-RD-020 can test whether one mixed surface is the clearest synthetic harder front before moving on to mechanism and noise comparisons.
- Hypothesis: `mixed` drift at `graph_scale=0.5`, `mechanism_scale=0.5`, and `variance_scale=0.5` may be the strongest shift candidate, but it risks being harder to interpret than the single-axis rows.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_shift_mixed_v1`.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_shift_mixed`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves `shift.mode=mixed` with all three `0.5` drift scales in effective config artifacts.
  - Compare final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third against the carried medium control and the other shift rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#149` winner.
- Interpretation status: `pending`
- Decision: `None`
- Benchmark metrics: pending

### 8. `delta_data_manifest_root_tf_rd_020_mechanism_piecewise`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `tf_rd_020_mechanism_piecewise_v1`
- Description: Replace the carried medium corpus with the piecewise-plus-linear mechanism front.
- Rationale: Open issue `#150` with the shipped piecewise control so TF-RD-020 can test whether broader mechanism family structure is a cleaner harder front than explicit train/test drift.
- Hypothesis: A `piecewise=0.3`, `linear=0.7` family mix may create a more data-hungry synthetic front without changing evaluation-time shift assumptions.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_mechanism_piecewise_v1`.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_mechanism_piecewise`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the piecewise-plus-linear family mix in the per-invocation effective config artifacts.
  - Compare final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third against the carried medium control and the other mechanism or noise rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#150` winner.
- Interpretation status: `pending`
- Decision: `None`
- Benchmark metrics: pending

### 9. `delta_data_manifest_root_tf_rd_020_mechanism_gp`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `tf_rd_020_mechanism_gp_v1`
- Description: Replace the carried medium corpus with the GP-only mechanism front.
- Rationale: Add the GP-only mechanism row so the mechanism lane can distinguish the shipped piecewise control from the widened GP family path.
- Hypothesis: `gp=1.0` may create a more sample-hungry but still interpretable harder front if the current corpus is too easy because it under-exercises smoother nonlinear families.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_mechanism_gp_v1`.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_mechanism_gp`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the GP-only family mix in the per-invocation effective config artifacts.
  - Compare final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third against the carried medium control and the other mechanism or noise rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#150` winner.
- Interpretation status: `pending`
- Decision: `None`
- Benchmark metrics: pending

### 10. `delta_data_manifest_root_tf_rd_020_noise_laplace`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `tf_rd_020_noise_laplace_v1`
- Description: Replace the carried medium corpus with the Laplace-noise front at `base_scale=1.0`.
- Rationale: Add the simplest non-Gaussian noise row so the mechanism or noise lane can test whether heavier tails alone create a clearer harder front than function-family changes.
- Hypothesis: Laplace noise at `base_scale=1.0` may increase stochastic difficulty without the broader regime ambiguity of the mixture row.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_noise_laplace_v1`.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_noise_laplace`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the Laplace noise family in the per-invocation effective config artifacts.
  - Compare final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third against the carried medium control and the other mechanism or noise rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#150` winner.
- Interpretation status: `pending`
- Decision: `None`
- Benchmark metrics: pending

### 11. `delta_data_manifest_root_tf_rd_020_noise_mixture`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `tf_rd_020_noise_mixture_v1`
- Description: Replace the carried medium corpus with the mixture-noise front at `base_scale=1.0`, `student_t_df=6.0`, and fixed Gaussian/Laplace/Student-t weights.
- Rationale: Close issue `#150` with the heaviest-tailed candidate so TF-RD-020 can test whether one mixed noise surface is the clearest non-mechanism harder front.
- Hypothesis: Mixture noise with `gaussian=0.5`, `laplace=0.3`, `student_t=0.2`, and `student_t_df=6.0` may be the strongest noise candidate, but it risks becoming a noisier and less interpretable front than the mechanism rows.
- Anchor delta: Keep the settled row-first model, preprocessing surface, and four-task warmup-decay recipe fixed, but replace the carried medium corpus with `tf_rd_020_noise_mixture_v1`.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_020_noise_mixture`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Parameter adequacy plan:
  - Confirm the materialized corpus preserves the carried `8/28/4` invocation mix and resolves the mixture-noise family plus mixture weights in the per-invocation effective config artifacts.
  - Compare final ROC AUC first, final log loss and Brier second, and best-to-final ROC delta third against the carried medium control and the other mechanism or noise rows.
  - Use runtime, clipped-step fraction, fallback, NaNs, and OOM or retry behavior only as guardrails when nominating the issue `#150` winner.
- Interpretation status: `pending`
- Decision: `None`
- Benchmark metrics: pending

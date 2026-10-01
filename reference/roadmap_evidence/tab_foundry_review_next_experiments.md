# Proposed next experiments for tab foundry

Status: **proposed, not executed or selected**. This review proposes an
evaluation-first sequence; it does not change the [roadmap](../../docs/development/roadmap.md),
select a sweep, authorize compute, or create issues. Prioritize generalization and
robustness of the carried dense recurrent grid before more architecture variants.

## Evidence boundary and fixed starting points

- Reviewed main: `509bd4ce0c56fdb1d87cd21f100f3ecb2115ee3d`.
- Also reviewed the unmerged [PR #294](https://github.com/bensonlee5/tab-foundry/pull/294),
  head `b99b22721952092466c86105f8751c4cc05547b4`; its MoE results are branch evidence.
- **G2**: `sd_tf_rd_026_grid_sandwich_broad_ml_v1_10_delta_tf_rd_026_grid_recurrent_8_unique2_swiglu_v1_v2`.
  This is the 5,000-step, width-144, eight-iteration/two-unique-layer SwiGLU grid:
  final log loss `0.4181767299`, `2,205,754` parameters, `seed_set=[1]`.
- **G4**: `sd_tf_rd_026_grid_sandwich_broad_ml_v1_11_delta_tf_rd_026_grid_recurrent_8_unique4_swiglu_v1_v1`.
  Final log loss `0.4183364953`, also 5,000 steps and `seed_set=[1]`.
- **S4**: `sd_tf_rd_009_muon_ns_one_epoch_medium_v1_12_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1`,
  the carried 144x4 low-batch sandwich comparator imported into the LMO study.
- Resolve these exact run IDs through the [benchmark registry](../../src/tab_foundry/bench/benchmark_run_registry_v1.json).
  Inspect the retained terminal checkpoint and verify `global_step=5000`; do not
  silently substitute `best.pt`. Checkpoint bytes are not included in this review.
- Training starts from [cls_workstation_grid_sandwich](../../configs/experiment/cls_workstation_grid_sandwich.yaml),
  but replay its diff against G2's `training_surface_record.json` first. Pin the
  corpus artifact `tf_rd_010_dagzoo_medium_control_curated_v6/tf_rd_010_dagzoo_medium_control_curated_v6__b54f7409b6d5`.
- Medium/large manifests are respectively
  `data/manifests/bench/openml_classification_medium_v1/manifest.parquet` and
  `data/manifests/bench/openml_classification_large_v1/manifest.parquet`.
  Their controls are `cls_benchmark_linear_multiclass_medium_v1` and
  `cls_benchmark_linear_multiclass_large_v1`. The [large bundle](../../src/tab_foundry/bench/openml_classification_large_v1.json)
  contains only three pinned tasks; treat its intervals as descriptive and any
  transfer conclusion as provisional, not population-level generalization evidence.

The [TF-RD-026 matrix](../system_delta_sweeps/tf_rd_026_grid_sandwich_broad_ml_v1/matrix.md)
shows only a `0.0001598` G2/G4 gap, without independent training-seed replication.
The 2,500-step [TF-RD-027 screens](../system_delta_sweeps/tf_rd_027_grid_stability_impl_v1/matrix.md)
used a shorter budget than G2; their no-promotion result is not a matched-budget
rejection of those mechanisms. PR #294's completed full MoE rows also did not beat
G2; shared top-1 reached `0.4210020364` with healthy routing. Its load-balance-decay
row remains interpretation-confounded because coefficient telemetry stayed constant.
These caveats qualify attribution; they do not invalidate the recorded results.
The accompanying review fixes affect training evaluation/metric accumulation, not
the separate OpenML benchmark evaluator; no benchmark reranking follows from them.

## Decision and measurement rules

All thresholds below are **suggested pre-registration choices**, not existing
project policy or evidence that an effect exists. Lock them before execution.
Let `delta_LL` be candidate minus paired control log loss; negative is better.
Report the current mean-over-datasets aggregation plus dataset-level deltas,
Brier, ROC AUC diagnostics, measured latency, peak VRAM, and task coverage.
Use 2,000 paired dataset-level bootstrap resamples for 95% intervals; never treat
correlated rows as independent datasets. Report training-seed variation separately.
Use CE `final_log_loss_at_matched_regime_budget`, not historical BPC; flag failures/non-finite output.
Predeclare dataset strata and a deterministic diagnostic/confirmation partition;
the repeatedly used medium benchmark is not a previously untouched holdout.
Choose candidates on the diagnostic partition and confirm once before expanding.

## Compact experiment matrix

| Order | Question | Exact start | Changed variable | First-stage compute | Suggested decision |
| --- | --- | --- | --- | --- | --- |
| 1 | Does the anchor transfer? | G2, G4, S4 terminal checkpoints | Evaluation rung only | Six endpoint evaluations | Confirm G2's large-rung advantage; otherwise defer wider claims |
| 2 | Does arbitrary order change predictions? | G2 terminal checkpoint | Row or feature permutation, separately | 25 medium-equivalent evaluations | Investigate if average excess LL reaches 0.005 |
| 3 | Is recurrent depth useful at inference? | G2 terminal checkpoint | Iterations 4/6/8/10/12, weights fixed | Five depth evaluations | Keep a confirmed quality or latency improvement |
| 4 | Which missingness mismatch hurts? | G2 terminal checkpoint | Mask rate and location, then mechanism | Paired checkpoint-only stress ladder | Follow up if excess LL reaches 0.02 at 30% masking |
| 5 | Is minority context scarcity the bottleneck? | G2 terminal checkpoint | Class prevalence and context size | Paired checkpoint-only binary ladder | Follow up on a reproducible context/shift gap |
| 6 | Do batch gains survive equal exposure? | Grid config above; G2 is historical comparator | Batch within each fixed width | Six training runs; gated replication | Expand only on replicated >=0.005 LL improvement |

## 1 Confirm harder rung transfer before replication

- **Hypothesis:** G2's medium gain persists on larger tables; G2/G4 are practically tied.
- **Ablations:** evaluate G2, G4, S4 endpoints on the locked medium and large
  manifests with identical splits, feature metadata, preprocessing, and coverage.
- **Repeats:** repeat one unchanged endpoint to check determinism. If G2/G4 ranking
  matters afterward, add paired training seeds 2 and 3 to their historical seed 1,
  using the original resolved surfaces and matching each seed's task stream.
- **Metrics:** paired LL/Brier, win/tie/loss counts, and row/feature-count strata.
- **Go:** large G2-minus-S4 mean LL <= -0.005, CI upper bound < 0, and Brier
  delta <= +0.002. Otherwise defer the transfer claim and inspect strata.
- **Stop:** do not replace G2 with G4 unless the replicated mean advantage is
  >=0.002 LL with consistent seed direction; a tie favors the smaller carried G2.
- **Cost:** evaluation-only first; optional replication adds four full training runs.
  Evidence: [roadmap's larger-rung gap](../../docs/development/roadmap.md#current-capability-matrix).

## 2 Test table order sensitivity

- **Hypothesis:** absolute row/column positions create avoidable sensitivity to order.
- **Ablations:** baseline plus eight permutations each of support rows, feature
  columns, and query rows, tested separately. Permute support labels and feature
  types consistently; undo query permutations before scoring and preserve roles.
- **Repeats:** fixed permutation seeds 101-108; unchanged-input repeat as noise control.
- **Metrics:** average/worst permutation excess LL, probability disagreement, Brier,
  and adequately populated shape strata (at least ten datasets per reported stratum).
- **Go:** mean excess LL >=0.005, or stratum excess >=0.02 reproducible on confirmation
  datasets, justifies a four-permutation ensemble test or one isolated training ablation.
- **Stop:** if below both gates, retain the current encoding; do not remove positions
  merely because exact invariance is absent. Measure ensemble latency separately.
- **Cost:** 25 evaluations, no training. Evidence: [feature encoding](../../src/tab_foundry/model/architectures/tabfoundry_sandwich/feature_flow.py)
  explicitly adds `row_pos + col_pos` before mixing; sensitivity is not a proven bug.

## 3 Measure inference depth without changing parameters

- **Hypothesis:** shared grid layers permit useful extra refinement or cheaper truncation.
- **Ablations:** an explicit eval-only override for iterations `{4,6,8,10,12}`,
  with G2 weights, two-layer cyclic order, preprocessing, and readout held fixed.
- **Repeats:** same task order and three timing repetitions after warmup; trace
  per-iteration state/prediction changes. No additional training seeds at this stage.
- **Go:** confirm >=0.002 LL improvement with CI upper <0, or >=20% lower median
  latency with LL CI upper <=+0.002; require Brier delta <=+0.002 in either case.
- **Stop:** retain eight iterations if neither gate passes; a divergent/non-finite
  depth is failed evidence, not permission to retune unrelated stability knobs.
- **Cost:** five depth evaluations. New diagnostic support is required:
  [grid_core_diagnostic.py](../../src/tab_foundry/research/grid_core_diagnostic.py)
  currently rejects recurrent checkpoints. Report parameters and FLOPs separately.
- **Prior:** [Universal Transformers, equations 4-7](https://arxiv.org/pdf/1807.03819)
  use `H_t=LayerNorm(A_t+Transition(A_t))` with shared transitions and step encodings;
  this motivates the question but does not guarantee extrapolation for this model.

## 4 Diagnose missingness mismatch before augmentation

- **Hypothesis:** support/query mismatch matters beyond aggregate missing percentage.
- **Ablations:** initially complete datasets; MCAR `{0,10,30,50}%`, applied to query
  only versus both support/query. Add matched-rate MAR/MNAR only at the first material gap.
- **Repeats:** paired mask seeds 201-205; preserve splits and example membership.
  Define MAR/MNAR from permitted covariates/values, never query labels.
- **Metrics:** excess LL/Brier versus clean, severity-curve area, and missingness strata.
- **Go:** excess LL >=0.02 at 30% missingness with CI lower >0, or a support/query
  location gap >=0.01, justifies one targeted robustness candidate in a separate lane.
- **Stop:** do not promote augmentation unless it later recovers >=25% of excess LL
  while clean-data LL regresses <=0.002; keep TF-RD-009's training corpus unchanged.
- **Cost:** checkpoint-only initially. Evidence: [explicit missingness tokenizer](../../src/tab_foundry/model/components/tabular_primitives.py),
  [TF-RD-014 plan](tf_rd_014_missingness_robustness.md), and the roadmap's negative
  first-pass medium/large missingness-training results.

## 5 Separate class scarcity from prior shift

- **Hypothesis:** minority support count explains errors hidden by overall ROC AUC.
- **Ablations:** binary prevalence `{20,5,1}%` at support sizes `{256,1024}` on
  eligible datasets; match query priors first, then hold queries fixed while changing support priors.
- **Repeats:** five paired subsample seeds 301-305; report actual minority counts,
  zero-minority contexts, and insufficient-data exclusions rather than inventing examples.
- **Metrics:** AP/PR AUC, balanced accuracy, per-class recall, LL/Brier, calibration;
  compare AP only within matched query prevalence and use a fixed threshold initially.
- **Go:** at fixed queries, a support-condition LL gap >=0.02 or AP gap >=0.05
  reproducible on confirmation datasets justifies a focused remedy; otherwise stop.
- **Controls/cost:** evaluation-only; no focal loss or class weighting initially.
  Any optional prior correction must beat its uncorrected paired control by >=0.005 LL.
- **Evidence:** [TF-RD-017](tf_rd_017_class_imbalance_robustness.md) lacks a dedicated
  skew ladder. [BBSE](https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf)
  solves `C_p w = q(predicted_y)` for `w(y)=q(y)/p(y)` under invariant `p(x|y)`,
  label-support coverage, and an invertible confusion operator; do not assume those here.

## 6 Build a small equal exposure batch surface

- **Hypothesis:** optimization batch, not extra exposure alone, improves the grid.
- **Ablations:** widths `{96,144}`, each with fixed recurrence 8/2 and effective
  batches `{32,64,128}`. With microbatch 16, accumulation is `{2,4,8}` and
  320,000 task exposures imply `{10000,5000,2500}` optimizer steps respectively.
- **Controls:** establish a fresh anchor per width; change only training within it.
  Keep the same ordered task stream, corpus, runtime, and shape policy; verify task-ID
  membership, no within-run repetition, and actual tokens rather than assuming step equality.
  Use exposure-normalized warmup/decay and carried LR first; any LR adequacy check
  is a separately declared training row. Do not treat an undertuned miss as rejection.
- **Repeats:** seed 1 for six exploratory cells; confirm the selected cell and its
  same-width batch-64 control with paired seeds 2 and 3 before any promotion.
- **Metrics:** held-out synthetic curves, matched-exposure benchmark LL/Brier,
  measured FLOPs/wall time/VRAM, unique tasks, and geometry-by-batch interactions.
- **Go:** >=0.005 replicated LL gain with CI upper <0 and consistent seed direction,
  or >=20% lower measured compute with LL CI upper <=+0.002; otherwise defer expansion.
- **Cost:** six training runs, then four only if confirmation is warranted. Evidence:
  [corrected scaling audit](tf_rd_009_scaling_law_measurement.md#phase-2-functional-forms)
  keeps `ready_for_cmin=false`; this pilot cannot establish `Bcrit` or compute optimality.

## Execution gate

Complete cheap evaluation stages before optional retraining; no execution is selected
by this note. Before launching, follow [program.md](../../program.md): use new TF-RD
sweep IDs, explicit single-dimension rows, W&B, research cards/campaigns, surface records,
and result cards where required. Preserve historical artifacts, verify checkpoint/data
availability, and retain [#269](https://github.com/bensonlee5/tab-foundry/issues/269)
and [#259](https://github.com/bensonlee5/tab-foundry/issues/259) as separately gated decisions.

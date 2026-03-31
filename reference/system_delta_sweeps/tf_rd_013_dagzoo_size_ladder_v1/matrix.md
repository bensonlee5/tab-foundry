# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_013_dagzoo_size_ladder_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_013_dagzoo_size_ladder_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_013_shape_aware_dagzoo_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_qass_tfcol_large_missing_validation_v1_01_delta_qass_no_column_v3_v1`
- Benchmark bundle: `src/tab_foundry/bench/openml_binary_large_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- External benchmarks: `nanotabpfn`
- Training experiment: `cls_benchmark_staged_corpus`
- Training config profile: `cls_benchmark_staged_corpus`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.4215`, final Brier score `0.2644`, best ROC AUC `0.6702`, final ROC AUC `0.6702`, final training time `2550.1s`

## Anchor Comparison

Upstream reference: `TabICLv2` from `https://arxiv.org/abs/2602.11139`.

| Dimension | Upstream TabICLv2 | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| model anchor | TabICLv2 is the primary row-first architectural reference, but it does not define this exact repo-local promoted-anchor contract. | The settled promoted row-first benchmark anchor `row_cls + qass + no tfcol`. | TF-RD-013 size-ladder work changes only the training-data comparison surface and the stop-condition control, not the promoted model surface. |
| training data surface | TabICLv2 motivates synthetic pretraining at scale but does not define this repo-local manifest contract. | TF-RD-008-scale fresh current-corpus recipe `tf_rd_013_current_corpus_default_v1` with data surface label `anchor_manifest_default`. | The fresh current corpus now mirrors the TF-RD-008 promotion-run manifest scale while the sweep compares three modestly broader shape-aware dagzoo alternatives under the same row caps through first-class `data.corpus_ref` resolution. |
| dagzoo size ladder | Not applicable. | Row 1 uses a 10-dataset default dagzoo recipe as the current-corpus control rather than carrying over a historical local snapshot. | Rows 2-4 now form 20-, 40-, and 80-dataset support rungs through tracked corpus recipes so the realized corpus identity is queryable from local corpus records and training-surface artifacts rather than from inline sweep-local provenance blobs. |
| benchmark and control context | TabICLv2 is the architectural reference, while nanoTabPFN remains the benchmark/control bundle family used by this repo for this decision. | Benchmark bundle `openml_binary_large` remains the benchmark-facing evaluation surface. | TF-RD-013 should keep benchmark/control context stable while it reads corpus-size effects. |
| training stop contract | TabICLv2 does not define this repo-local manifest trainer stop rule. | Registered anchor training surface label `prior_linear_warmup_decay` with `max_steps=2500`. | The fresh current-corpus control clears `runtime.target_train_seconds` so later rows isolate corpus size rather than backend stop semantics. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_training_current_corpus_uncapped` | baseline_replay | yes | completed | none | Run the settled row-first anchor against a TF-RD-008-scale fresh current-corpus manifest from current dagzoo output with the inherited time cap cleared so the 2500-step contract is actually reached. | Run first on the CUDA machine after materializing TF-RD-008-scale corpus recipe `tf_rd_013_current_corpus_default_v1`. |
| 2 | `delta_data_manifest_root_dagzoo_shape_aware_size_small` | provenance | yes | completed | none | Point training at the 20-dataset shape-aware dagzoo support manifest while keeping the three-regime config mix explicit. | Run second and compare directly against the TF-RD-008-scale fresh current-corpus control before spending budget on the broader support rungs. |
| 3 | `delta_data_manifest_root_dagzoo_shape_aware_size_medium` | provenance | yes | completed | none | Point training at the 40-dataset shape-aware dagzoo support manifest so TF-RD-013 can test a middle rung between the 20-dataset and 80-dataset ladders. | Carry this 40-dataset medium rung forward as the representative post-008 synthetic training-data surface for TF-RD-018. |
| 4 | `delta_data_manifest_root_dagzoo_shape_aware_size_large` | provenance | yes | completed | none | Point training at the 80-dataset shape-aware dagzoo support manifest as the upper rung of the TF-RD-013 size ladder. | Treat as the upper support bound only; do not promote over the medium rung unless a later sweep explicitly targets the lower-drift tradeoff. |

## Detailed Rows

### 1. `delta_training_current_corpus_uncapped`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Run the settled row-first anchor against a TF-RD-008-scale fresh current-corpus manifest from current dagzoo output with the inherited time cap cleared so the 2500-step contract is actually reached.
- Rationale: Re-establish the settled row-first anchor on a TF-RD-008-scale fresh current-corpus recipe generated from the current dagzoo default config, then clear the inherited 330-second cap so the remaining TF-RD-013 rows compare against one current same-backend control.
- Hypothesis: A TF-RD-008-scale fresh current-corpus recipe plus cleared `target_train_seconds` should produce the canonical control for the dagzoo size ladder under the current manifest-backed training path.
- Upstream delta: Not applicable; this is a repo-local TF-RD-008-scale fresh current-corpus control row for TF-RD-013.
- Anchor delta: Keep the settled row-first model and preprocessing surfaces fixed, point data at the TF-RD-008-scale fresh current-corpus corpus recipe generated from the current dagzoo default config, and clear the inherited time cap so the manifest trainer stops only at `max_steps=2500`.
- Expected effect: Establish the canonical TF-RD-008-scale fresh current-corpus control before reading any dagzoo size-rung comparison.
- Effective labels: model=`delta_qass_no_column_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.6605`); context (grad `0.2134`)
- Training overrides: `{'apply_schedule': True, 'runtime': {'max_steps': 2500, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Confirm the resolved backend is `manifest`, `runtime.max_steps` is `2500`, and `runtime.target_train_seconds` is `null` before reading any benchmark outcome.
  - Use this row as the TF-RD-008-scale fresh current-corpus control for rows 2-4 rather than as a new training optimization claim.
- Adequacy knobs to dimension explicitly:
  - training.overrides.runtime.target_train_seconds
  - training.overrides.runtime.max_steps
  - training.overrides.schedule.stages[0].steps
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row resolves through corpus recipe `tf_rd_013_current_corpus_default_v1` and does not reuse the stale 2026-02-22 absolute-path snapshot.
  - The historical locked anchor artifact predates the backend-aware manifest runner; this row establishes the canonical TF-RD-008-scale fresh current-corpus control for the remaining TF-RD-013 decision.
  - Canonical rerun registered as `sd_tf_rd_013_dagzoo_size_ladder_v1_01_delta_training_current_corpus_uncapped_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_dagzoo_size_ladder_v1/delta_training_current_corpus_uncapped/result_card.md`
- Registered run: `sd_tf_rd_013_dagzoo_size_ladder_v1_01_delta_training_current_corpus_uncapped_v1` with final log loss `4.9823`, delta final log loss `+4.5608`, final Brier score `0.6889`, delta final Brier score `+0.4245`, best ROC AUC `0.5135`, final ROC AUC `0.4889`, final-minus-best `-0.0246`, delta final ROC AUC `-0.1814`, delta drift `-0.0246`, delta final training time `-2306.1s`

### 2. `delta_data_manifest_root_dagzoo_shape_aware_size_small`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the 20-dataset shape-aware dagzoo support manifest while keeping the three-regime config mix explicit.
- Rationale: Start the reopened TF-RD-013 follow-up with a 20-dataset shape-aware dagzoo support surface so the first size-rung read stays close to the TF-RD-008-scale control while still broadening the mixed-regime evidence.
- Hypothesis: The 20-dataset rung may recover a stronger promoted-anchor read by giving the uncapped 2500-step run more reuse than the old large current-corpus control without collapsing to a single-regime replay.
- Upstream delta: Not applicable; this is a repo-local synthetic-data corpus-size axis.
- Anchor delta: Starting from row 1's TF-RD-008-scale fresh current-corpus control, replace the default current corpus with the `small` 20-dataset shape-aware dagzoo manifest while keeping the model, optimizer family, and stop contract fixed.
- Expected effect: A 20-dataset synthetic support corpus should let the uncapped 2500-step manifest run revisit each sampled regime many times while staying close to the TF-RD-008-scale control.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_shape_aware_size_small`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.6814`); context (grad `0.1596`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Compare manifest characteristics against row 1 before reading benchmark output and confirm the intended `small` rung counts are present.
  - Interpret this row strictly as corpus-size evidence; do not mix in filtering, multiclass, or curated-comparator conclusions.
- Adequacy knobs to dimension explicitly:
  - explicit config ladder coverage across the selected dagzoo shape regimes
  - per-invocation dataset counts and combined split distribution
  - manifest-contract deltas versus the fresh current-corpus control
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Support-bundle regeneration instructions live in `reference/system_delta_sweeps/tf_rd_013_dagzoo_size_ladder_v1/support/README.md`; the tracked JSON summaries should be refreshed on the remote materialization machine.
  - The size ladder stays binary-only, omits the curated comparator, and uses CPU dagzoo generation so the training read stays about corpus content rather than generator hardware.
  - Canonical rerun registered as `sd_tf_rd_013_dagzoo_size_ladder_v1_02_delta_data_manifest_root_dagzoo_shape_aware_size_small_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_dagzoo_size_ladder_v1/delta_data_manifest_root_dagzoo_shape_aware_size_small/result_card.md`
- Registered run: `sd_tf_rd_013_dagzoo_size_ladder_v1_02_delta_data_manifest_root_dagzoo_shape_aware_size_small_v1` with final log loss `2.5230`, delta final log loss `+2.1015`, final Brier score `0.6269`, delta final Brier score `+0.3625`, best ROC AUC `0.5420`, final ROC AUC `0.5404`, final-minus-best `-0.0017`, delta final ROC AUC `-0.1299`, delta drift `-0.0017`, delta final training time `-2322.5s`

### 3. `delta_data_manifest_root_dagzoo_shape_aware_size_medium`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the 40-dataset shape-aware dagzoo support manifest so TF-RD-013 can test a middle rung between the 20-dataset and 80-dataset ladders.
- Rationale: Add a 40-dataset middle rung so the reopened TF-RD-013 decision can distinguish between “dagzoo only helps very close to TF-RD-008 scale” and “dagzoo remains useful across a modest support ladder.”
- Hypothesis: The 40-dataset rung may balance reuse and regime breadth better than either the 20-dataset rung or the broader 80-dataset support surface.
- Upstream delta: Not applicable; this is a repo-local synthetic-data corpus-size axis.
- Anchor delta: Starting from row 1's TF-RD-008-scale fresh current-corpus control, replace the default current corpus with the `medium` 40-dataset shape-aware dagzoo manifest while keeping the model, optimizer family, and stop contract fixed.
- Expected effect: A 40-dataset support corpus should preserve broader regime diversity than the smallest rung while still giving the uncapped 2500-step run substantial reuse relative to the prior oversized surface.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_shape_aware_size_medium`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.4947`); context (grad `0.2903`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Compare manifest characteristics against row 1 before reading benchmark output and confirm the intended `medium` rung counts are present.
  - Use this row to decide whether any improvement appears only after the smallest rung is relaxed.
- Adequacy knobs to dimension explicitly:
  - explicit config ladder coverage across the selected dagzoo shape regimes
  - per-invocation dataset counts and combined split distribution
  - manifest-contract deltas versus the fresh current-corpus control
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - Support-bundle regeneration instructions live in `reference/system_delta_sweeps/tf_rd_013_dagzoo_size_ladder_v1/support/README.md`; the tracked JSON summaries should be refreshed on the remote materialization machine.
  - Canonical rerun registered as `sd_tf_rd_013_dagzoo_size_ladder_v1_03_delta_data_manifest_root_dagzoo_shape_aware_size_medium_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Reused the parent-row nanoTabPFN helper-failure outcome instead of rerunning the nanoTabPFN benchmark helper for this row.
  - Selected on 2026-03-23 as the best-balanced representative post-008 synthetic surface; materially better final log loss, Brier, and ROC AUC than the 10-dataset control without the late-run drift seen on the 80-dataset rung.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_dagzoo_size_ladder_v1/delta_data_manifest_root_dagzoo_shape_aware_size_medium/result_card.md`
- Registered run: `sd_tf_rd_013_dagzoo_size_ladder_v1_03_delta_data_manifest_root_dagzoo_shape_aware_size_medium_v1` with final log loss `2.2604`, delta final log loss `+1.8389`, final Brier score `0.4912`, delta final Brier score `+0.2268`, best ROC AUC `0.5711`, final ROC AUC `0.5625`, final-minus-best `-0.0086`, delta final ROC AUC `-0.1077`, delta drift `-0.0086`, delta final training time `-2323.4s`

### 4. `delta_data_manifest_root_dagzoo_shape_aware_size_large`

- Dimension family: `data`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the 80-dataset shape-aware dagzoo support manifest as the upper rung of the TF-RD-013 size ladder.
- Rationale: Keep one 80-dataset upper support rung in the reopened TF-RD-013 follow-up so the ladder still tests a broader synthetic surface while remaining close enough to the TF-RD-008-scale control to get repeated exposure under the 2500-step contract.
- Hypothesis: The 80-dataset rung may retain enough shape diversity to help without drifting so far from the TF-RD-008-scale control that the comparison becomes mostly about low-exposure breadth.
- Upstream delta: Not applicable; this is a repo-local synthetic-data corpus-size axis.
- Anchor delta: Starting from row 1's TF-RD-008-scale fresh current-corpus control, replace the default current corpus with the `large` 80-dataset shape-aware dagzoo manifest while keeping the model, optimizer family, and stop contract fixed.
- Expected effect: The 80-dataset rung should retain the broadest synthetic coverage in this ladder while still staying close enough to the TF-RD-008-scale control for repeated exposure.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_shape_aware_size_large`, preprocessing=`runtime_default`, training=`linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.1008`); context (grad `0.1378`)
- Data overrides: `{}`
- Parameter adequacy plan:
  - Compare manifest characteristics against row 1 before reading benchmark output and confirm the intended `large` rung counts are present.
  - Use this as the upper boundary of the size ladder before deciding whether TF-RD-013 needs any broader synthetic-data follow-up at all.
- Adequacy knobs to dimension explicitly:
  - explicit config ladder coverage across the selected dagzoo shape regimes
  - per-invocation dataset counts and combined split distribution
  - manifest-contract deltas versus the fresh current-corpus control
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Support-bundle regeneration instructions live in `reference/system_delta_sweeps/tf_rd_013_dagzoo_size_ladder_v1/support/README.md`; the tracked JSON summaries should be refreshed on the remote materialization machine.
  - Canonical rerun registered as `sd_tf_rd_013_dagzoo_size_ladder_v1_04_delta_data_manifest_root_dagzoo_shape_aware_size_large_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Reused the parent-row nanoTabPFN helper-failure outcome instead of rerunning the nanoTabPFN benchmark helper for this row.
  - This rung reached the lowest final log loss in the ladder, but its final Brier regressed and its late-run ROC AUC drift was materially worse than the 40-dataset medium rung.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_dagzoo_size_ladder_v1/delta_data_manifest_root_dagzoo_shape_aware_size_large/result_card.md`
- Registered run: `sd_tf_rd_013_dagzoo_size_ladder_v1_04_delta_data_manifest_root_dagzoo_shape_aware_size_large_v1` with final log loss `2.1742`, delta final log loss `+1.7526`, final Brier score `0.7078`, delta final Brier score `+0.4434`, best ROC AUC `0.6376`, final ROC AUC `0.5378`, final-minus-best `-0.0999`, delta final ROC AUC `-0.1325`, delta drift `-0.0999`, delta final training time `-2332.9s`

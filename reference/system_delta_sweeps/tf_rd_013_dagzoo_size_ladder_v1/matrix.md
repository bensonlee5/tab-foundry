# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_013_dagzoo_size_ladder_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_013_dagzoo_size_ladder_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_013_shape_aware_dagzoo_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_qass_tfcol_large_missing_validation_v1_01_delta_qass_no_column_v3_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- External benchmarks: `nanotabpfn`
- Training experiment: `cls_benchmark_staged`
- Training config profile: `cls_benchmark_staged`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.4215`, final Brier score `0.2644`, best ROC AUC `0.6702`, final ROC AUC `0.6702`, final training time `2550.1s`

## Anchor Comparison

Upstream reference: `TabICLv2` from `https://arxiv.org/abs/2602.11139`.

| Dimension | Upstream TabICLv2 | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| model anchor | TabICLv2 is the primary row-first architectural reference, but it does not define this exact repo-local promoted-anchor contract. | The settled promoted row-first benchmark anchor `row_cls + qass + no tfcol`. | TF-RD-013 size-ladder work changes only the training-data comparison surface and the stop-condition control, not the promoted model surface. |
| training data surface | TabICLv2 motivates synthetic pretraining at scale but does not define this repo-local manifest contract. | Fresh current-corpus manifest generated from the current dagzoo default config with data surface label `anchor_manifest_default`. | The fresh default current corpus remains the baseline while the sweep compares three shrunken shape-aware dagzoo alternatives under the same row caps. |
| dagzoo size ladder | Not applicable. | Row 1 uses the default dagzoo recipe as the current-corpus control rather than carrying over a historical local snapshot. | Each candidate row should keep one top-level `dagzoo_provenance` payload with explicit per-invocation counts so size sensitivity is reviewable rather than implicit. |
| benchmark and control context | TabICLv2 is the architectural reference, while nanoTabPFN remains the benchmark/control bundle family used by this repo for this decision. | Benchmark bundle `nanotabpfn_openml_binary_large` remains the benchmark-facing evaluation surface. | TF-RD-013 should keep benchmark/control context stable while it reads corpus-size effects. |
| training stop contract | TabICLv2 does not define this repo-local manifest trainer stop rule. | Training surface label `prior_linear_warmup_decay` with `max_steps=2500`. | The fresh current-corpus control clears `runtime.target_train_seconds` so later rows isolate corpus size rather than backend stop semantics. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_training_current_corpus_uncapped` | baseline_replay | yes | ready | none | Run the settled row-first anchor against a fresh current-corpus manifest from current dagzoo output with the inherited time cap cleared so the 2500-step contract is actually reached. | Run first on the CUDA machine after bootstrapping `data/manifests/default.parquet` from current dagzoo output. |
| 2 | `delta_data_manifest_root_dagzoo_shape_aware_size_small` | provenance | yes | ready | none | Point training at the smallest shrunken shape-aware dagzoo manifest while keeping the three-regime config mix explicit. | Run second and compare directly against the fresh current-corpus control before spending budget on the larger ladder rungs. |
| 3 | `delta_data_manifest_root_dagzoo_shape_aware_size_medium` | provenance | yes | ready | none | Point training at the medium shrunken shape-aware dagzoo manifest so TF-RD-013 can test a middle rung between the smallest ladder and the previous broad shape-aware surface. | Run after the small rung so size sensitivity is read as a ladder rather than as isolated one-off rows. |
| 4 | `delta_data_manifest_root_dagzoo_shape_aware_size_large` | provenance | yes | ready | none | Point training at the largest shrunken shape-aware dagzoo manifest as the upper rung of the TF-RD-013 size ladder. | Run last so the size ladder resolves from the fresh current-corpus control through small, medium, and upper-bound large corpus sizes. |

## Detailed Rows

### 1. `delta_training_current_corpus_uncapped`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Run the settled row-first anchor against a fresh current-corpus manifest from current dagzoo output with the inherited time cap cleared so the 2500-step contract is actually reached.
- Rationale: Re-establish the settled row-first anchor on a fresh current-corpus manifest generated from the current dagzoo default config, then clear the inherited 330-second cap so the remaining TF-RD-013 rows compare against one current same-backend control.
- Hypothesis: A fresh current-corpus bootstrap plus cleared `target_train_seconds` should produce the canonical control for the dagzoo size ladder under the current manifest-backed training path.
- Upstream delta: Not applicable; this is a repo-local fresh current-corpus control row for TF-RD-013.
- Anchor delta: Keep the settled row-first model and preprocessing surfaces fixed, point data at the fresh current-corpus manifest generated from the current dagzoo default config, and clear the inherited time cap so the manifest trainer stops only at `max_steps=2500`.
- Expected effect: Establish the canonical fresh current-corpus control before reading any dagzoo size-rung comparison.
- Effective labels: model=`delta_qass_no_column_v3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'runtime': {'max_steps': 2500, 'target_train_seconds': None, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False, 'val_batches': 0}, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Confirm the resolved backend is `manifest`, `runtime.max_steps` is `2500`, and `runtime.target_train_seconds` is `null` before reading any benchmark outcome.
  - Use this row as the fresh current-corpus control for rows 2-4 rather than as a new training optimization claim.
- Adequacy knobs to dimension explicitly:
  - training.overrides.runtime.target_train_seconds
  - training.overrides.runtime.max_steps
  - training.overrides.schedule.stages[0].steps
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - This row relies on an explicit `tab-foundry data dagzoo generate-manifest` bootstrap and does not reuse the stale 2026-02-22 absolute-path snapshot.
  - The historical locked anchor artifact predates the backend-aware manifest runner; this row establishes the canonical fresh current-corpus control for the remaining TF-RD-013 decision.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_dagzoo_size_ladder_v1/delta_training_current_corpus_uncapped/result_card.md`
- Benchmark metrics: pending

### 2. `delta_data_manifest_root_dagzoo_shape_aware_size_small`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the smallest shrunken shape-aware dagzoo manifest while keeping the three-regime config mix explicit.
- Rationale: Start the reopened TF-RD-013 follow-up with the smallest shrunken shape-aware dagzoo surface so the first size-rung read is aggressively different from the previous 6272-record shape-aware corpus.
- Hypothesis: The smallest rung may recover a stronger promoted-anchor read by giving the uncapped 2500-step run materially more reuse of the mixed shape-aware regimes.
- Upstream delta: Not applicable; this is a repo-local synthetic-data corpus-size axis.
- Anchor delta: Starting from row 1's fresh current-corpus control, replace the default current corpus with the `small` shape-aware dagzoo manifest while keeping the model, optimizer family, and stop contract fixed.
- Expected effect: A much smaller synthetic corpus should let the uncapped 2500-step manifest run see each sampled regime multiple times without collapsing to the curated tiny-surface failure mode.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_shape_aware_size_small`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Data overrides: `{}`
- Parameter adequacy plan:
  - Compare manifest characteristics against row 1 before reading benchmark output and confirm the intended `small` rung counts are present.
  - Interpret this row strictly as corpus-size evidence; do not mix in filtering, multiclass, or curated-comparator conclusions.
- Adequacy knobs to dimension explicitly:
  - explicit config ladder coverage across the selected dagzoo shape regimes
  - per-invocation dataset counts and combined split distribution
  - manifest-contract deltas versus the fresh current-corpus control
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Support-bundle regeneration instructions live in `reference/system_delta_sweeps/tf_rd_013_dagzoo_size_ladder_v1/support/README.md`; the tracked JSON summaries should be refreshed on the remote materialization machine.
  - The size ladder stays binary-only, omits the curated comparator, and uses CPU dagzoo generation so the training read stays about corpus content rather than generator hardware.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_dagzoo_size_ladder_v1/delta_data_manifest_root_dagzoo_shape_aware_size_small/result_card.md`
- Benchmark metrics: pending

### 3. `delta_data_manifest_root_dagzoo_shape_aware_size_medium`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the medium shrunken shape-aware dagzoo manifest so TF-RD-013 can test a middle rung between the smallest ladder and the previous broad shape-aware surface.
- Rationale: Add a middle size rung so the reopened TF-RD-013 decision can distinguish between “dagzoo only works when heavily shrunken” and “dagzoo is size-insensitive on this anchor.”
- Hypothesis: The medium rung may balance reuse and regime breadth better than either the smallest rung or the previous broad shape-aware surface.
- Upstream delta: Not applicable; this is a repo-local synthetic-data corpus-size axis.
- Anchor delta: Starting from row 1's fresh current-corpus control, replace the default current corpus with the `medium` shape-aware dagzoo manifest while keeping the model, optimizer family, and stop contract fixed.
- Expected effect: A medium corpus should preserve broader regime diversity than the smallest rung while still giving the uncapped 2500-step run meaningfully more reuse than the previous 6272-record surface.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_shape_aware_size_medium`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Data overrides: `{}`
- Parameter adequacy plan:
  - Compare manifest characteristics against row 1 before reading benchmark output and confirm the intended `medium` rung counts are present.
  - Use this row to decide whether any improvement appears only after the smallest rung is relaxed.
- Adequacy knobs to dimension explicitly:
  - explicit config ladder coverage across the selected dagzoo shape regimes
  - per-invocation dataset counts and combined split distribution
  - manifest-contract deltas versus the fresh current-corpus control
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Support-bundle regeneration instructions live in `reference/system_delta_sweeps/tf_rd_013_dagzoo_size_ladder_v1/support/README.md`; the tracked JSON summaries should be refreshed on the remote materialization machine.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_dagzoo_size_ladder_v1/delta_data_manifest_root_dagzoo_shape_aware_size_medium/result_card.md`
- Benchmark metrics: pending

### 4. `delta_data_manifest_root_dagzoo_shape_aware_size_large`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Point training at the largest shrunken shape-aware dagzoo manifest as the upper rung of the TF-RD-013 size ladder.
- Rationale: Keep one upper size rung in the reopened TF-RD-013 follow-up so the ladder still tests a broader synthetic surface while remaining far smaller than the prior shape-aware run.
- Hypothesis: The largest shrunken rung may retain enough shape diversity to help without reintroducing the severe underexposure seen on the previous 6272-record manifest.
- Upstream delta: Not applicable; this is a repo-local synthetic-data corpus-size axis.
- Anchor delta: Starting from row 1's fresh current-corpus control, replace the default current corpus with the `large` shape-aware dagzoo manifest while keeping the model, optimizer family, and stop contract fixed.
- Expected effect: The large shrunken rung should retain the broadest synthetic coverage in this ladder while staying much smaller than the earlier shape-aware follow-up.
- Effective labels: model=`delta_qass_no_column_v3`, data=`tf_rd_013_dagzoo_shape_aware_size_large`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Data overrides: `{}`
- Parameter adequacy plan:
  - Compare manifest characteristics against row 1 before reading benchmark output and confirm the intended `large` rung counts are present.
  - Use this as the upper boundary of the size ladder before deciding whether TF-RD-013 needs any broader synthetic-data follow-up at all.
- Adequacy knobs to dimension explicitly:
  - explicit config ladder coverage across the selected dagzoo shape regimes
  - per-invocation dataset counts and combined split distribution
  - manifest-contract deltas versus the fresh current-corpus control
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Support-bundle regeneration instructions live in `reference/system_delta_sweeps/tf_rd_013_dagzoo_size_ladder_v1/support/README.md`; the tracked JSON summaries should be refreshed on the remote materialization machine.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_013_dagzoo_size_ladder_v1/delta_data_manifest_root_dagzoo_shape_aware_size_large/result_card.md`
- Benchmark metrics: pending

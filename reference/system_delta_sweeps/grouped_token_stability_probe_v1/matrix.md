# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/grouped_token_stability_probe_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `grouped_token_stability_probe_v1`
- Sweep status: `draft`
- Parent sweep id: `tokenization_migration_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_tokenization_migration_v1_01_delta_architecture_screen_grouped_tokens_v2`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Training experiment: `cls_benchmark_staged_prior`
- Training config profile: `cls_benchmark_staged_prior`
- Surface role: `hybrid_diagnostic`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.6372`, final Brier score `0.4451`, best ROC AUC `0.5229`, final ROC AUC `0.5229`, final training time `81.6s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Shared feature encoder path with benchmark-external normalization. | Feature encoder swaps change both the representation path and where normalization lives. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Label-token target conditioning. | Target-conditioning swaps change how labels enter the model and need their own attribution. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Pre-norm cell transformer block with test-self attention enabled. | Block-style changes alter attention flow and should not be conflated with tokenizer or readout deltas. |
| tokenizer | One scalar token per feature. | Shifted grouped tokenizer. | Tokenizer changes reshape the effective table sequence and need their own adequacy commentary. |
| column encoder | None on the upstream direct path. | No column-set encoder on the anchor path. | Column-set modeling remains absent and should not explain anchor behavior. |
| row readout | Target-column readout from the final cell tensor. | Same target-column row pool. | Readout remains on the direct upstream-style path. |
| context encoder | None on the upstream direct path. | None on the anchor path. | Context encoding remains absent; later context rows will change both depth and label-flow semantics. |
| prediction head | Direct binary logits head. | Small-class direct head. | Head changes alter the task contract and should be interpreted separately from shared trunk changes. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_medium` (10 tasks) with data surface label `anchor_manifest_default`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `training_default`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_anchor_activation_trace_baseline` | diagnostics | yes | completed | none | Keep the locked anchor model surface fixed but rerun prior training with activation tracing enabled so forward-pass norm dynamics are captured in `gradient_history.jsonl` and `telemetry.json`. | Use this traced row as the telemetry anchor only; the benchmark-facing grouped-token replay should follow the row-3 no-trace warmup-decay surface rather than this traced run. |
| 2 | `delta_training_linear_decay` | schedule | yes | completed | none | Keep the anchor model, data, and preprocessing surfaces but replace the exact-prior constant LR with single-stage linear decay. | Keep this row as a ROC-oriented tradeoff reference only; prefer the warmup-decay grouped-token surface before any benchmark-facing replay or later row-first architecture work. |
| 3 | `delta_training_linear_warmup_decay` | schedule | yes | completed | none | Keep the anchor model, data, and preprocessing surfaces but use single-stage linear decay with a short warmup. | Run one benchmark-facing grouped-token replay on this no-trace warmup-decay surface, then let TF-RD-005, TF-RD-006, and TF-RD-007 inherit grouped tokens from that replay rather than from `prenorm_block`. |

## Detailed Rows

### 1. `delta_anchor_activation_trace_baseline`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the locked anchor model surface fixed but rerun prior training with activation tracing enabled so forward-pass norm dynamics are captured in `gradient_history.jsonl` and `telemetry.json`.
- Rationale: Establish a traced grouped-token baseline before trying schedule changes so activation windows and benchmark retention can be interpreted against the unchanged grouped-token surface.
- Hypothesis: Activation tracing should be additive only; if the mixed grouped-token benchmark read was mainly a training-adequacy issue, this traced rerun should stay close to the architecture-screen result while exposing whether the longer grouped-token lane keeps stage-local norms bounded.
- Upstream delta: Upstream nanoTabPFN does not emit comparable forward-pass activation telemetry.
- Anchor delta: Keep the grouped-token model, data, and preprocessing surfaces fixed and change only the training recipe by enabling activation tracing on the grouped-token anchor surface.
- Expected effect: No model-quality change is expected; this row establishes the traced anchor baseline needed to interpret later shared-encoder stabilization runs.
- Effective labels: model=`delta_architecture_screen_grouped_tokens`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr_trace_activations`
- Stage-local stability: column (grad `0.0000`, act delta `+1.1362`); row (grad `0.0000`, act delta `+1.2563`)
- Training overrides: `{'runtime': {'trace_activations': True}}`
- Parameter adequacy plan:
  - Confirm the traced run preserves the anchor model, data, and preprocessing surfaces.
  - Use the emitted activation norms as the baseline comparator for all later stabilization rows.
- Adequacy knobs to dimension explicitly:
  - Treat this as a diagnostic rerun of the anchor, not a new structural model claim.
  - Confirm the additive tracing overhead does not perturb the run contract or artifact schema.
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_grouped_token_stability_probe_v1_01_delta_anchor_activation_trace_baseline_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - The traced grouped-token anchor recovered the mixed architecture-screen result: final log loss `0.4002`, final Brier `0.2618`, final ROC AUC `0.7411`, clipped-step fraction `0.0012`, and zero drift.
  - Treat this row as the activation/telemetry reference, not as the preferred default execution surface, because row 3 reproduces the same quality without trace overhead.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/grouped_token_stability_probe_v1/delta_anchor_activation_trace_baseline/result_card.md`
- Registered run: `sd_grouped_token_stability_probe_v1_01_delta_anchor_activation_trace_baseline_v1` with final log loss `0.4002`, delta final log loss `-0.2370`, final Brier score `0.2618`, delta final Brier score `-0.1833`, best ROC AUC `0.7411`, final ROC AUC `0.7411`, final-minus-best `+0.0000`, delta final ROC AUC `+0.2181`, delta drift `+0.0000`, delta final training time `+407.7s`

### 2. `delta_training_linear_decay`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces but replace the exact-prior constant LR with single-stage linear decay.
- Rationale: Test whether grouped tokens mainly need a longer decayed prior-training schedule rather than a different structure.
- Hypothesis: Plain linear decay may recover some late ROC retention on the grouped-token surface, but it may still underperform if the early schedule is too abrupt for the fixed grouped-token trunk.
- Upstream delta: Not applicable; this is a repo-local exact-prior training recipe change.
- Anchor delta: Keep the grouped-token model, data, and preprocessing surfaces fixed and replace the `training_default` anchor recipe with single-stage linear decay.
- Expected effect: Better late-curve retention or stability if the constant exact-prior LR is too aggressive for the anchored surface.
- Effective labels: model=`delta_architecture_screen_grouped_tokens`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_decay`
- Stage-local stability: column (grad `0.0000`, act delta `+1.9172`); row (grad `0.0000`, act delta `+1.9917`)
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0004}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.0}]}}`
- Parameter adequacy plan:
  - Interpret against the constant-LR anchor on both ROC and post-warmup stability diagnostics.
  - Keep optimizer, model, data, and preprocessing fixed for the first pass.
- Adequacy knobs to dimension explicitly:
  - schedule.stages[0].lr_max
  - optimizer.min_lr
  - schedule.stages[0].warmup_ratio
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_grouped_token_stability_probe_v1_02_delta_training_linear_decay_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - No-warmup linear decay improved final ROC AUC to `0.7540`, but it lost to row 1 on final log loss (`0.4055` vs `0.4002`) and final Brier (`0.2664` vs `0.2618`).
  - Stability also regressed: `max_grad_norm` rose to `9.9875` and clipped-step fraction rose to `0.0040`, so this is not the preferred grouped-token training surface.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/grouped_token_stability_probe_v1/delta_training_linear_decay/result_card.md`
- Registered run: `sd_grouped_token_stability_probe_v1_02_delta_training_linear_decay_v1` with final log loss `0.4055`, delta final log loss `-0.2316`, final Brier score `0.2664`, delta final Brier score `-0.1787`, best ROC AUC `0.7542`, final ROC AUC `0.7540`, final-minus-best `-0.0002`, delta final ROC AUC `+0.2310`, delta drift `-0.0002`, delta final training time `+440.0s`

### 3. `delta_training_linear_warmup_decay`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces but use single-stage linear decay with a short warmup.
- Rationale: Separate warmup from decay itself on the grouped-token surface before deciding whether TF-RD-004 stays open.
- Hypothesis: A short warmup may preserve the grouped-token log-loss and Brier gains while recovering some ROC and reducing clipped-step fraction relative to no-warmup decay.
- Upstream delta: Not applicable; this is a repo-local exact-prior training recipe change.
- Anchor delta: Keep the grouped-token model, data, and preprocessing surfaces fixed and replace the `training_default` anchor recipe with linear decay plus short warmup.
- Expected effect: Reduced early instability versus constant LR or plain linear decay, with uncertain final quality impact.
- Effective labels: model=`delta_architecture_screen_grouped_tokens`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`, act delta `+1.1364`); row (grad `0.0000`, act delta `+1.2562`)
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0004}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Compare against both the constant-LR anchor and the no-warmup linear-decay row.
  - Treat best/final ROC, post-warmup train-loss variance, and gradient norms as the primary interpretation surface.
- Adequacy knobs to dimension explicitly:
  - schedule.stages[0].lr_max
  - optimizer.min_lr
  - schedule.stages[0].warmup_ratio
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - Canonical rerun registered as `sd_grouped_token_stability_probe_v1_03_delta_training_linear_warmup_decay_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Warmup+decay without tracing reproduced row 1 almost exactly: final log loss `0.4002`, final Brier `0.2618`, final ROC AUC `0.7414`, clipped-step fraction `0.0012`, and zero drift.
  - Treat this as the preferred grouped-token adequacy surface: it preserves the strong row-1 quality/stability read without the trace-only overhead and clearly dominates the no-warmup decay tradeoff on loss, Brier, and gradient stability.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/grouped_token_stability_probe_v1/delta_training_linear_warmup_decay/result_card.md`
- Registered run: `sd_grouped_token_stability_probe_v1_03_delta_training_linear_warmup_decay_v1` with final log loss `0.4002`, delta final log loss `-0.2370`, final Brier score `0.2618`, delta final Brier score `-0.1833`, best ROC AUC `0.7414`, final ROC AUC `0.7414`, final-minus-best `+0.0000`, delta final ROC AUC `+0.2184`, delta drift `+0.0000`, delta final training time `+382.1s`

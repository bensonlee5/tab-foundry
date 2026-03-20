# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tokenization_migration_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tokenization_migration_v1`
- Sweep status: `completed`
- Parent sweep id: `shared_surface_bridge_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_shared_surface_bridge_v1_03_delta_architecture_screen_prenorm_block_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Training experiment: `cls_benchmark_staged`
- Training config profile: `cls_benchmark_staged`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.6538`, final Brier score `0.4610`, best ROC AUC `0.5479`, final ROC AUC `0.5479`, final training time `79.3s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Shared feature encoder path with benchmark-external normalization. | Feature encoder swaps change both the representation path and where normalization lives. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Label-token target conditioning. | Target-conditioning swaps change how labels enter the model and need their own attribution. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Pre-norm cell transformer block without test-self attention. | Block-style changes alter attention flow and should not be conflated with tokenizer or readout deltas. |
| tokenizer | One scalar token per feature. | Same scalar-per-feature tokenizer. | This sweep isolates the first stage-native tokenizer change on the shared-surface handoff. |
| column encoder | None on the upstream direct path. | No column-set encoder on the anchor path. | Column-set modeling remains absent and should not explain anchor behavior. |
| row readout | Target-column readout from the final cell tensor. | Same target-column row pool. | Readout remains on the direct upstream-style path. |
| context encoder | None on the upstream direct path. | None on the anchor path. | Context encoding remains absent; later context rows will change both depth and label-flow semantics. |
| prediction head | Direct binary logits head. | Direct binary logits head. | The prediction head remains on the narrow upstream-style binary path. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_medium` (10 tasks) with data surface label `anchor_manifest_default`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `training_default`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_architecture_screen_grouped_tokens` | tokenization | yes | completed | grouped_tokens | Replace scalar-per-feature tokenization with the public grouped-token staged recipe on the locked shared-surface handoff. | Run the first stage-native grouped-token benchmark row on top of the locked `prenorm_block` handoff, then create `grouped_token_stability_probe_v1` only if this row registers successfully. |
| 2 | `delta_training_linear_warmup_decay` | schedule | yes | completed | none | Keep the anchor model, data, and preprocessing surfaces but use single-stage linear decay with a short warmup. | Use `sd_tokenization_migration_v1_02_delta_training_linear_warmup_decay_v1` as the canonical grouped-token predecessor for TF-RD-005, TF-RD-006, and TF-RD-007; do not reopen the tokenization handoff unless a later benchmark-facing replay supersedes it. |

## Detailed Rows

### 1. `delta_architecture_screen_grouped_tokens`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `grouped_tokens`
- Description: Replace scalar-per-feature tokenization with the public grouped-token staged recipe on the locked shared-surface handoff.
- Rationale: Evaluate whether the public `grouped_tokens` stage should become the first tokenization milestone after the locked `prenorm_block` shared-surface handoff.
- Hypothesis: Grouped tokenization may provide the first attributable row-first preparation gain on the shared surface, but a mixed benchmark read still needs bounded stability follow-up before later row-CLS or TFCol interpretation moves.
- Upstream delta: Upstream nanoTabPFN keeps one scalar token per feature on the direct binary path.
- Anchor delta: Starting from the locked `prenorm_block` bridge row, switch only to the public `grouped_tokens` stage.
- Expected effect: The tokenizer change becomes active on the shared surface and should be interpreted as the first stage-native row-first preparation step.
- Effective labels: model=`delta_architecture_screen_grouped_tokens`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`training_default`
- Stage-local stability: column (grad `0.0000`, act delta `-0.0005`); row (grad `0.0000`, act delta `-0.0450`)
- Model overrides: `{'stage': 'grouped_tokens'}`
- Parameter adequacy plan:
  - Compare directly against the locked `prenorm_block` shared-surface handoff on the architecture-screen lane.
  - Treat the first run as tokenizer evidence, not as row-pool or context evidence.
  - Require a bounded stability read before treating a weak benchmark result as decisive rejection.
- Adequacy knobs to dimension explicitly:
  - Treat this as a structural milestone before row-CLS, TFCol, or QASS follow-ons.
  - If the benchmark read is mixed, follow with bounded stability tracing before redirecting later migration rows.
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tokenization_migration_v1_01_delta_architecture_screen_grouped_tokens_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Supersedes historical queue run `sd_tokenization_migration_v1_01_delta_architecture_screen_grouped_tokens_v1`; that registry entry is retained as history only.
  - Canonical rerun registered as `sd_tokenization_migration_v1_01_delta_architecture_screen_grouped_tokens_v2`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tokenization_migration_v1/delta_architecture_screen_grouped_tokens/result_card.md`
- Registered run: `sd_tokenization_migration_v1_01_delta_architecture_screen_grouped_tokens_v2` with final log loss `0.6372`, delta final log loss `-0.0166`, final Brier score `0.4451`, delta final Brier score `-0.0159`, best ROC AUC `0.5229`, final ROC AUC `0.5229`, final-minus-best `+0.0000`, delta final ROC AUC `-0.0250`, delta drift `+0.0000`, delta final training time `+2.4s`

### 2. `delta_training_linear_warmup_decay`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the anchor model, data, and preprocessing surfaces but use single-stage linear decay with a short warmup.
- Rationale: Replay the grouped-token row on the benchmark-facing architecture-screen lane with the no-trace warmup-decay training contract selected by the grouped-token adequacy probe.
- Hypothesis: The benchmark-facing grouped-token replay should recover the probe's stable warmup-decay quality while keeping the architecture-screen lane, making the grouped-token handoff canonical for TF-RD-005, TF-RD-006, and TF-RD-007.
- Upstream delta: Not applicable; this is a repo-local exact-prior training recipe change.
- Anchor delta: Keep the stage-native grouped-token model, data, and preprocessing surfaces fixed on `cls_benchmark_staged`, then replace `training_default` with the preferred no-trace `prior_linear_warmup_decay` contract selected by `grouped_token_stability_probe_v1`.
- Expected effect: Reduced early instability versus constant LR or plain linear decay, with uncertain final quality impact.
- Effective labels: model=`delta_architecture_screen_grouped_tokens`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Stage-local stability: column (grad `0.0000`); row (grad `0.0000`)
- Training overrides: `{'apply_schedule': True, 'optimizer': {'name': 'schedulefree_adamw', 'require_requested': True, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'min_lr': 0.0004, 'muon_per_parameter_lr': False}, 'runtime': {'max_steps': 2500, 'eval_every': 25, 'checkpoint_every': 25, 'trace_activations': False}, 'schedule': {'stages': [{'name': 'stage1', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Compare directly against both the mixed architecture-screen grouped-token row and the warmup-decay probe result.
  - Treat this as the benchmark-facing canonical replay for later row-first rows, not as a reopened tokenizer adequacy screen.
- Adequacy knobs to dimension explicitly:
  - schedule.stages[0].lr_max
  - optimizer.min_lr
  - schedule.stages[0].warmup_ratio
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - Canonical rerun registered as `sd_tokenization_migration_v1_02_delta_training_linear_warmup_decay_v1`.
  - Registered the benchmark-facing grouped-token warmup-decay replay; TF-RD-005, TF-RD-006, and TF-RD-007 should anchor on this run.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tokenization_migration_v1/delta_training_linear_warmup_decay/result_card.md`
- Registered run: `sd_tokenization_migration_v1_02_delta_training_linear_warmup_decay_v1` with final log loss `0.4002`, delta final log loss `-0.2536`, final Brier score `0.2618`, delta final Brier score `-0.1992`, best ROC AUC `0.7413`, final ROC AUC `0.7413`, final-minus-best `+0.0000`, delta final ROC AUC `+0.1934`, delta drift `+0.0000`, delta final training time `+342.7s`

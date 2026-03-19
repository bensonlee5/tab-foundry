# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/cuda_stability_followup/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `cuda_stability_followup`
- Sweep status: `completed`
- Parent sweep id: `cuda_capacity_pilot`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.3972`, final Brier score `0.1307`, best ROC AUC `0.7634`, final ROC AUC `0.7634`, final training time `257.5s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| bridge architecture | The registered anchor remains the promoted prenorm plus row-cls bridge recipe from the completed follow-up. | Keep the large CUDA base `nano_exact`, `table_block_style=prenorm`, `row_pool=row_cls`, `tfrow_cls_tokens=2`, `d_col=128`, `d_icl=512`, `tficl_n_heads=8`, `tficl_n_layers=12`, and `head_hidden_dim=1024` fixed across the ladder. | This sweep is primarily a stability-surface investigation; it is not reopening width or depth as a live axis. |
| input normalization | The completed normalization follow-up promoted the clipped batch64 row, and the no-normalization follow-up underperformed on the same systems surface. | Keep `train_zscore_clip` fixed across all rows. | Any movement here should be attributed to batch/LR coupling, not preprocessing. |
| batch and LR coupling | The compact anchor promoted `prior_dump_batch_size=64` with sqrt LR scaling relative to the batch32 reference surface. | Use `batch32` replay as the primary discriminator, then lower the LR ceiling on the same `batch32` surface before revisiting any architecture-level change; keep batch64 no-scale rows as deferred backlog only. | Use this sweep to decide whether the large anchor only needs the compact replay surface or a softer LR surface before any deeper structural change. |
| post-encoder normalization | Upstream nanoTabPFN does not expose this exact staged post-encoder normalization ladder on the large CUDA bridge path. | Keep `post_encoder_norm=none` on the schedule-only rows, then compare `post_encoder_norm=rmsnorm` and `post_encoder_norm=layernorm` on the same batch32 replay surface if drift persists. | Treat the norm-family rows as late-rung stabilization probes after schedule-only batch32 evidence, not as the first response to the drift. |
| execution policy | `cuda_capacity_pilot` would normally benchmark row 1 before width/depth probes. | Treat rows 1-4 as an ordered stability gate and keep the batch64 rows deferred unless the batch32 ladder settles or batch64 memory behavior becomes interpretable. | A row that still shows persistent activation drift is not sufficient to unblock the capacity pilot even if the loss curve looks superficially acceptable. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `dpnb_cuda_large_anchor_batch32_replay` | batch_size | yes | completed | none | Re-run the large CUDA bridge anchor on the compact batch32 replay surface. | Run this row first as the clean batch32 discriminator before lower-LR or post-encoder-norm probes. |
| 2 | `dpnb_cuda_large_anchor_batch32_lr3e3` | schedule | yes | completed | none | Re-run the large CUDA bridge anchor on the compact batch32 replay surface with a lower 3e-3 LR ceiling. | Run after row 1 if the clean batch32 replay still shows upward activation drift. |
| 3 | `dpnb_cuda_large_anchor_batch32_postrms` | normalization | yes | completed | none | Keep the large CUDA bridge anchor on the batch32 replay surface and add explicit post-encoder RMSNorm before the transformer blocks. | Run after the two batch32 schedule rows if upward drift still persists. |
| 4 | `dpnb_cuda_large_anchor_batch32_postln` | normalization | yes | completed | none | Keep the large CUDA bridge anchor on the batch32 replay surface and add explicit post-encoder LayerNorm before the transformer blocks. | Run after the RMSNorm row if the direct norm-family comparison is still needed. |
| 5 | `dpnb_cuda_large_anchor_batch64_noscale` | schedule | yes | deferred_separate_workstream | none | Re-run the large CUDA bridge anchor at batch64 without sqrt LR scaling. | Leave deferred until the batch32 ladder settles and batch64 can be revisited without first-attempt OOM fallback. |
| 6 | `dpnb_cuda_large_anchor_batch64_noscale_lr3e3` | schedule | yes | deferred_separate_workstream | none | Re-run the large CUDA bridge anchor at batch64 without sqrt LR scaling and with a lower 3e-3 LR ceiling. | Keep deferred as backlog evidence until the batch32 ladder resolves and the batch64 memory surface is worth retesting. |

## Detailed Rows

### 1. `dpnb_cuda_large_anchor_batch32_replay`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Re-run the large CUDA bridge anchor on the compact batch32 replay surface.
- Rationale: Use the pre-batch64 replay surface as the cleanest discriminator after both batch64 diagnostics were confounded by activation drift and memory fallback.
- Hypothesis: The large CUDA anchor may train cleanly on the batch32 replay surface even if both batch64 variants drift or need first-attempt OOM retry.
- Upstream delta: Not applicable; this is a repo-local batch/LR-coupling probe on the large CUDA anchor.
- Anchor delta: Keeps the large CUDA anchor fixed but reverts to `prior_dump_batch_size=32`, `prior_dump_lr_scale_rule=none`, and `prior_dump_batch_reference_size=32`.
- Expected effect: Reverting to the batch32 replay surface should show whether the large CUDA anchor can train on the pre-batch64 baseline without reopening architecture.
- Effective labels: model=`dpnb_cuda_large_anchor_batch32_replay`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0004}, 'runtime': {'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': True}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Compare directly against the archived batch64-sqrt and batch64-no-scale diagnostic bundles before attributing any gain to architecture.
  - Treat this as the clean batch32 discriminator and only promote it if it is both stable and benchmark-competitive.
- Adequacy knobs to dimension explicitly:
  - training.prior_dump_batch_size
  - training.prior_dump_lr_scale_rule
  - training.prior_dump_batch_reference_size
  - training.overrides.optimizer.min_lr
  - training.overrides.schedule.stages[0].lr_max
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Stop early if upper-block norms keep a clear positive linear trend after warmup even if loss looks acceptable.
  - Use the archived batch64 diagnostics as comparators rather than as the default execution path.
  - Canonical rerun registered as `sd_cuda_stability_followup_01_dpnb_cuda_large_anchor_batch32_replay_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_stability_followup/dpnb_cuda_large_anchor_batch32_replay/result_card.md`
- Registered run: `sd_cuda_stability_followup_01_dpnb_cuda_large_anchor_batch32_replay_v1` with final log loss `0.5865`, delta final log loss `+0.1893`, final Brier score `0.2020`, delta final Brier score `+0.0712`, best ROC AUC `0.5484`, final ROC AUC `0.5649`, final-minus-best `+0.0166`, delta final ROC AUC `-0.1985`, delta drift `+0.0166`, delta final training time `+1754.5s`

### 2. `dpnb_cuda_large_anchor_batch32_lr3e3`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Re-run the large CUDA bridge anchor on the compact batch32 replay surface with a lower 3e-3 LR ceiling.
- Rationale: If the clean batch32 replay still drifts, lower the LR ceiling before reopening architecture.
- Hypothesis: A 3e-3 LR ceiling on the same batch32 replay surface may flatten upper-block growth without changing the large CUDA bridge architecture.
- Upstream delta: Not applicable; this is a repo-local lower-LR neighbor on the batch32 replay surface for the large CUDA anchor.
- Anchor delta: Keeps the batch32 replay surface fixed but lowers `optimizer.min_lr` to `0.0003` and `schedule.lr_max` to `0.003`.
- Expected effect: If the baseline batch32 replay still drifts, a milder LR surface may flatten upper-block norms without reopening architecture.
- Effective labels: model=`dpnb_cuda_large_anchor_batch32_lr3e3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0003}, 'runtime': {'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': True}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.003, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Compare against the plain batch32 replay before attributing any stabilization gain to architecture rather than LR softness.
  - Treat this as the last schedule-only rung before reopening architecture on the same batch32 surface.
- Adequacy knobs to dimension explicitly:
  - training.prior_dump_batch_size
  - training.prior_dump_lr_scale_rule
  - training.prior_dump_batch_reference_size
  - training.overrides.optimizer.min_lr
  - training.overrides.schedule.stages[0].lr_max
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Prefer this row only if boundedness improves without an obvious quality collapse.
  - Do not reopen normalization until the batch32 LR ladder is exhausted.
  - Canonical rerun registered as `sd_cuda_stability_followup_02_dpnb_cuda_large_anchor_batch32_lr3e3_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_stability_followup/dpnb_cuda_large_anchor_batch32_lr3e3/result_card.md`
- Registered run: `sd_cuda_stability_followup_02_dpnb_cuda_large_anchor_batch32_lr3e3_v1` with final log loss `0.6078`, delta final log loss `+0.2106`, final Brier score `0.2100`, delta final Brier score `+0.0792`, best ROC AUC `0.6755`, final ROC AUC `0.4596`, final-minus-best `-0.2159`, delta final ROC AUC `-0.3039`, delta drift `-0.2159`, delta final training time `+1761.8s`

### 3. `dpnb_cuda_large_anchor_batch32_postrms`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the large CUDA bridge anchor on the batch32 replay surface and add explicit post-encoder RMSNorm before the transformer blocks.
- Rationale: If the schedule-only batch32 rows still drift, try post-encoder RMSNorm as the first architecture-level stabilization probe.
- Hypothesis: Adding `post_encoder_norm=rmsnorm` may re-anchor the transformer input scale on the large prenorm stack without changing the rest of the bridge surface.
- Upstream delta: Upstream nanoTabPFN does not expose this exact staged post-encoder normalization surface on the large CUDA bridge path.
- Anchor delta: Keeps the batch32 replay surface fixed but sets `model.module_overrides.post_encoder_norm=rmsnorm`.
- Expected effect: RMSNorm may bound transformer-input scale on the large prenorm stack if the schedule-only batch32 rows still drift.
- Effective labels: model=`dpnb_cuda_large_anchor_batch32_postrms`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'module_overrides': {'table_block_style': 'prenorm', 'allow_test_self_attention': False, 'post_encoder_norm': 'rmsnorm', 'row_pool': 'row_cls'}, 'stage': 'nano_exact', 'd_col': 128, 'd_icl': 512, 'input_normalization': 'train_zscore_clip', 'tfrow_n_heads': 8, 'tfrow_n_layers': 3, 'tfrow_cls_tokens': 2, 'tfrow_norm': 'layernorm', 'tficl_n_heads': 8, 'tficl_n_layers': 12, 'head_hidden_dim': 1024}`
- Parameter adequacy plan:
  - Compare against both batch32 schedule rows before attributing any gain to architecture rather than LR softness.
  - Only benchmark this row if the activation traces are materially flatter than the schedule-only batch32 rows.
- Adequacy knobs to dimension explicitly:
  - model.module_overrides.post_encoder_norm
  - training.prior_dump_batch_size
  - training.overrides.optimizer.min_lr
  - training.overrides.schedule.stages[0].lr_max
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - RMSNorm is intentionally ordered before LayerNorm in this sweep.
  - Keep the batch32 replay surface otherwise unchanged so the norm-family effect is attributable.
  - Canonical rerun registered as `sd_cuda_stability_followup_03_dpnb_cuda_large_anchor_batch32_postrms_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_stability_followup/dpnb_cuda_large_anchor_batch32_postrms/result_card.md`
- Registered run: `sd_cuda_stability_followup_03_dpnb_cuda_large_anchor_batch32_postrms_v1` with final log loss `0.5932`, delta final log loss `+0.1960`, final Brier score `0.2051`, delta final Brier score `+0.0743`, best ROC AUC `0.5819`, final ROC AUC `0.5482`, final-minus-best `-0.0337`, delta final ROC AUC `-0.2153`, delta drift `-0.0337`, delta final training time `+1760.5s`

### 4. `dpnb_cuda_large_anchor_batch32_postln`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the large CUDA bridge anchor on the batch32 replay surface and add explicit post-encoder LayerNorm before the transformer blocks.
- Rationale: If RMSNorm is mixed or still drifts, run the direct LayerNorm comparator on the same batch32 surface.
- Hypothesis: Adding `post_encoder_norm=layernorm` may stabilize the large prenorm stack more strongly than RMSNorm even if it rescales features more aggressively.
- Upstream delta: Upstream nanoTabPFN does not expose this exact staged post-encoder normalization surface on the large CUDA bridge path.
- Anchor delta: Keeps the batch32 replay surface fixed but sets `model.module_overrides.post_encoder_norm=layernorm`.
- Expected effect: LayerNorm is the direct norm-family comparator to the RMSNorm probe if the first architecture-level stabilization row is mixed or still drifts.
- Effective labels: model=`dpnb_cuda_large_anchor_batch32_postln`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'module_overrides': {'table_block_style': 'prenorm', 'allow_test_self_attention': False, 'post_encoder_norm': 'layernorm', 'row_pool': 'row_cls'}, 'stage': 'nano_exact', 'd_col': 128, 'd_icl': 512, 'input_normalization': 'train_zscore_clip', 'tfrow_n_heads': 8, 'tfrow_n_layers': 3, 'tfrow_cls_tokens': 2, 'tfrow_norm': 'layernorm', 'tficl_n_heads': 8, 'tficl_n_layers': 12, 'head_hidden_dim': 1024}`
- Parameter adequacy plan:
  - Compare directly against the RMSNorm row on the same batch32 replay surface before preferring one normalization family.
  - Only treat this row as a winner if it improves boundedness without giving back too much benchmark quality.
- Adequacy knobs to dimension explicitly:
  - model.module_overrides.post_encoder_norm
  - training.prior_dump_batch_size
  - training.overrides.optimizer.min_lr
  - training.overrides.schedule.stages[0].lr_max
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Earlier repo evidence favored LayerNorm on a different surface, but this row exists as the same-surface comparator to RMSNorm.
  - Keep the batch32 replay surface otherwise unchanged so the norm-family comparison stays clean.
  - Canonical rerun registered as `sd_cuda_stability_followup_04_dpnb_cuda_large_anchor_batch32_postln_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_stability_followup/dpnb_cuda_large_anchor_batch32_postln/result_card.md`
- Registered run: `sd_cuda_stability_followup_04_dpnb_cuda_large_anchor_batch32_postln_v1` with final log loss `0.5901`, delta final log loss `+0.1929`, final Brier score `0.2035`, delta final Brier score `+0.0727`, best ROC AUC `0.5565`, final ROC AUC `0.5841`, final-minus-best `+0.0277`, delta final ROC AUC `-0.1793`, delta drift `+0.0277`, delta final training time `+1761.3s`

### 5. `dpnb_cuda_large_anchor_batch64_noscale`

- Dimension family: `training`
- Status: `deferred_separate_workstream`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Re-run the large CUDA bridge anchor at batch64 without sqrt LR scaling.
- Rationale: Keep the batch64 no-scale probe as backlog evidence rather than the next execution target because both no-scale diagnostics drifted and retried on essentially every step.
- Hypothesis: Removing sqrt LR scaling may still matter, but batch64 remains a confounded surface until it can run without constant first-attempt OOM fallback.
- Upstream delta: Not applicable; this is a repo-local training-surface stabilization probe on the large CUDA anchor.
- Anchor delta: Keeps the large CUDA anchor architecture fixed but removes sqrt LR scaling by setting `prior_dump_lr_scale_rule=none` and `prior_dump_batch_reference_size=64`.
- Expected effect: Lower effective LR should reduce activation drift if the copied batch64-sqrt surface is the main trigger.
- Effective labels: model=`dpnb_cuda_large_anchor_batch64_noscale`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0004}, 'runtime': {'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': True}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Revisit only if the batch32 ladder settles or batch64 memory behavior becomes interpretable without first-attempt fallback.
  - Compare against the archived batch64-sqrt and both archived batch64-no-scale diagnostics before treating any future batch64 retry as new evidence.
- Adequacy knobs to dimension explicitly:
  - training.prior_dump_batch_size
  - training.prior_dump_lr_scale_rule
  - training.prior_dump_batch_reference_size
  - training.overrides.optimizer.min_lr
  - training.overrides.schedule.stages[0].lr_max
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - A stopped diagnostic archive exists at `sd_cuda_stability_followup_01_dpnb_cuda_large_anchor_batch64_noscale_v1_stopped_activation_drift_20260318T233315Z`.
  - A second stopped diagnostic archive exists at `sd_cuda_stability_followup_05_dpnb_cuda_large_anchor_batch64_noscale_v1_stopped_activation_drift_20260319T043910Z`.
  - The second attempted execution was stopped at step `2025` without benchmark registration after continued activation drift and `2026` first-attempt OOM retries to `microbatch_size=32`.
  - Do not use this row as the default next execution target.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_stability_followup/dpnb_cuda_large_anchor_batch64_noscale/result_card.md`
- Benchmark metrics: pending

### 6. `dpnb_cuda_large_anchor_batch64_noscale_lr3e3`

- Dimension family: `training`
- Status: `deferred_separate_workstream`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Re-run the large CUDA bridge anchor at batch64 without sqrt LR scaling and with a lower 3e-3 LR ceiling.
- Rationale: Keep the lower-LR batch64 neighbor as backlog evidence, not the default next run, because the underlying batch64 surface is still confounded by fallback behavior.
- Hypothesis: Lowering LR may still help on batch64, but it is lower priority than resolving whether batch32 can train the large anchor cleanly.
- Upstream delta: Not applicable; this is a repo-local lower-LR neighbor for the large-anchor stability probe.
- Anchor delta: Keeps the large CUDA anchor and batch64 no-scale coupling from row 5 but lowers `optimizer.min_lr` to `0.0003` and `schedule.lr_max` to `0.003`.
- Expected effect: Further reducing the LR surface should help if the no-scale row is still too aggressive for the 12-layer CUDA anchor.
- Effective labels: model=`dpnb_cuda_large_anchor_batch64_noscale_lr3e3`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0003}, 'runtime': {'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': True}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 2500, 'lr_max': 0.003, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Revisit only after the batch32 ladder resolves so any batch64 retry is answering a distinct question.
  - If revived later, compare against the archived batch64 diagnostics before attributing any gain to LR rather than memory behavior.
- Adequacy knobs to dimension explicitly:
  - training.prior_dump_batch_size
  - training.prior_dump_lr_scale_rule
  - training.prior_dump_batch_reference_size
  - training.overrides.optimizer.min_lr
  - training.overrides.schedule.stages[0].lr_max
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - Do not run this row while batch64 still requires first-attempt OOM fallback to microbatch 32 on essentially every step.
  - Treat this as a backlog neighbor to the deferred batch64 no-scale row, not as the next active rung.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_stability_followup/dpnb_cuda_large_anchor_batch64_noscale_lr3e3/result_card.md`
- Benchmark metrics: pending

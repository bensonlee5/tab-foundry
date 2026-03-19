# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/cuda_stack_scale_followup/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `cuda_stack_scale_followup`
- Sweep status: `active`
- Parent sweep id: `cuda_stability_followup`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_cuda_stability_followup_01_dpnb_cuda_large_anchor_batch32_replay_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.5865`, final Brier score `0.2020`, best ROC AUC `0.5484`, final ROC AUC `0.5649`, final training time `2012.0s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| bridge architecture | The inherited large-anchor bridge remains the repo-local prenorm plus row-cls staged surface from the parent sweep. | Keep `nano_exact`, `table_block_style=prenorm`, `row_pool=row_cls`, `d_col=128`, `d_icl=512`, `tficl_n_heads=8`, `tficl_n_layers=12`, and `head_hidden_dim=1024` fixed. | Any movement in rows 1-4 should be attributed to stack-scale interventions or the short-screen policy, not to a new capacity shape. |
| screening budget | The anchor run is a full 2500-step benchmarked batch32 replay. | Rows 1-4 use a 1000-step train-only screen with the same batch32 replay LR surface; row 5 returns to the full 2500-step budget. | Short-screen rows are diagnostic evidence only and must not be read as benchmark-facing replacements for the anchor. |
| post-stack normalization | Upstream nanoTabPFN does not expose a staged post-stack norm on this large-anchor bridge path. | Compare `post_stack_norm=rmsnorm` and `post_stack_norm=layernorm` only on the short-screen surface. | This tests whether final stack export normalization can tame the downstream interface even if prenorm blocks still grow internally. |
| prenorm residual scaling | Upstream nanoTabPFN does not expose a depth-scaled residual branch on this staged prenorm path. | Compare `table_block_residual_scale=depth_scaled` on the same short-screen surface, then combine it with the winning post-stack norm at full budget. | Residual scaling is the more direct root-cause intervention because it changes the in-stack accumulation itself. |
| execution policy | Ordinary sweep rows benchmark and register immediately. | Rows 1-4 are `screen_only`; row 5 is `benchmark_full`. | The queue itself records which rows are diagnostic screens versus benchmark-facing runs so the workflow stays reproducible. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `dpnb_cuda_stack_scale_control` | screening | yes | ready | none | Re-screen the large-anchor batch32 replay surface with a short activation-first diagnostic run. | Run this short-screen control first, then compare the two post-stack norm candidates before deciding on the combined benchmark row. |
| 2 | `dpnb_cuda_stack_scale_poststack_rms` | normalization | yes | ready | none | Add post-stack RMSNorm after the prenorm table-block stack on the batch32 replay control screen. | Run after the short-screen control and compare directly against the LayerNorm variant before carrying one norm into a full benchmark row. |
| 3 | `dpnb_cuda_stack_scale_poststack_ln` | normalization | yes | ready | none | Add post-stack LayerNorm after the prenorm table-block stack on the batch32 replay control screen. | Run after the RMSNorm short screen so the post-stack norm family comparison is available for a winner-take-forward decision. |
| 4 | `dpnb_cuda_stack_scale_depth_scaled` | residual_scaling | yes | ready | none | Keep the batch32 replay control screen fixed and depth-scale each prenorm residual branch by stack depth. | Run after the norm-family screens so the final full-budget row can combine depth scaling with the winning post-stack norm if needed. |
| 5 | `dpnb_cuda_stack_scale_depth_scaled_plus_norm_winner` | normalization | yes | ready | none | Run the full batch32 replay benchmark with depth-scaled prenorm residuals plus whichever post-stack norm wins the short-screen comparison. | Wait for the short-screen norm comparison to resolve, then benchmark the combined depth-scaled row with the winning post-stack norm. |

## Detailed Rows

### 1. `dpnb_cuda_stack_scale_control`

- Dimension family: `training`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Re-screen the large-anchor batch32 replay surface with a short activation-first diagnostic run.
- Rationale: Reconfirm the inherited batch32 replay activation-drift profile on a short diagnostic budget before reading any new stabilization change.
- Hypothesis: The batch32 replay control will still show steadily rising upper-block activation norms over 1000 steps, reproducing the parent-sweep failure mode on a cheaper screen.
- Upstream delta: Not applicable; this is a repo-local control screen for the stack-scale follow-up.
- Anchor delta: Keep the large-anchor batch32 replay surface fixed, but replace the full benchmarked run with a short 1000-step train-only screen.
- Expected effect: Reconfirm the control upper-block activation drift on the short screen budget before reading any new stabilization change.
- Effective labels: model=`dpnb_cuda_stack_scale_control`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0004}, 'runtime': {'grad_clip': 1.0, 'max_steps': 1000, 'trace_activations': True}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 1000, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Treat this row as the short-screen control, not as a benchmark-facing replacement for the registered anchor.
  - Compare the same upper-block screen diagnostics against rows 2-4 before choosing any combined full-budget row.
- Adequacy knobs to dimension explicitly:
  - training.overrides.runtime.max_steps
  - training.overrides.schedule.stages[0].steps
  - training.overrides.schedule.stages[0].warmup_ratio
- Execution policy: `screen_only`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - If upper-block means remain clearly upward after warmup, treat the control as reproduced even if train loss looks superficially acceptable.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_stack_scale_followup/dpnb_cuda_stack_scale_control/result_card.md`
- Benchmark metrics: pending

### 2. `dpnb_cuda_stack_scale_poststack_rms`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Add post-stack RMSNorm after the prenorm table-block stack on the batch32 replay control screen.
- Rationale: Test the cheapest downstream stabilization path first by adding only a post-stack RMSNorm on the same short-screen control surface.
- Hypothesis: RMSNorm after the full block stack may reduce the exported residual scale enough to beat the control on upper-block final-window mean, even if it does not fully address in-stack growth.
- Upstream delta: Upstream nanoTabPFN does not expose this staged post-stack normalization probe on the large-anchor bridge surface.
- Anchor delta: Keep the large-anchor batch32 replay screen fixed and add `model.module_overrides.post_stack_norm=rmsnorm` after the full prenorm table-block stack.
- Expected effect: A post-stack RMSNorm may bound the exported residual stream even if internal prenorm blocks still grow.
- Effective labels: model=`dpnb_cuda_stack_scale_poststack_rms`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'module_overrides': {'table_block_style': 'prenorm', 'allow_test_self_attention': False, 'row_pool': 'row_cls', 'post_stack_norm': 'rmsnorm'}, 'stage': 'nano_exact', 'd_col': 128, 'd_icl': 512, 'input_normalization': 'train_zscore_clip', 'tfrow_n_heads': 8, 'tfrow_n_layers': 3, 'tfrow_cls_tokens': 2, 'tfrow_norm': 'layernorm', 'tficl_n_heads': 8, 'tficl_n_layers': 12, 'head_hidden_dim': 1024}`
- Parameter adequacy plan:
  - Compare directly against the matching LayerNorm row on the same short-screen surface.
  - Use upper-block final-window mean as the first discriminator and keep train loss secondary.
- Adequacy knobs to dimension explicitly:
  - model.module_overrides.post_stack_norm
  - training.overrides.runtime.max_steps
  - training.overrides.schedule.stages[0].steps
- Execution policy: `screen_only`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_stack_scale_followup/dpnb_cuda_stack_scale_poststack_rms/result_card.md`
- Benchmark metrics: pending

### 3. `dpnb_cuda_stack_scale_poststack_ln`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Add post-stack LayerNorm after the prenorm table-block stack on the batch32 replay control screen.
- Rationale: Complete the norm-family comparison on the same short-screen surface so the combined row can choose a downstream norm mechanically instead of by inspection.
- Hypothesis: LayerNorm may outperform RMSNorm on upper-block boundedness and clip rate on this large-anchor bridge surface, even though both act only after the block stack.
- Upstream delta: Upstream nanoTabPFN does not expose this staged post-stack normalization probe on the large-anchor bridge surface.
- Anchor delta: Keep the large-anchor batch32 replay screen fixed and add `model.module_overrides.post_stack_norm=layernorm` after the full prenorm table-block stack.
- Expected effect: A post-stack LayerNorm is the direct norm-family comparator to the RMSNorm probe on the same short-screen surface.
- Effective labels: model=`dpnb_cuda_stack_scale_poststack_ln`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'module_overrides': {'table_block_style': 'prenorm', 'allow_test_self_attention': False, 'row_pool': 'row_cls', 'post_stack_norm': 'layernorm'}, 'stage': 'nano_exact', 'd_col': 128, 'd_icl': 512, 'input_normalization': 'train_zscore_clip', 'tfrow_n_heads': 8, 'tfrow_n_layers': 3, 'tfrow_cls_tokens': 2, 'tfrow_norm': 'layernorm', 'tficl_n_heads': 8, 'tficl_n_layers': 12, 'head_hidden_dim': 1024}`
- Parameter adequacy plan:
  - Compare directly against the matching RMSNorm row on the same short-screen surface.
  - Use the same upper-block-first winner rule so the combined row remains reproducible.
- Adequacy knobs to dimension explicitly:
  - model.module_overrides.post_stack_norm
  - training.overrides.runtime.max_steps
  - training.overrides.schedule.stages[0].steps
- Execution policy: `screen_only`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_stack_scale_followup/dpnb_cuda_stack_scale_poststack_ln/result_card.md`
- Benchmark metrics: pending

### 4. `dpnb_cuda_stack_scale_depth_scaled`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Keep the batch32 replay control screen fixed and depth-scale each prenorm residual branch by stack depth.
- Rationale: Test the root-cause intervention separately by reducing every prenorm residual branch before it is added back into the stack.
- Hypothesis: Depth-scaled residual branches should lower the upper-block growth rate more effectively than a post-stack norm alone because they change the in-stack accumulation itself.
- Upstream delta: Upstream nanoTabPFN does not expose this staged depth-scaled prenorm residual probe on the large-anchor bridge surface.
- Anchor delta: Keep the large-anchor batch32 replay screen fixed and add `model.module_overrides.table_block_residual_scale=depth_scaled`.
- Expected effect: Depth-scaled residual branches should target the source of the in-stack activation ratchet more directly than a final normalization alone.
- Effective labels: model=`dpnb_cuda_stack_scale_depth_scaled`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'module_overrides': {'table_block_style': 'prenorm', 'table_block_residual_scale': 'depth_scaled', 'allow_test_self_attention': False, 'row_pool': 'row_cls'}, 'stage': 'nano_exact', 'd_col': 128, 'd_icl': 512, 'input_normalization': 'train_zscore_clip', 'tfrow_n_heads': 8, 'tfrow_n_layers': 3, 'tfrow_cls_tokens': 2, 'tfrow_norm': 'layernorm', 'tficl_n_heads': 8, 'tficl_n_layers': 12, 'head_hidden_dim': 1024}`
- Parameter adequacy plan:
  - Compare directly against the control and post-stack norm rows on the same short-screen surface.
  - Use this row to decide whether the root-cause residual intervention deserves the one full-budget benchmark row.
- Adequacy knobs to dimension explicitly:
  - model.module_overrides.table_block_residual_scale
  - training.overrides.runtime.max_steps
  - training.overrides.schedule.stages[0].steps
- Execution policy: `screen_only`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_stack_scale_followup/dpnb_cuda_stack_scale_depth_scaled/result_card.md`
- Benchmark metrics: pending

### 5. `dpnb_cuda_stack_scale_depth_scaled_plus_norm_winner`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Run the full batch32 replay benchmark with depth-scaled prenorm residuals plus whichever post-stack norm wins the short-screen comparison.
- Rationale: Spend the full benchmark budget only once, after the short screens identify the better downstream norm to pair with depth-scaled residuals.
- Hypothesis: Depth-scaled prenorm residuals plus the winning post-stack norm will produce the strongest stability evidence and the best chance of a benchmarkable large-anchor candidate.
- Upstream delta: Not applicable; this is the combined full-budget follow-up row derived from the repo-local stack-scale screens.
- Anchor delta: Keep the large-anchor batch32 replay benchmark surface fixed, add `table_block_residual_scale=depth_scaled`, and resolve `post_stack_norm` dynamically from rows 2-3 with RMSNorm winning any unresolved tie.
- Expected effect: The combined row should test whether root-cause residual scaling plus the better downstream stack export norm is enough to produce a benchmarkable large-anchor candidate.
- Effective labels: model=`dpnb_cuda_stack_scale_depth_scaled_plus_norm_winner`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'module_overrides': {'table_block_style': 'prenorm', 'table_block_residual_scale': 'depth_scaled', 'allow_test_self_attention': False, 'row_pool': 'row_cls'}, 'stage': 'nano_exact', 'd_col': 128, 'd_icl': 512, 'input_normalization': 'train_zscore_clip', 'tfrow_n_heads': 8, 'tfrow_n_layers': 3, 'tfrow_cls_tokens': 2, 'tfrow_norm': 'layernorm', 'tficl_n_heads': 8, 'tficl_n_layers': 12, 'head_hidden_dim': 1024}`
- Dynamic model overrides: `{'post_stack_norm': {'kind': 'screen_winner', 'compare_orders': [{'order': 2, 'value': 'rmsnorm'}, {'order': 3, 'value': 'layernorm'}], 'tie_break_preference': 'rmsnorm'}}`
- Parameter adequacy plan:
  - Resolve the winning post-stack norm from rows 2-3 using upper-block final-window scale first, then post-warmup slope, then clip rate, then loss EMA, with RMSNorm as the final tie-breaker.
  - Benchmark only this combined row at full budget after the short-screen control and component rows complete.
- Adequacy knobs to dimension explicitly:
  - model.module_overrides.table_block_residual_scale
  - model.module_overrides.post_stack_norm
  - training.overrides.runtime.max_steps
  - training.overrides.schedule.stages[0].steps
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - The runner resolves `model.module_overrides.post_stack_norm` at execution time from the recorded screen metrics in rows 2-3.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_stack_scale_followup/dpnb_cuda_stack_scale_depth_scaled_plus_norm_winner/result_card.md`
- Benchmark metrics: pending

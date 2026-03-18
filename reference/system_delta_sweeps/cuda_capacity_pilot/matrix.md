# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/cuda_capacity_pilot/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `cuda_capacity_pilot`
- Sweep status: `active`
- Parent sweep id: `input_norm_followup`
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
| bridge architecture | The registered anchor remains the promoted prenorm plus row-cls bridge recipe from the completed follow-up. | Keep `nano_exact`, `table_block_style=prenorm`, `row_pool=row_cls`, `tfrow_cls_tokens=2`, and `post_encoder_norm=none` fixed. | Capacity rows should be read as model-size probes on the bridge surface, not as reopened module-selection experiments. |
| model capacity | The promoted anchor run keeps the compact bridge width while moving to the batch64 sqrt-scaled training surface. | Row 1 moves to the large CUDA base `d_col=128`, `d_icl=512`, `tficl_n_heads=8`, `tficl_n_layers=12`, `head_hidden_dim=1024`. | Rows 2 and 3 should be compared against row 1 first so width and depth evidence stays attributable. |
| input normalization | The completed normalization follow-up promoted the clipped batch64 row, and the no-normalization follow-up underperformed on the same systems surface. | Keep `train_zscore_clip` fixed across this pilot because `input_norm_none_followup` underperformed and did not justify removing explicit clipping. | Any movement in this sweep should not be attributed to preprocessing changes. |
| training recipe | The registered anchor already uses the promoted `prior_linear_warmup_decay` recipe with batch size 64 and sqrt LR scaling. | Keep `prior_linear_warmup_decay`, `prior_dump_batch_size=64`, `prior_dump_lr_scale_rule=sqrt`, `prior_dump_batch_reference_size=32`, and `2500` steps fixed. | Longer-budget adequacy checks belong in `cuda_budget_followup`, not in this capacity readout. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Legacy stage alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `dpnb_cuda_large_anchor` | size_scaling | yes | ready | none | Establish the large CUDA-sized bridge baseline on the stabilized prenorm plus row-cls recipe. | Run this row first to establish the large CUDA bridge baseline before evaluating width or depth probes. |
| 2 | `dpnb_cuda_large_width_x2` | size_scaling | yes | ready | none | Double the large-anchor width-related capacity while keeping bridge modules and depth fixed. | Compare this row against the large anchor and decide whether width deserves a dedicated second sweep. |
| 3 | `dpnb_cuda_large_depth_plus4` | size_scaling | yes | ready | none | Add four ICL transformer layers on top of the large CUDA bridge anchor while keeping width fixed. | Compare this row against the large anchor and then choose whether width, depth, or the large anchor itself should seed the budget follow-up. |

## Detailed Rows

### 1. `dpnb_cuda_large_anchor`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Establish the large CUDA-sized bridge baseline on the stabilized prenorm plus row-cls recipe.
- Rationale: Establish a large CUDA-sized bridge baseline before asking whether width or depth matters more.
- Hypothesis: The promoted bridge recipe should benefit from a larger exact-prior trunk even without changing the optimizer or preprocessing family.
- Upstream delta: Not applicable; this is a repo-local capacity expansion relative to the promoted batch64 sqrt-scaled bridge anchor.
- Anchor delta: Reuses the registered bridge anchor run but promotes the row itself to the large CUDA base with the same bridge modules and training recipe.
- Expected effect: A larger exact-prior trunk may improve quality if the promoted bridge surface is still capacity-limited at the current budget.
- Effective labels: model=`dpnb_cuda_large_anchor`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'module_overrides': {'table_block_style': 'prenorm', 'allow_test_self_attention': False, 'row_pool': 'row_cls'}, 'stage': 'nano_exact', 'd_col': 128, 'd_icl': 512, 'input_normalization': 'train_zscore_clip', 'tfrow_n_heads': 8, 'tfrow_n_layers': 3, 'tfrow_cls_tokens': 2, 'tfrow_norm': 'layernorm', 'tficl_n_heads': 8, 'tficl_n_layers': 12, 'head_hidden_dim': 1024}`
- Parameter adequacy plan:
  - Run this row first and treat it as the local large-model comparator for the rest of the pilot.
  - Rank by the task-family primary final metric first, then supporting diagnostics and final training-time delta versus the promoted anchor.
  - Use activation and clipping telemetry to explain wins or losses, not to override benchmark quality.
- Adequacy knobs to dimension explicitly:
  - model.d_icl
  - model.tficl_n_heads
  - model.tficl_n_layers
  - model.head_hidden_dim
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - This row is the baseline for the pilot even though the sweep anchor remains the promoted compact batch64 bridge run.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_capacity_pilot/dpnb_cuda_large_anchor/result_card.md`
- Benchmark metrics: pending

### 2. `dpnb_cuda_large_width_x2`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Double the large-anchor width-related capacity while keeping bridge modules and depth fixed.
- Rationale: Test whether a substantially wider bridge trunk beats the large anchor without also changing depth.
- Hypothesis: Increasing representational width at fixed depth may recover more quality than adding more layers on this benchmark bundle.
- Upstream delta: Not applicable; this is a repo-local width probe on the large bridge baseline.
- Anchor delta: Changes only the width-related capacity knobs relative to row 1 while leaving bridge modules, normalization, and schedule fixed.
- Expected effect: More width may improve representation quality if the large anchor remains bottlenecked on hidden size rather than depth.
- Effective labels: model=`dpnb_cuda_large_width_x2`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'module_overrides': {'table_block_style': 'prenorm', 'allow_test_self_attention': False, 'row_pool': 'row_cls'}, 'stage': 'nano_exact', 'd_col': 256, 'd_icl': 1024, 'input_normalization': 'train_zscore_clip', 'tfrow_n_heads': 8, 'tfrow_n_layers': 3, 'tfrow_cls_tokens': 2, 'tfrow_norm': 'layernorm', 'tficl_n_heads': 8, 'tficl_n_layers': 12, 'head_hidden_dim': 2048}`
- Parameter adequacy plan:
  - Compare directly against row 1 before reading row 3.
  - Treat this as a sibling model-capacity probe rather than a new long-budget family.
  - Prefer width only if the benchmark win survives the training-time increase.
- Adequacy knobs to dimension explicitly:
  - model.d_col
  - model.d_icl
  - model.head_hidden_dim
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - This row deliberately leaves `tficl_n_layers=12` fixed so the width effect stays attributable.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_capacity_pilot/dpnb_cuda_large_width_x2/result_card.md`
- Benchmark metrics: pending

### 3. `dpnb_cuda_large_depth_plus4`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Add four ICL transformer layers on top of the large CUDA bridge anchor while keeping width fixed.
- Rationale: Test whether modestly deeper ICL processing helps more than width once the large bridge base is established.
- Hypothesis: Extra transformer depth may improve benchmark quality if the large anchor is still under-expressive at fixed width.
- Upstream delta: Not applicable; this is a repo-local depth probe on the large bridge baseline.
- Anchor delta: Changes only `tficl_n_layers` relative to row 1 while keeping width, normalization, and training budget fixed.
- Expected effect: More depth may improve benchmark quality if the large anchor needs extra iterative processing more than extra width.
- Effective labels: model=`dpnb_cuda_large_depth_plus4`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'module_overrides': {'table_block_style': 'prenorm', 'allow_test_self_attention': False, 'row_pool': 'row_cls'}, 'stage': 'nano_exact', 'd_col': 128, 'd_icl': 512, 'input_normalization': 'train_zscore_clip', 'tfrow_n_heads': 8, 'tfrow_n_layers': 3, 'tfrow_cls_tokens': 2, 'tfrow_norm': 'layernorm', 'tficl_n_heads': 8, 'tficl_n_layers': 16, 'head_hidden_dim': 1024}`
- Parameter adequacy plan:
  - Compare directly against row 1 after the width probe is available.
  - Treat this as the depth sibling of row 2, not as evidence for longer training.
  - Open the budget follow-up only after rows 1-3 establish the best architecture of this pilot.
- Adequacy knobs to dimension explicitly:
  - model.tficl_n_layers
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - This row deliberately keeps the large-anchor width fixed so the depth effect stays attributable.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_capacity_pilot/dpnb_cuda_large_depth_plus4/result_card.md`
- Benchmark metrics: pending

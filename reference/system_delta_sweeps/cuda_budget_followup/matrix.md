# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/cuda_budget_followup/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `cuda_budget_followup`
- Sweep status: `draft`
- Parent sweep id: `cuda_capacity_pilot`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_input_norm_followup_01_dpnb_input_norm_anchor_replay_v2`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Comparison policy: `anchor_only`
- Anchor metrics: best ROC AUC `0.7634`, final ROC AUC `0.7567`, final training time `143.0s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| bridge architecture | The capacity pilot will choose between the large anchor, width, and depth probes. | Use the provisional large-anchor model surface for now, then re-anchor this sweep to the winning capacity row before execution. | Architecture choice must be settled before any budget result is interpreted. |
| training budget | The capacity pilot fixes `2500` steps across all rows. | This follow-up isolates `5000` and `10000` steps with the same optimizer and schedule family. | Budget wins indicate undertraining of the chosen architecture, not a new architecture family. |
| input normalization | The completed normalization follow-up found no significant benchmark difference across the tested rows. | Keep `train_zscore_clip` fixed until the separate no-normalization workstream changes the default with stronger evidence. | Budget rows must not be conflated with preprocessing experiments. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Legacy stage alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `dpnb_cuda_budget_5k` | budget | yes | blocked_on_anchor_selection | none | Re-run the chosen large-capacity bridge row for 5000 steps under the same optimizer and schedule family. | Wait for `cuda_capacity_pilot` to choose the winning architecture, then replace the provisional model surface before running this budget row. |
| 2 | `dpnb_cuda_budget_10k` | budget | yes | blocked_on_anchor_selection | none | Re-run the chosen large-capacity bridge row for 10000 steps under the same optimizer and schedule family. | Leave this row blocked until the capacity winner is known and the `5000`-step result shows remaining headroom. |

## Detailed Rows

### 1. `dpnb_cuda_budget_5k`

- Dimension family: `training`
- Status: `blocked_on_anchor_selection`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Re-run the chosen large-capacity bridge row for 5000 steps under the same optimizer and schedule family.
- Rationale: Extend the winning capacity row to `5000` steps only after the architecture choice is complete.
- Hypothesis: If the chosen large-capacity row is compute-limited at `2500` steps, doubling the budget should improve final ROC AUC without reopening architecture questions.
- Upstream delta: Not applicable; this is a repo-local training-budget adequacy probe.
- Anchor delta: Keeps the provisional large-anchor model surface for scaffolding now, but this row must be rebound to the winning capacity architecture before execution.
- Expected effect: If the chosen capacity row is compute-limited at 2500 steps, doubling the budget should improve final ROC AUC without changing the mechanism family.
- Effective labels: model=`dpnb_cuda_large_anchor`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0004}, 'runtime': {'grad_clip': 1.0, 'max_steps': 5000, 'trace_activations': True}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 5000, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Do not run this row until `cuda_capacity_pilot` identifies the winning architecture.
  - Re-anchor the model surface first, then compare directly against that winning `2500`-step row.
  - Treat any gain as budget adequacy evidence only.
- Adequacy knobs to dimension explicitly:
  - training.overrides.runtime.max_steps
  - training.overrides.schedule.stages[0].steps
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - The model payload is intentionally provisional and mirrors the large anchor only to keep the scaffold renderable.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_budget_followup/dpnb_cuda_budget_5k/result_card.md`
- Benchmark metrics: pending

### 2. `dpnb_cuda_budget_10k`

- Dimension family: `training`
- Status: `blocked_on_anchor_selection`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Re-run the chosen large-capacity bridge row for 10000 steps under the same optimizer and schedule family.
- Rationale: Test whether the chosen capacity winner is still compute-limited even after a `5000`-step rerun.
- Hypothesis: A further increase to `10000` steps should help only if the winning architecture remains undertrained after the `5000`-step row.
- Upstream delta: Not applicable; this is a repo-local training-budget adequacy probe.
- Anchor delta: Keeps the provisional large-anchor model surface for scaffolding now, but this row must be rebound to the winning capacity architecture before execution.
- Expected effect: Further budget may help only if the chosen capacity row still looks compute-limited after the 5000-step rerun.
- Effective labels: model=`dpnb_cuda_large_anchor`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Training overrides: `{'apply_schedule': True, 'optimizer': {'min_lr': 0.0004}, 'runtime': {'grad_clip': 1.0, 'max_steps': 10000, 'trace_activations': True}, 'schedule': {'stages': [{'name': 'prior_dump', 'steps': 10000, 'lr_max': 0.004, 'lr_schedule': 'linear', 'warmup_ratio': 0.05}]}}`
- Parameter adequacy plan:
  - Do not run this row until `cuda_capacity_pilot` completes and the `5000`-step row is interpretable.
  - Re-anchor the model surface first, then compare against the winning `2500`-step row and the `5000`-step follow-up.
  - Stop if `5000` steps already saturates the benchmark outcome.
- Adequacy knobs to dimension explicitly:
  - training.overrides.runtime.max_steps
  - training.overrides.schedule.stages[0].steps
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - This row should remain dormant unless the `5000`-step follow-up still looks compute-limited.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/cuda_budget_followup/dpnb_cuda_budget_10k/result_card.md`
- Benchmark metrics: pending

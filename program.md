# Agent Program

This file is the agent contract for the anchor-only system-delta sweep system in
`tab-foundry`.

Use `docs/workflows.md` for command syntax and artifact expectations. Use this
file for the objective, the locked comparison surface, the queue discipline,
and the interpretation policy.

## Objective

Optimize for attributable evidence against the locked anchor
`sd_input_norm_followup_01_dpnb_input_norm_anchor_replay_v2`, not for rapid base
promotion.

The primary score remains `final_roc_auc` on the canonical benchmark bundle
`src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`.

Supporting metrics are:

- `best_roc_auc`
- `final_minus_best`
- training-time deltas versus the anchor
- manifest and preprocessing surface deltas recorded in `training_surface_record.json`
- loss/gradient instability evidence from `train_history.jsonl`,
  `gradient_history.jsonl`, and `telemetry.json`

`best_roc_auc` is a tie-breaker and diagnostic, not the main score.

## Locked Anchor Surface

Hold this surface fixed unless the queue row explicitly declares a different
dimension family:

- active sweep id: `cuda_capacity_pilot`
- anchor run id: `sd_input_norm_followup_01_dpnb_input_norm_anchor_replay_v2`
- anchor prior run: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_anchor_replay/sd_input_norm_followup_01_dpnb_input_norm_anchor_replay_v2/train`
- anchor benchmark: `outputs/staged_ladder/research/input_norm_followup/dpnb_input_norm_anchor_replay/sd_input_norm_followup_01_dpnb_input_norm_anchor_replay_v2/benchmark`
- canonical benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- canonical control baseline id: `cls_benchmark_linear_v2`
- canonical registry: `src/tab_foundry/bench/benchmark_run_registry_v1.json`
- delta catalog: `reference/system_delta_catalog.yaml`
- sweep index: `reference/system_delta_sweeps/index.yaml`
- canonical sweep queue: `reference/system_delta_sweeps/cuda_capacity_pilot/queue.yaml`
- canonical sweep matrix: `reference/system_delta_sweeps/cuda_capacity_pilot/matrix.md`
- active queue alias: `reference/system_delta_queue.yaml`
- active matrix alias: `reference/system_delta_matrix.md`
- research template: `reference/system_delta_campaign_template.md`
- research sources: `reference/stage_research_sources.yaml`

Keep these invariant by default:

- prior-trained experiment family: `cls_benchmark_staged_prior`
- benchmark bundle path
- control baseline id
- history, checkpoint, benchmark, and `training_surface_record.json` artifact contracts

The benchmark registry is the historical system of record.

The `outputs/staged_ladder/...` anchor paths above are convenience runtime
references for local workspaces. They may be absent in a fresh clone or CI
checkout. Resolve canonical identity through
`src/tab_foundry/bench/benchmark_run_registry_v1.json`.

## Dimension Families

This sweep is not architecture-only. Every queue row must isolate exactly one
declared dimension family against the anchor:

- model
- training
- data
- preprocessing

Examples of valid dimensions include:

- module selection inside `tabfoundry_staged`
- training data source and manifest root
- dagzoo provenance for a manifest-backed surface
- runtime preprocessing and encoding policy

Any mechanism, data, or preprocessing candidate is allowed as long as the row
states the exact preserved settings and the exact changed settings.

## Queue And Matrix

The canonical source-of-truth hierarchy is:

- `reference/system_delta_catalog.yaml`
- `reference/system_delta_sweeps/index.yaml`
- `reference/system_delta_sweeps/<sweep_id>/queue.yaml`
- `reference/system_delta_sweeps/<sweep_id>/matrix.md`

The top-level files `reference/system_delta_queue.yaml` and
`reference/system_delta_matrix.md` are generated compatibility aliases for the active sweep only.
They are convenient views, not the canonical editable sources.

The queue must carry, at minimum:

- `order`, `delta_ref`, `status`
- `description`, `rationale`, `hypothesis`
- model/data/preprocessing labels and the one active override family
- `parameter_adequacy_plan`
- `run_id`, `followup_run_ids`
- `decision`, `interpretation_status`, `confounders`, `next_action`, `notes`

The matrix must be rerendered from the active or selected sweep plus the
canonical benchmark registry. Metrics belong in the registry, not duplicated in
the queue.

Use `scripts/system_delta_queue.py` to:

- list sweeps and show the active sweep
- create a new sweep or set a different active sweep
- list rows in order
- print the next `ready` row
- validate completed rows
- render `reference/system_delta_sweeps/<sweep_id>/matrix.md`

Every benchmark-facing run belongs to exactly one `sweep_id`. New complexity
passes should create a new sweep instead of mutating an old completed one.

## Required Research Package

Before any empirical run for a queue row, create:

- `outputs/staged_ladder/research/<sweep_id>/<delta_id>/research_card.md`
- `outputs/staged_ladder/research/<sweep_id>/<delta_id>/campaign.yaml`

After the run is benchmarked and registered, also create:

- `outputs/staged_ladder/research/<sweep_id>/<delta_id>/result_card.md`

Use `reference/system_delta_campaign_template.md` and
`reference/stage_research_sources.yaml`.

Agents should use optional sibling-workspace sources when available, but must
be able to proceed from the required repo-local sources alone.

Every completed run must have a `training_surface_record.json` artifact. That
record is the system-surface evidence source for:

- model surface labels and effective module selections
- surfaced subsystem hyperparameters
- data source and manifest fingerprint
- dagzoo provenance references when applicable
- dataset-characteristic summaries
- preprocessing surface labels and explicit overrides

Queue reruns for instability debugging must also produce:

- `train_history.jsonl` as the canonical scalar timeline
- `gradient_history.jsonl` with module-level gradient traces
- `telemetry.json` with run summary, artifact pointers, checkpoint snapshots,
  missingness diagnostics, and failure context

Treat the completed first-pass `binary_md_v1` outputs under
`outputs/staged_ladder/sd_binary_md_v1_*` as read-only baseline evidence. Do
not overwrite those run directories when adding instability instrumentation.
Use fresh rerun roots such as
`outputs/staged_ladder/<run_id>_diag_v1/train`. Historical runs can only be
audited from their scalar histories; true module-level traces only exist for
new reruns.

## Execution Loop

For each queue row:

1. Select the active sweep from `reference/system_delta_sweeps/index.yaml`, or pass an explicit `--sweep-id`.
1. Select the next `status=ready` row from `reference/system_delta_sweeps/<sweep_id>/queue.yaml`.
1. Write or update `research_card.md` and `campaign.yaml`.
1. Train on the locked anchor surface, changing only the declared dimension.
1. Benchmark on `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`.
1. Register the benchmark-facing run in `src/tab_foundry/bench/benchmark_run_registry_v1.json`, including its `sweep_id`.
1. Ensure the run has `training_surface_record.json`,
   `gradient_history.jsonl`, and `telemetry.json`.
1. Write `result_card.md`.
1. Rerender `reference/system_delta_sweeps/<sweep_id>/matrix.md`.
1. Regenerate the top-level alias files if and only if this is the active sweep.
1. Update the queue row status, run ids, interpretation, and next action.

To rank the existing first-pass `binary_md_v1` outputs before rerunning,
generate the scalar instability audit report under
`outputs/staged_ladder/reports/` with:

```bash
uv run python -m tab_foundry.bench.instability_audit \
  --staged-ladder-root outputs/staged_ladder \
  --sweep-id binary_md_v1
```

This pass is attribution-first. No row becomes the new base during the sweep.

## Decisions

Use these decisions:

- `keep`: the row is isolated, evidence is at least neutral or positive on `final_roc_auc`, and the interpretation does not reveal unresolved confounding severe enough to block the signal
- `defer`: evidence is mixed, the row is not isolated enough yet, or the introduced degrees of freedom have not been checked adequately
- `reject`: only allowed when the row is isolated, the adequacy plan was completed, and the result is clearly worse without offsetting benefit

Underperformance alone is not enough for `reject`.

Every `result_card.md` must explain:

- what changed
- what metrics moved versus the anchor
- whether the change was actually isolated
- whether introduced hyperparameters were adequate
- why the change may have helped or hurt
- which confounders remain
- what bounded follow-up is still required, if any

Rows like `delta_row_cls_pool` are the template for this policy. If `row_cls`
underperforms once, that is not evidence against the mechanism by itself. The
result card must discuss whether `tfrow_n_heads`, `tfrow_n_layers`, and
`tfrow_cls_tokens` were likely appropriate before any negative conclusion is
treated as strong evidence.

The same rule applies to data and preprocessing rows. If a data-surface or
preprocessing row underperforms, the result card must discuss manifest
provenance, dataset-characteristic shifts, and preprocessing interaction before
any rejection is allowed.

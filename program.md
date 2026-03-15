# Agent Program

This file is the agent contract for the anchor-only system-delta sweep in
`tab-foundry`.

Use `docs/workflows.md` for command syntax and artifact expectations. Use this
file for the objective, the locked comparison surface, the queue discipline,
and the interpretation policy.

## Objective

Optimize for attributable evidence against the locked anchor
`01_nano_exact_md_prior_parity_fix_binary_medium_v1`, not for rapid base
promotion.

The primary score remains `final_roc_auc` on the canonical benchmark bundle
`src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`.

Supporting metrics are:

- `best_roc_auc`
- `final_minus_best`
- training-time deltas versus the anchor
- manifest and preprocessing surface deltas recorded in `training_surface_record.json`

`best_roc_auc` is a tie-breaker and diagnostic, not the main score.

## Locked Anchor Surface

Hold this surface fixed unless the queue row explicitly declares a different
dimension family:

- anchor run id: `01_nano_exact_md_prior_parity_fix_binary_medium_v1`
- anchor prior run: `outputs/staged_ladder/01_nano_exact_md/prior_parity_fix`
- anchor benchmark: `outputs/staged_ladder/01_nano_exact_md/prior_benchmark_binary_medium_v1`
- canonical benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- canonical control baseline id: `cls_benchmark_linear_v2`
- canonical registry: `src/tab_foundry/bench/benchmark_run_registry_v1.json`
- queue source of truth: `reference/system_delta_queue.yaml`
- rendered comparison matrix: `reference/system_delta_matrix.md`
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

`reference/system_delta_queue.yaml` is the machine-readable source of truth for
the sweep. `reference/system_delta_matrix.md` is the rendered human-readable
view.

The queue must carry, at minimum:

- `order`, `delta_id`, `status`, `dimension_family`, `family`
- `description`, `rationale`, `hypothesis`
- model/data/preprocessing labels and the one active override family
- `parameter_adequacy_plan`
- `run_id`, `followup_run_ids`
- `decision`, `interpretation_status`, `confounders`, `next_action`, `notes`

The matrix must be rerendered from the queue plus the canonical benchmark
registry. Metrics belong in the registry, not duplicated in the queue.

Use `scripts/system_delta_queue.py` to:

- list rows in order
- print the next `ready` row
- validate completed rows
- render `reference/system_delta_matrix.md`

## Required Research Package

Before any empirical run for a queue row, create:

- `outputs/staged_ladder/research/<delta_id>/research_card.md`
- `outputs/staged_ladder/research/<delta_id>/campaign.yaml`

After the run is benchmarked and registered, also create:

- `outputs/staged_ladder/research/<delta_id>/result_card.md`

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

## Execution Loop

For each queue row:

1. Select the next `status=ready` row from `reference/system_delta_queue.yaml`.
2. Write or update `research_card.md` and `campaign.yaml`.
3. Train on the locked anchor surface, changing only the declared dimension.
4. Benchmark on `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`.
5. Register the benchmark-facing run in `src/tab_foundry/bench/benchmark_run_registry_v1.json`.
6. Ensure the run has `training_surface_record.json`.
7. Write `result_card.md`.
8. Rerender `reference/system_delta_matrix.md`.
9. Update the queue row status, run ids, interpretation, and next action.

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

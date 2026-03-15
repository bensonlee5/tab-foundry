# Agent Program

This file is the agent contract for architecture search in `tab-foundry`.
Use it to decide what to optimize, what must stay fixed, what evidence is
required, and when a candidate can become the new base.

Use `docs/workflows.md` for command syntax and artifact expectations. Use this
file for objective, search policy, and decision rules.

## Objective

Optimize prior-trained candidates against the locked exact anchor
`01_nano_exact_md_prior_parity_fix_binary_medium_v1`.

The primary optimization target for now is `final_roc_auc` from the canonical
benchmark summary on
`src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`.

Supporting checks are:

- `best_roc_auc`
- `final_minus_best`
- post-warmup train-loss variance
- training-time delta versus the exact anchor

`best_roc_auc` is a tie-breaker and diagnostic, not the main score.

## Locked Baseline And Comparison Surface

Hold this surface fixed unless the research package explicitly declares a new
comparison surface and the benchmark conclusion calls it out:

- anchor run id: `01_nano_exact_md_prior_parity_fix_binary_medium_v1`
- anchor prior run: `outputs/staged_ladder/01_nano_exact_md/prior_parity_fix`
- anchor benchmark: `outputs/staged_ladder/01_nano_exact_md/prior_benchmark_binary_medium_v1`
- stage template: `reference/stage_campaign_template.md`
- research sources: `reference/stage_research_sources.yaml`
- canonical benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- canonical control baseline id: `cls_benchmark_linear_v2`
- canonical registry: `src/tab_foundry/bench/benchmark_run_registry_v1.json`

Keep these invariant by default:

- prior dump path
- benchmark bundle path
- `nanoTabPFN` helper settings
- device class
- prior-trained experiment family: `cls_benchmark_staged_prior`
- history, checkpoint, and benchmark artifact contracts

The legacy 3-task binary surface remains available at
`src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json`, and
`cls_benchmark_linear_v1` remains in the control-baseline registry for
historical reproduction only. Results are not directly comparable across the
legacy and medium binary surfaces unless the benchmark bundle path and control
baseline id match exactly.

Use cached `nanotabpfn_curve.jsonl` only when the comparison surface is
unchanged.

`01_nano_exact_md_prior_parity_fix_binary_medium_v1` is the canonical anchor
identity for this search cycle. Resolve its artifact paths through
`src/tab_foundry/bench/benchmark_run_registry_v1.json`; that registry is the
historical system of record.

The benchmark registry is the historical system of record.

The `outputs/staged_ladder/...` anchor paths above are convenience runtime
references for local workspaces. They may be absent in a fresh clone or CI
checkout.

## Search Scope

Any mechanism or preprocessing candidate is allowed, including later
mechanisms or unrelated preprocessing changes, as long as the candidate is
well documented and benchmarked on the locked surface.

Candidates may change code under:

- `src/tab_foundry/model/`
- `src/tab_foundry/preprocessing/`

Changes to `src/tab_foundry/training/` or `src/tab_foundry/bench/` require
explicit research and benchmark justification in the candidate package.

Prefer simple, attributable changes over broad sweeps that make the effect
unclear.

## Required Research Package

Before any empirical run for a candidate, create:

- `outputs/staged_ladder/research/<candidate_id>/research_card.md`
- `outputs/staged_ladder/research/<candidate_id>/campaign.yaml`

Use `reference/stage_campaign_template.md` and
`reference/stage_research_sources.yaml`.

Agents should use optional sibling-workspace sources when available, but must
be able to proceed from the required repo-local sources alone.

`research_card.md` must state:

- `candidate_id`
- `mechanism_family`
- `touched_subsystems`
- why the candidate is entering now, especially if it jumps ahead of more
  obvious candidates
- whether it preserves the current comparison surface or intentionally defines
  a new one
- what exact baseline settings should be preserved
- what settings should shift, and why
- direct evidence used for that recommendation
- one prediction: `preserve baseline settings`, `directional shift`, or `uncertain`

`campaign.yaml` must state:

- `candidate_id`
- `stage`
- `mechanism_family`
- `touched_subsystems`
- `comparison_surface`
- `primary_metric: final_roc_auc`
- `anchor_run_id`
- `recommended_recipe`
- `preserved_settings`
- `shifted_settings`
- `tunable_params`
- `full_budget_steps`
- `full_bundle_path`
- `decision_hypothesis`

If the research pass says a setting should stay unchanged, record the preserved
baseline value explicitly.

## Execution Loop

For each candidate:

1. Write the research package.
2. Run one confirmatory prior-trained branch with the research-recommended
   recipe via `scripts/train_tabfoundry_staged_prior.py`.
3. Benchmark it via `scripts/benchmark_nanotabpfn.py`.
4. If the result is stable but inconclusive, perform bounded local refinement
   only inside `tunable_params`.
5. Promote at most two promising variants to full-benchmark candidates.
6. Register each benchmark-facing candidate via `scripts/register_benchmark_run.py`.

Bounded local refinement may search only inside the neighborhood declared in
`campaign.yaml`. Do not broaden the search space ad hoc.

## Decisions

Use these branch decisions:

- `keep`: driven primarily by `final_roc_auc`; a full-benchmark candidate is
  near-neutral or better on `final_roc_auc` versus the exact anchor and does
  not materially worsen late drift
- `defer`: the candidate is stable but evidence is mixed, too narrow, or
  compute-sensitive
- `reject`: the candidate is clearly worse across multiple benchmark-facing
  attempts or is genuinely unstable

Never reject a candidate from one run alone.

## Scaling Confirmation

Only `keep` candidates enter scaling confirmation.

For a kept candidate:

1. Run the exact control `01_nano_exact_md_prior_budget1250`.
2. Run the winning candidate recipe at `1250` and `2500` steps.
3. Benchmark both on `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`.
4. Compare candidate versus exact at both budgets.

Promote the candidate only if it is at least neutral or better on
`final_roc_auc` at both budgets and the gain does not disappear at `2500`
steps.

If it helps at one budget but not the other, mark it `defer` and keep
`01_nano_exact_md_prior_parity_fix_binary_medium_v1` as the active base.

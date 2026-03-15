# Agent Program

This file is the agent contract for architecture search in `tab-foundry`.
Use it to decide what to optimize, what must stay fixed, what evidence is
required, and when a staged branch can become the new base.

Use `docs/workflows.md` for command syntax and artifact expectations. Use this
file for objective, policy, and decision rules.

## Objective

Optimize staged prior-trained branches against the locked exact anchor
`01_nano_exact_md_prior_parity_fix`.

The target is:

- improve or match the exact anchor on the canonical benchmark surface
- preserve training stability and prior-training semantics
- require compute-scaling confirmation before a stage becomes the new base

## Locked Baseline And Comparison Surface

Hold this surface fixed unless the stage research package explicitly declares a
new comparison surface and the benchmark conclusion calls it out:

- anchor run id: `01_nano_exact_md_prior_parity_fix`
- anchor prior run: `outputs/staged_ladder/01_nano_exact_md/prior_parity_fix`
- anchor benchmark: `outputs/staged_ladder/01_nano_exact_md/prior_benchmark_parity_fix`
- stage template: `reference/stage_campaign_template.md`
- research sources: `reference/stage_research_sources.yaml`
- canonical benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json`
- canonical registry: `src/tab_foundry/bench/benchmark_run_registry_v1.json`

Keep these invariant by default:

- prior dump path
- benchmark bundle path
- `nanoTabPFN` helper settings
- device class
- prior-trained experiment family: `cls_benchmark_staged_prior`
- history, checkpoint, and benchmark artifact contracts

Use cached `nanotabpfn_curve.jsonl` only when the comparison surface is
unchanged.

## Experimentation

### What you can do

Modify any files in src/tab_foundry/model/, src/tab_foundry/preprocessing/ to implement a new mechanism.  

### What you cannot do

- Do not change the prior dump path or benchmark bundle path unless the research package explicitly calls for it and the benchmark conclusion calls it out.
- Do not change the `nanoTabPFN` helper settings, device class, or prior-trained experiment family unless the research package explicitly calls for it and the benchmark conclusion calls it out.
- Do not change the history, checkpoint, or benchmark artifact contracts unless the research package explicitly calls for it and the benchmark conclusion calls it out.
- Do not change any files in src/tab_foundry/training/ or src/tab_foundry/bench/ without explicit research and benchmark justification.

### Simplicity Bias

Prefer simple, targeted changes that are easy to attribute to the mechanism under test. Avoid broad, sweeping changes that make it difficult to isolate the effect of the mechanism.

## Required Research Package

Before any empirical run for a stage, create:

- `outputs/staged_ladder/research/<stage>/research_card.md`
- `outputs/staged_ladder/research/<stage>/campaign.yaml`

Use `reference/stage_campaign_template.md` and
`reference/stage_research_sources.yaml`.

`research_card.md` must state:

- the mechanism change
- why it should help or hurt at the current compact scale
- what exact baseline settings should be preserved
- what settings should shift, and why
- direct evidence used for that recommendation
- one prediction: `preserve baseline settings`, `directional shift`, or `uncertain`

`campaign.yaml` must state:

- `stage`
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

For each stage branch:

1. Write the research package.
2. Run one confirmatory prior-trained branch with the research-recommended
   recipe via `scripts/train_tabfoundry_staged_prior.py`.
3. Benchmark it via `scripts/benchmark_nanotabpfn.py`.
4. If the result is stable but inconclusive, perform bounded local refinement
   only inside `tunable_params`.
5. Promote at most two promising variants to full-benchmark candidates.
6. Register each benchmark-facing candidate via `scripts/register_benchmark_run.py`.

Bounded local refinement is allowed to search only inside the neighborhood
declared in `campaign.yaml`. Do not broaden the search space ad hoc.

## Decisions

Use these branch decisions:

- `keep`: a full-benchmark candidate is near-neutral or better on final ROC AUC
  versus the exact anchor and does not materially worsen late drift
- `defer`: the branch is stable but evidence is mixed or the benefit is too
  narrow to promote yet
- `reject`: the branch is clearly worse across multiple benchmark-facing
  attempts or is genuinely unstable

Never reject a stage from one run alone.

## Scaling Confirmation

Only `keep` stages enter scaling confirmation.

For a kept stage:

1. Run the exact control `01_nano_exact_md_prior_budget1250`.
2. Run the winning stage recipe at `1250` and `2500` steps.
3. Benchmark both on `src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json`.
4. Compare stage versus exact at both budgets.

Promote the stage only if it is at least neutral or better at both budgets and
the gain does not disappear at `2500` steps.

If it helps at one budget but not the other, mark it `defer` and keep
`01_nano_exact_md_prior_parity_fix` as the active base.

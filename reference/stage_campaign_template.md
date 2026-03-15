# Stage Campaign Template

Use one copy of this template per staged architecture branch under:

- `outputs/staged_ladder/research/<stage>/research_card.md`
- `outputs/staged_ladder/research/<stage>/campaign.yaml`

Keep the research card short. The goal is to produce a settings recommendation,
not a long literature review. Use `reference/stage_research_sources.yaml` as
the pinned starting corpus.

## Research Card

### Stage

- `stage`: `<label_token|shared_norm|prenorm_block|...>`
- `anchor_run_id`: `01_nano_exact_md_prior_parity_fix`
- `prediction`: `<preserve baseline settings|directional shift|uncertain>`

### Mechanism

2-4 sentences on what the stage changes and why that should matter at the
current compact model size.

### Evidence

- `tab-foundry`: what local staged results already say
- `nanochat`: what small-transformer evidence transfers cleanly
- `nanoTabPFN` / curated externals: what should dominate if local evidence is weak

### Recommended Settings

- `preserve`: settings that should stay at the exact baseline values
- `shift`: settings that should move, with direction and rationale
- `tunable`: the small neighborhood that bounded local refinement is allowed to explore

### Red Flags

List failure modes that would count as meaningful evidence against the stage.

## campaign.yaml

```yaml
stage: label_token
anchor_run_id: 01_nano_exact_md_prior_parity_fix
decision_hypothesis: directional_shift
full_budget_steps: 2500
full_bundle_path: src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json
recommended_recipe:
  experiment: cls_benchmark_staged_prior
  model:
    stage: label_token
    input_normalization: train_zscore_clip
  optimizer:
    min_lr: 0.004
    weight_decay: 0.0
  runtime:
    grad_clip: 1.0
    max_steps: 2500
preserved_settings:
  model.input_normalization: train_zscore_clip
  optimizer.betas: [0.9, 0.999]
  runtime.eval_every: 25
  runtime.checkpoint_every: 25
  runtime.max_steps: 2500
shifted_settings:
  optimizer.min_lr: 0.004
tunable_params:
  optimizer.min_lr:
    kind: float
    low: 0.003
    high: 0.005
    scale: log
  runtime.grad_clip:
    kind: float
    values: [0.75, 1.0, 1.25]
  optimizer.weight_decay:
    kind: float
    values: [0.0, 0.00001, 0.00005, 0.0001]
```

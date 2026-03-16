# System Delta Package Template

Use one copy of this template per queue row under:

- `outputs/staged_ladder/research/<sweep_id>/<delta_id>/research_card.md`
- `outputs/staged_ladder/research/<sweep_id>/<delta_id>/campaign.yaml`
- `outputs/staged_ladder/research/<sweep_id>/<delta_id>/result_card.md`

The canonical queue source of truth is
`reference/system_delta_sweeps/<sweep_id>/queue.yaml`, backed by
`reference/system_delta_catalog.yaml`. The top-level alias
`reference/system_delta_queue.yaml` is the generated active-sweep view. These
files explain and interpret one row; they do not redefine the row.

## research_card.md

### Delta

- `delta_id`
- `sweep_id`
- `dimension_family`
- `family`
- `anchor_run_id`
- `comparison_policy: anchor_only`
- `locked_bundle_path`
- `locked_control_baseline_id`

### What Changes

- exact change versus upstream `nanoTabPFN`
- exact change versus the locked anchor
- touched subsystems
- explicit preserved settings

### Why This Row Is Informative

- why this row is entering now
- what signal would count as informative
- what would still be ambiguous after one run

### Adequacy Plan

- subsystem-specific knobs that may need bounded follow-up
- what would make a negative result still ambiguous
- what follow-up is allowed before any reject decision

## campaign.yaml

```yaml
sweep_id: binary_md_v1
delta_id: delta_row_cls_pool
dimension_family: model
family: row_pool
comparison_policy: anchor_only
anchor_run_id: 01_nano_exact_md_prior_parity_fix_binary_medium_v1
locked_bundle_path: src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json
locked_control_baseline_id: cls_benchmark_linear_v2
training_experiment: cls_benchmark_staged_prior
preserved_settings:
  model.arch: tabfoundry_staged
  model.input_normalization: train_zscore_clip
  queue_ref: reference/system_delta_sweeps/binary_md_v1/queue.yaml
  preprocessing.impute_missing: true
changed_settings:
  model.stage_label: delta_row_cls_pool
  model.module_overrides:
    row_pool: row_cls
  model.norm_type: layernorm
  training.surface_label: prior_constant_lr
  training.overrides: {}
adequacy_knobs:
  - model.tfrow_n_heads
  - model.tfrow_n_layers
  - model.tfrow_cls_tokens
bounded_followup_policy:
  allow_followups: true
  max_followup_runs: 3
decision_hypothesis: needs_followup
```

## result_card.md

Every completed row needs a `result_card.md` before queue validation should
pass, and the associated run must expose a `training_surface_record.json`
artifact.

Required sections:

- what changed
- measured metrics versus the anchor
- whether the change was actually isolated
- whether introduced hyperparameters were adequate
- why the change may have helped or hurt
- remaining confounders
- recommended next action: `accept_signal`, `needs_followup`, or `unambiguously_worse`

Underperformance alone is not enough for `reject`. The result card must make
that reasoning explicit.

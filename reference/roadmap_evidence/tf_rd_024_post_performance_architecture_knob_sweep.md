# TF-RD-024: Post-Performance Architecture-Knob Sweep On The Classification-First Sandwich Target

This is the canonical long-form evidence note for
[TF-RD-024](../../docs/development/roadmap.md#tf-rd-024-post-performance-architecture-knob-sweep-on-the-classification-first-sandwich-target).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows
  [TF-RD-022](tf_rd_022_training_runtime_vram_efficiency.md) on the closed
  TF-RD-010 classification benchmark contract, and precedes
  [TF-RD-009](tf_rd_009_scaling_law_measurement.md); TF-RD-021 and dagzoo
  RD-002 / RD-005 remain sidecar context rather than blockers

## External Evidence

- Dedicated literature on these exact sandwich-local knobs is lighter than the
  scaling-law or runtime literature, so the next sources to curate are:
  - Perceiver-family references for latent-count and latent-refinement tradeoffs
  - tabular Perceiver-style papers for readout-capacity and summary-bandwidth
    choices
  - compact-transformer ablation references where head count, FF expansion, or
    summary bandwidth were read as bounded architecture knobs rather than
    scaling-law dimensions

## Repo-Local Evidence

- issue [#233](https://github.com/bensonlee5/tab-foundry/issues/233) is the
  umbrella for this bounded post-performance architecture lane
- sweep `tf_rd_024_classification_knob_sweep_v1` is drafted on the closed
  TF-RD-010 medium benchmark contract and inherits
  `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1`
- the sweep reuses historical TF-RD-021B sandwich delta families rather than
  introducing a separate architecture-search path
- every drafted row is currently blocked on the TF-RD-022 keep anchor so the
  first execution can happen on one explicit inherited runtime surface

## Current Interpretation

- TF-RD-024 is a bounded post-performance follow-up, not a scaling-law program
- the live knob set is:
  - `head_hidden_dim`
  - `sandwich_summary_tokens_per_axis`
  - `sandwich_latents`
  - `sandwich_heads`
  - `sandwich_ff_expansion`
  - `sandwich_self_attention_per_cross`
- the medium benchmark rung is the screening stage; any keep-worthy signal must
  still validate on the closed TF-RD-010 large rung before promotion
- keep the runtime policy inherited from TF-RD-022 fixed across every row
- keep these dimensions out of scope here:
  - `d_icl`
  - `sandwich_layers`
  - batch size
  - LR
  - clipping
  - optimizer-family or other training-dynamics changes

## Open Evidence Gaps

- the TF-RD-022 keep anchor is not yet closed, so no TF-RD-024 rows should run
  yet
- the repo does not yet have medium-rung execution results on the bounded knob
  set under the inherited TF-RD-022 policy
- the repo does not yet have large-rung validation for any keep-worthy
  post-performance architecture signal

## Exit Signals

- the repo has one explicit keep/defer decision on the bounded post-performance
  sandwich knob set under the inherited TF-RD-022 runtime policy
- TF-RD-009 can freeze the remaining non-scaling architecture knobs and proceed
  on the inherited benchmark and runtime contract

# TF-RD-024: Post-Performance Architecture-Knob Sweep On The Classification-First Sandwich Target

This is the canonical long-form evidence note for
[TF-RD-024](../../docs/development/roadmap.md#tf-rd-024-post-performance-architecture-knob-sweep-on-the-classification-first-sandwich-target).

- Status: `completed`
- Milestone: `Complete`
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
  umbrella for this bounded post-performance architecture lane and closes on
  the medium-only closeout recorded here
- sweep `tf_rd_024_classification_knob_sweep_v1` executed on the closed
  TF-RD-010 medium benchmark contract and inherited
  `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1`
- that inherited experiment now carries the kept TF-RD-022 runtime surface:
  `mixed_precision=bf16`, `trace_activations=false`,
  `activation_checkpointing=true`, `compile_model=true`,
  `compile_backend=eager`, and `compile_dynamic=true`
- the fresh compile-eager-dynamic anchor is now benchmarked on both closed
  TF-RD-010 rungs at medium `final_log_loss=0.6820309591` and large
  `final_log_loss=0.9298541427`
- the completed seven-row medium screen produced the following results against
  the fresh compile anchor:
  - `sandwich_latents=12`: `0.7050649599`
  - `sandwich_heads=2`: `0.6762878243`
  - `sandwich_ff_expansion=1`: `0.7113842731`
  - `sandwich_summary_tokens_per_axis=1`: `0.6940640479`
  - `sandwich_self_attention_per_cross=1`: `0.7173978046`
  - `head_hidden_dim=64`: `0.6912802875`
  - `head_hidden_dim=128`: `0.6834016822`
- only `sandwich_heads=2` beat the anchor in the first bounded medium screen
- one explicit two-row medium-only follow-up then ran as a separate sweep:
  - `sandwich_heads=1`: `0.6603575333`
  - `sandwich_pre_row_attention_layers=2`: `0.6780725432`
- `sandwich_heads=1` is now the best medium result in this family
- the sweep reused historical TF-RD-021B sandwich delta families where
  possible rather than introducing a separate architecture-search path

## Current Interpretation

- TF-RD-024 was a bounded post-performance follow-up, not a scaling-law program
- the live knob set is:
  - `head_hidden_dim`
  - `sandwich_summary_tokens_per_axis`
  - `sandwich_latents`
  - `sandwich_heads`
  - `sandwich_ff_expansion`
  - `sandwich_self_attention_per_cross`
- medium was the screening stage for this epic, but the final closeout is
  intentionally medium-only by decision
- keep the runtime policy inherited from TF-RD-022 fixed across every row:
  `mixed_precision=bf16`, `trace_activations=false`,
  `activation_checkpointing=true`, `compile_model=true`,
  `compile_backend=eager`, `compile_dynamic=true`
- the completed medium screen separates three interpretations:
  - most downward capacity ablations were clearly harmful, so those axes remain
    active capacity rather than slack on the current benchmark
  - reducing `sandwich_heads` from `4` to `2` improved the medium result, and
    the follow-up showed the gain extends further to `sandwich_heads=1`
  - increasing `sandwich_pre_row_attention_layers` from `1` to `2` improved
    over the anchor but was not strong enough to replace the lower-head winner
- keep these dimensions out of scope here:
  - `d_icl`
  - `sandwich_layers`
  - batch size
  - LR
  - clipping
  - optimizer-family or other training-dynamics changes

## Closeout Read

- TF-RD-024 closes on medium-only evidence by decision; large validation was
  intentionally skipped rather than left pending
- `sandwich_heads=1` is the carry-forward winner for TF-RD-009
- `sandwich_pre_row_attention_layers=2` is retained only as positive-but-not-
  promoted evidence
- any remaining question about upward movement on other capacity axes now moves
  into TF-RD-009 or later post-scale follow-up, not back into TF-RD-024

## Exit Signals

- satisfied: the repo has one explicit keep/defer decision on the bounded
  post-performance sandwich knob set under the inherited TF-RD-022 runtime
  policy
- satisfied: TF-RD-009 can now freeze the remaining non-scaling architecture
  knobs around `sandwich_heads=1` and proceed on the inherited benchmark and
  runtime contract

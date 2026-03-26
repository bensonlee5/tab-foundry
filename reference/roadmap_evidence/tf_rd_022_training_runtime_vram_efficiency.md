# TF-RD-022: Training Runtime And VRAM Efficiency On The Promoted Anchor

This is the canonical long-form evidence note for
[TF-RD-022](../../docs/development/roadmap.md#tf-rd-022-training-runtime-and-vram-efficiency-on-the-promoted-anchor).

- Status: `planned`
- Milestone: `Next`
- Dependency position: runs as a sibling to
  [TF-RD-018](tf_rd_018_training_surface_adequacy.md); it should not reopen
  optimizer or LR adequacy, but it should hand one explicit runtime policy back
  to TF-RD-018, the deferred CUDA-capacity line, and later
  [TF-RD-009](tf_rd_009_scaling_law_measurement.md) preparation

## External Evidence

- Dedicated external literature is not yet curated for this epic.
- Sources to curate next:
  - PyTorch AMP and activation-checkpointing guidance for bf16 and memory-speed
    tradeoffs on A100-class hardware
  - reproducibility references for throughput and CUDA-memory telemetry in
    training loops
  - any tabular-model case studies where runtime-policy changes altered the
    practical batch or scaling frontier without changing architecture

## Repo-Local Evidence

- [#58](https://github.com/bensonlee5/tab-foundry/issues/58) already exists as
  the deferred runtime or VRAM measurement follow-up, but it stayed attached to
  the closed TF-RD-002 measurement epic and never expanded into a full runtime
  tuning lane
- new epic [#168](https://github.com/bensonlee5/tab-foundry/issues/168) now
  tracks runtime and VRAM efficiency end to end, with child issues
  [#169](https://github.com/bensonlee5/tab-foundry/issues/169),
  [#170](https://github.com/bensonlee5/tab-foundry/issues/170), and
  [#171](https://github.com/bensonlee5/tab-foundry/issues/171)
- bounded sidecar issues [#174](https://github.com/bensonlee5/tab-foundry/issues/174)
  and [#175](https://github.com/bensonlee5/tab-foundry/issues/175) now track a
  non-canonical `tabfoundry_sandwich` simplification screen; the companion note
  is [TF-RD-021A](tf_rd_021a_latent_bank_sandwich_prototype.md)
- canonical benchmark prior configs still inherit `runtime.mixed_precision: "no"` from `configs/experiment/_shared/compact_binary_prior.yaml` unless a
  higher-level experiment overrides it
- canonical training telemetry records loss, gradient, and instability
  summaries, but it does not yet expose peak CUDA memory or throughput
  summaries
- `tabfoundry_staged` already supports `runtime.activation_checkpointing`, and
  the benchmark-facing runtime defaults currently keep it disabled
- benchmark-facing exact-prior experiments still enable
  `runtime.trace_activations: true`, which is useful for diagnostics but not yet
  separated cleanly from ordinary benchmark-facing execution

## Current Interpretation

- the highest-probability low-risk win is to make runtime policy explicit and
  measurable before chasing larger architecture or optimizer changes for speed
- the hard-surface decision anchor for this epic must stay CUDA-only; MPS OOMs
  are useful for local iteration but should not be mixed into A100 memory
  conclusions
- TF-RD-022 should treat `bf16`, benchmark-facing activation-trace policy, and
  activation checkpointing as the first bounded runtime knobs on the current
  harder-surface training path
- TF-RD-021A can run in parallel as a bounded simplification screen, but it
  should start on nanoTabPFN prior-dump data, only promote onto dagzoo after
  the fast screens settle viable latent or width rows, and it does not replace
  the runtime-telemetry prerequisite for CUDA decisions
- activation checkpointing is primarily a headroom tool, not a default speed
  tool; prefer it only if the non-checkpointed rows still leave inadequate VRAM
  margin
- once runtime policy is explicit, the next decision-relevant read is not a new
  architecture sweep but a bounded reopen of harder-surface batching under a
  conservative OOM guardrail

## Open Evidence Gaps

- the repo still lacks canonical peak-VRAM and throughput summaries in
  `telemetry.json`, sweep summaries, and result cards
- the repo still lacks one explicit keep or defer decision on whether bf16 is
  benchmark-safe on the current harder-surface carried recipe
- the repo still lacks one explicit keep or defer decision on benchmark-facing
  activation tracing versus screen-only tracing
- the repo still lacks one measured reopen rule for `task_batch_size=2` or `4`
  on the harder carry-forward surface under an 80 GB A100 memory budget

## Exit Signals

- one explicit runtime policy exists for the promoted-anchor harder surface,
  justified by repo-local time and VRAM evidence
- sweep artifacts expose runtime and VRAM summaries compactly enough to compare
  future runs without manual log inspection
- TF-RD-018, the deferred CUDA-capacity line, and TF-RD-009 can inherit the
  same runtime policy and batching keep or stop rule without re-deriving them

# TF-RD-022: Training Runtime And VRAM Efficiency Before Classification Scaling

This is the canonical long-form evidence note for
[TF-RD-022](../../docs/development/roadmap.md#tf-rd-022-training-runtime-and-vram-efficiency-before-classification-scaling).

- Status: `planned`
- Milestone: `Next`
- Dependency position: runs after the first carried sandwich dagzoo many-class
  slice and the full TF-RD-021 carry-forward decision are explicit, and before
  TF-RD-009 scaling fits; it should not reopen sandwich-parent or regime-choice
  work, but it should hand one explicit kernel/runtime policy back to later
  scaling work

## External Evidence

- Dedicated runtime-policy literature is still lighter than the scaling-law
  note, but the main references to curate next are:
  - PyTorch AMP guidance for bf16 on A100-class hardware
  - activation-checkpointing references for memory-speed tradeoffs
  - reproducibility references for throughput and CUDA-memory telemetry in
    training loops
  - any tabular-model case studies where runtime-policy changes altered the
    practical batch frontier without changing architecture

## Repo-Local Evidence

- [#58](https://github.com/bensonlee5/tab-foundry/issues/58) already existed as
  the deferred runtime or VRAM measurement follow-up, but it stayed attached to
  the closed TF-RD-002 measurement epic and never became a full runtime lane
- epic [#168](https://github.com/bensonlee5/tab-foundry/issues/168) tracks
  runtime and VRAM efficiency end to end, with child issues
  [#169](https://github.com/bensonlee5/tab-foundry/issues/169),
  [#170](https://github.com/bensonlee5/tab-foundry/issues/170), and
  [#171](https://github.com/bensonlee5/tab-foundry/issues/171)
- sandwich architecture ownership now lives under the historical
  implementation record [#174](https://github.com/bensonlee5/tab-foundry/issues/174),
  umbrella issue [#178](https://github.com/bensonlee5/tab-foundry/issues/178),
  and active simplification follow-up
  [#184](https://github.com/bensonlee5/tab-foundry/issues/184); TF-RD-022 is a
  dependency surface for later classification work, not the owner of sandwich
  planning
- training telemetry and benchmark-registry records now preserve:
  - `peak_vram_allocated`
  - `peak_vram_reserved`
  - `throughput_examples_per_second`
  - `throughput_tokens_per_second`
  - `tokens_per_step`
  - `tokens_seen`
  - `token_budget`
  - `unique_task_budget`
  - `objective_metric`
  - `curriculum_id`
  - `curriculum_mix`
  - `scm_complexity_summary`
- canonical benchmark prior configs still inherit `runtime.mixed_precision: "no"` from `configs/experiment/_shared/compact_binary_prior.yaml` unless a
  higher-level experiment overrides it
- the shared runtime surface already supports `runtime.activation_checkpointing`,
  and benchmark-facing exact-prior runs still default to
  `runtime.trace_activations: true`

## Current Interpretation

- TF-RD-022 is now a hard pre-scaling gate rather than an optional adjacent
  runtime-cleanup lane
- the highest-probability low-risk win is still to make runtime policy explicit
  and measurable before chasing larger architecture or optimizer changes for
  speed
- the runtime ladder should stay classification-only and should inherit one
  frozen carried recipe rather than reopening sandwich-parent selection, the
  upstream dagzoo surface expansion, or law design
- the bounded runtime knobs remain:
  - `bf16`
  - benchmark-facing activation-trace policy
  - activation checkpointing
- include low-level kernel tuning only to the extent needed to keep the
  carried sandwich dagzoo slice reliable and efficient enough for scaling
- batching reopens only after those reads and only under an explicit 80 GB A100
  guardrail

## Open Evidence Gaps

- sweep and result summaries still need compact presentation of the new runtime
  and regime-budget fields
- the repo still lacks one explicit keep/defer decision on whether bf16 is
  benchmark-safe on the carried classification recipe
- the repo still lacks one explicit keep/defer decision on benchmark-facing
  activation tracing versus screen-only tracing
- the repo still lacks one measured reopen rule for `task_batch_size=2` or `4`
  on the carried harder-surface classification recipe under an 80 GB A100
  budget

## Exit Signals

- one explicit benchmark-safe runtime policy exists for the classification
  scaling target, justified by repo-local time and VRAM evidence
- artifacts and summaries expose runtime and VRAM metrics compactly enough to
  compare future runs without manual log inspection
- later scaling, deferred CUDA-capacity follow-up, and TF-RD-009 can inherit
  the same runtime policy and batching keep/stop rule without
  re-deriving them

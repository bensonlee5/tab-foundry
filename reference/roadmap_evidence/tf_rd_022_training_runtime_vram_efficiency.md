# TF-RD-022: Training Runtime And VRAM Efficiency Before Classification Scaling

This is the canonical long-form evidence note for
[TF-RD-022](../../docs/development/roadmap.md#tf-rd-022-training-runtime-and-vram-efficiency-before-classification-scaling).

- Status: `partial`
- Milestone: `Next`
- Dependency position: runs after the closed TF-RD-010 classification
  benchmark contract is explicit, and before
  [TF-RD-024](tf_rd_024_post_performance_architecture_knob_sweep.md) and
  [TF-RD-009](tf_rd_009_scaling_law_measurement.md); it should not reopen
  sandwich-parent or regime-choice work, and it is not blocked by TF-RD-021 or
  dagzoo RD-002 / RD-005

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
- `tab-foundry dev run-inspect`, sweep summary output, and result-card reporting
  now expose compact `runtime_summary` and `regime_budget` sections directly
- the repo now has a named runtime policy surface at
  `configs/runtime/tf_rd_022_policy.yaml` plus the inherited benchmark-facing
  experiment `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1`
- issue [#233](https://github.com/bensonlee5/tab-foundry/issues/233) is the
  downstream TF-RD-024 consumer that will inherit the kept TF-RD-022 policy

## Current Interpretation

- TF-RD-022 is now a hard pre-scaling gate rather than an optional adjacent
  runtime-cleanup lane
- TF-RD-022 runs directly on the closed TF-RD-010 benchmark contract; TF-RD-021
  and dagzoo RD-002 / RD-005 remain sidecar context only
- the highest-probability low-risk win is still to make runtime policy explicit
  and measurable before chasing larger architecture or optimizer changes for
  speed
- the runtime ladder should stay classification-only and should inherit one
  frozen recipe rather than reopening sandwich-parent selection, synthetic
  surface expansion, or law design
- the bounded runtime knobs remain:
  - `bf16`
  - benchmark-facing activation-trace policy
  - activation checkpointing
- use the existing medium benchmark rung as the fast screening stage, then
  validate the kept policy on the closed TF-RD-010 medium and large targets
- include low-level kernel tuning only to the extent needed to keep the
  inherited sandwich benchmark contract reliable and efficient enough for
  scaling
- batching reopens only after those reads and only under an explicit 80 GB A100
  guardrail

## Open Evidence Gaps

- the repo still lacks one explicit keep/defer decision on whether bf16 is
  benchmark-safe on the carried classification recipe
- the repo still lacks one explicit keep/defer decision on benchmark-facing
  activation tracing versus screen-only tracing
- the repo still lacks one explicit keep/defer decision on activation
  checkpointing on the inherited classification recipe
- the repo still lacks one measured reopen rule for `task_batch_size=2` or `4`
  on the inherited harder-surface classification recipe under an 80 GB A100
  budget

## Exit Signals

- one explicit benchmark-safe runtime policy exists for the classification
  scaling target, justified by repo-local time and VRAM evidence
- artifacts and summaries expose runtime and VRAM metrics compactly enough to
  compare future runs without manual log inspection
- later TF-RD-024 architecture work, deferred CUDA-capacity follow-up, and
  TF-RD-009 can inherit the same runtime policy and batching keep/stop rule
  without re-deriving them

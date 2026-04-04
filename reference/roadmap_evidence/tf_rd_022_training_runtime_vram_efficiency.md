# TF-RD-022: Performance Optimization On The Settled Sandwich Runtime Surface Before Classification Scaling

This is the canonical long-form evidence note for
[TF-RD-022](../../docs/development/roadmap.md#tf-rd-022-performance-optimization-on-the-settled-sandwich-runtime-surface-before-classification-scaling).

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

- completed historical issues [#58](https://github.com/bensonlee5/tab-foundry/issues/58),
  [#169](https://github.com/bensonlee5/tab-foundry/issues/169), and
  [#170](https://github.com/bensonlee5/tab-foundry/issues/170) now record the
  runtime-summary instrumentation, bounded medium ladder, and first-class
  runtime-policy surface that made TF-RD-022 explicit enough for downstream
  work; issue [#171](https://github.com/bensonlee5/tab-foundry/issues/171) is
  superseded because TF-RD-022 will not reopen harder-surface batching
- epic [#168](https://github.com/bensonlee5/tab-foundry/issues/168) now tracks
  performance optimization on the settled runtime surface end to end, with
  child issues [#239](https://github.com/bensonlee5/tab-foundry/issues/239),
  [#240](https://github.com/bensonlee5/tab-foundry/issues/240), and
  [#241](https://github.com/bensonlee5/tab-foundry/issues/241)
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
- sweep `tf_rd_022_runtime_policy_medium_v1` now records the intended
  benchmark-first medium runtime ladder on the closed TF-RD-010 control
  contract:
  - row 1 replays the no-AMP control
  - row 2 isolates bf16
  - row 3 isolates benchmark-facing activation tracing on top of bf16
  - row 4 isolates activation checkpointing on top of bf16
- the completed medium ladder now records one explicit keep/defer read for each
  runtime knob:
  - row 1 control `sd_tf_rd_022_runtime_policy_medium_v1_01_delta_tf_rd_022_cls_runtime_control_noamp_v1_v4`
    is the benchmark-safe no-AMP reference at `final_log_loss=0.6849302248`,
    `peak_vram_reserved=27084718080`, and `throughput_tokens_per_second=154192.6072`
  - row 2 bf16 `sd_tf_rd_022_runtime_policy_medium_v1_02_delta_tf_rd_022_cls_runtime_bf16_v1_v1`
    is benchmark-safe but deferred at `final_log_loss=0.6818472858`,
    `peak_vram_reserved=5312086016`, and `throughput_tokens_per_second=157185.8660`
  - row 3 trace `sd_tf_rd_022_runtime_policy_medium_v1_03_delta_tf_rd_022_cls_runtime_trace_v1_v2`
    is benchmark-safe but diagnostic-only at `final_log_loss=0.6754785052`,
    `peak_vram_reserved=5398069248`, and `throughput_tokens_per_second=151347.0572`
  - row 4 checkpointing `sd_tf_rd_022_runtime_policy_medium_v1_04_delta_tf_rd_022_cls_runtime_checkpoint_v1_v2`
    is the medium-rung winner at `final_log_loss=0.6765953232`,
    `peak_vram_reserved=3321888768`, and `throughput_tokens_per_second=150561.1995`
- that sweep uses a benchmark-first keep bar: a row is eligible only if
  `final_log_loss_at_matched_regime_budget` is non-worse than the no-AMP
  control, with `peak_vram_reserved`, `throughput_tokens_per_second`, and
  `non_train_overhead_seconds` used only as tie-breakers among benchmark-safe
  rows
- the carried TF-RD-022 runtime surface is now fixed to the measured medium
  winner: `mixed_precision=bf16`, `trace_activations=false`, and
  `activation_checkpointing=true`
- benchmark throughput is still the most credible remaining local win because
  medium benchmarking now takes more than an hour, the evaluator in
  `src/tab_foundry/bench/openml_benchmark/metrics.py` is fully serial, and the
  current medium manifest has repeated task signatures that can plausibly share
  batched inference on the existing sandwich surface
- training throughput still has bounded local headroom because the current
  loader, device-transfer, and runtime defaults in
  `src/tab_foundry/data/factory.py`,
  `src/tab_foundry/task_batching.py`, and
  `src/tab_foundry/training/runtime.py` are conservative rather than
  aggressively overlapped
- corpus materialization throughput remains worth a measured pass because the
  workflow is slow in practice and the local orchestration path in
  `src/tab_foundry/data/corpus_materialization_shared.py` still starts from a
  fixed process cap that may or may not be the dominant bottleneck
- issue [#233](https://github.com/bensonlee5/tab-foundry/issues/233) is the
  downstream TF-RD-024 consumer that will inherit the kept TF-RD-022 runtime
  policy after TF-RD-022 closes its performance follow-up work

## Current Interpretation

- TF-RD-022 is now a hard pre-scaling gate rather than an optional adjacent
  runtime-cleanup lane
- TF-RD-022 runs directly on the closed TF-RD-010 benchmark contract; TF-RD-021
  and dagzoo RD-002 / RD-005 remain sidecar context only
- runtime-policy selection is complete enough for downstream planning: the
  carried TF-RD-022 surface is `bf16` with activation tracing off and
  activation checkpointing on
- the remaining TF-RD-022 work is bounded speed optimization on top of that
  settled runtime surface rather than more runtime-policy selection
- benchmark execution is the highest-priority remaining lane because the medium
  benchmark runtime is already operationally expensive and the current
  evaluation path is serial
- training speed remains in scope because the codepath still leaves obvious
  data-pipeline and transfer-overlap questions unanswered, but it should be
  approached with profiling first and low-risk execution changes second
- corpus materialization remains in scope because it is slow in practice, but
  the first question there is bottleneck attribution between local
  orchestration and upstream `tab-realdata-hub` or dagzoo work
- TF-RD-022 should not reopen sandwich-parent selection, larger architecture
  changes, law-design work, or harder-surface batching while those three speed
  lanes are still unresolved

## Open Evidence Gaps

- the repo still lacks one explicit measured keep/defer decision for training
  throughput on the settled runtime surface under issue
  [#239](https://github.com/bensonlee5/tab-foundry/issues/239)
- the repo still lacks one explicit measured keep/defer decision for medium
  benchmark execution speed under issue
  [#240](https://github.com/bensonlee5/tab-foundry/issues/240)
- the repo still lacks one explicit measured keep/defer decision for corpus
  materialization throughput, including local-versus-upstream bottleneck
  attribution, under issue
  [#241](https://github.com/bensonlee5/tab-foundry/issues/241)

## Exit Signals

- one explicit benchmark-safe runtime policy exists for the classification
  scaling target, justified by repo-local time and VRAM evidence
- the repo has one explicit measured keep/defer outcome for training
  throughput, medium benchmark throughput, and corpus materialization
  throughput on that settled runtime surface
- artifacts and summaries expose runtime and timing reads compactly enough to
  compare future runs without manual log inspection
- later TF-RD-024 architecture work and TF-RD-009 can inherit the same runtime
  policy and closed performance gate without re-deriving them

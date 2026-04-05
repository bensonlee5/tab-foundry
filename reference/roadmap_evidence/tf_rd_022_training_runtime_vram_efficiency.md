# TF-RD-022: Performance Optimization On The Settled Sandwich Runtime Surface Before Classification Scaling

This is the canonical long-form evidence note for
[TF-RD-022](../../docs/development/roadmap.md#tf-rd-022-performance-optimization-on-the-settled-sandwich-runtime-surface-before-classification-scaling).

- Status: `completed`
- Milestone: `Completed`
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
  [#241](https://github.com/bensonlee5/tab-foundry/issues/241) now all
  completed
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
- issue [#239](https://github.com/bensonlee5/tab-foundry/issues/239) now
  records one explicit bounded training-throughput read on that settled runtime
  surface:
  - the named TF-RD-022 baseline experiment is now manifest-backed on
    `tf_rd_010_dagzoo_medium_control` /
    `tf_rd_010_dagzoo_medium_control_curated_v5` and pinned to CUDA rather than
    inheriting `legacy_prior`
  - the measured candidate only changed loader and transfer behavior:
    `num_workers=2`, `loader_pin_memory=true`,
    `loader_persistent_workers=true`, `loader_prefetch_factor=2`, and
    `non_blocking_device_transfer=true`
  - the completed same-host CUDA benchmark replay improved training time from
    `best_training_time=6117.0161` to `3429.1443` and
    `final_training_time=6244.0331` to `3456.8260`, but benchmark quality
    drifted the wrong way at `best_roc_auc=0.6619213 -> 0.6592971`,
    `best_log_loss=0.5339507 -> 0.5346940`,
    `best_brier_score=0.3631727 -> 0.3636804`, and
    `best_bpc=2.1102889 -> 2.1123012`
  - a follow-up same-host CUDA decomposition screen at `runtime.max_steps=24`
    ranked the split variants `baseline=68.6341s`, `workers=57.1585s`,
    `loader_overlap=55.6515s`, `transfer=50.8132s`, and
    `combined=50.9901s`, so the full replay advanced `transfer` and
    `loader_overlap`
  - the full same-host CUDA replay on those two split variants preserved the
    large speedup but still failed the strict benchmark-safe keep bar:
    `transfer` reached `best_training_time=3081.4472`,
    `final_training_time=3110.9282`, `best_roc_auc=0.6584391`,
    `best_log_loss=0.5329700`, `best_brier_score=0.3622467`,
    `best_bpc=2.1101876`; `loader_overlap` reached
    `best_training_time=3063.8761`, `final_training_time=3127.7445`,
    `best_roc_auc=0.6608182`, `best_log_loss=0.5358135`,
    `best_brier_score=0.3647582`, `best_bpc=2.1100665`; the same-host
    baseline was `best_training_time=5142.3102`,
    `final_training_time=5373.6851`, `best_roc_auc=0.6608562`,
    `best_log_loss=0.5353731`, `best_brier_score=0.3642339`,
    `best_bpc=2.1077142`
  - TF-RD-022 therefore records `#239` as a decomposition-backed measured
    defer on the low-risk overlap and copy path, and the carried runtime
    policy remains unchanged
- benchmark helper entrypoints now resolve checkpoint paths without eagerly
  importing the full training stack, which removed an incidental `omegaconf`
  dependency from the TabICLv2 helper environment and unblocked the official
  CUDA `#239` replay
- benchmark throughput is still the most credible remaining local win because
  medium benchmarking now takes more than an hour, the evaluator in
  `src/tab_foundry/bench/openml_benchmark/metrics.py` is fully serial, and the
  current medium manifest has repeated task signatures that can plausibly share
  batched inference on the existing sandwich surface
- the current loader, device-transfer, and runtime defaults in
  `src/tab_foundry/data/factory.py`,
  `src/tab_foundry/task_batching.py`, and
  `src/tab_foundry/training/runtime.py` remain conservative rather than
  aggressively overlapped, but `#239` has already closed the low-risk
  training-throughput lane as a measured defer on that surface
- `#240` is now completed: the kept batched sandwich checkpoint evaluator
  closes the bounded benchmark-throughput lane on the inherited TF-RD-022
  runtime surface
- `#241` is now completed:
  - invocation summaries and corpus records preserve generate, filter, copy,
    manifest-build, promotion, and aggregate recipe timing for representative
    same-host local-versus-upstream attribution
  - same-host `#241` microbenchmarking on a completed accepted-only baseline
    invocation showed the full-shard promotion step drop from `0.0463s` mean
    to `0.0224s` mean across `5` repetitions (`2.06x` faster), with hardlinks
    confirmed and copied dataset counts unchanged
  - TF-RD-022 therefore keeps the hardlink-backed full-shard promotion path as
    the bounded repo-local materialization win and closes the remaining
    materialization lane by persisting the attribution surfaces needed to show
    whether future wall time is local or upstream
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
- TF-RD-022 is now closed: the bounded speed follow-up work is complete on top
  of the settled runtime surface rather than remaining open as an adjacent
  runtime-policy task
- training speed now has one explicit measured defer on the low-risk
  overlap-and-transfer path, including the completed CUDA decomposition of the
  combined candidate into `workers`, `loader_overlap`, and `transfer`
  variants, so no further training-throughput work remains in TF-RD-022 unless
  a later lane uncovers a new, narrower hypothesis
- corpus materialization now has both a bounded local keep on the accepted-only
  full-shard promotion step and the persisted timing attribution needed to
  separate local orchestration from upstream `tab-realdata-hub` or dagzoo work
- TF-RD-022 should not reopen sandwich-parent selection, larger architecture
  changes, law-design work, or harder-surface batching while those three speed
  lanes are closed

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

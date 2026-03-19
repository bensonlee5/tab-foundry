# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.7] - 2026-03-19

### Changed

- User-facing note: the regular trainer now writes `gradient_history.jsonl`
  and `telemetry.json`, the canonical `cls_benchmark_staged` surface enables
  activation tracing by default, and the persisted telemetry schema is now
  `tab-foundry-training-telemetry-v2`.

- Added regular-trainer parity for the TF-RD-002 stability surface: stage-local
  activation traces and module-gradient summaries now flow through
  `train run`, are summarized into wandb, and surface in result cards and sweep
  matrix rows alongside the existing exact-prior telemetry.

- Added the missing `post_context_encoder` activation trace to staged model
  paths so context-enabled rows now expose column, row, and context activation
  boundaries consistently.

- Corrected activation telemetry aggregation so distributed regular-training
  runs now reconstruct stage-local activation RMS from raw sum-squared/count
  traces across microsteps and accelerator ranks, recursive many-class
  forwards aggregate repeated `post_context_encoder` traces instead of keeping
  only the last call, and `gradient_history.jsonl` distinguishes non-finite
  global grad norms with `global_grad_norm_kind`.

## [0.8.6] - 2026-03-19

### Changed

- User-facing note: `tab-foundry bench smoke dagzoo` now reuses the shared
  dagzoo handoff-manifest workflow instead of owning a separate raw
  `dagzoo generate --out` path, and its smoke telemetry now records the
  validated `dagzoo_handoff` identity plus a combined
  `timings_seconds.dagzoo_generate_manifest` stage.

- Tightened completed-row system-delta queue validation so checked-in
  `benchmark_metrics` must stay consistent with the canonical benchmark run
  registry for registry-backed tab-foundry metrics and anchor deltas.

- Refreshed stale Brier-score fields in the checked-in
  `cuda_stability_followup`, `input_norm_followup`, and
  `input_norm_none_followup` sweep queues so their completed-row
  `benchmark_metrics` match the corrected registry-backed Brier definition.

- Export bundle producer metadata now falls back to the checked-in
  `pyproject.toml` version when editable-install package metadata is missing or
  malformed, keeping manifest validation stable under local `uv` environments.

## [0.8.4] - 2026-03-19

### Changed

- User-facing note: classification benchmark Brier scores now use the summed
  per-example classwise squared error definition, so persisted Brier metrics in
  the checked-in benchmark run registry were refreshed to the corrected scale.

## [0.8.3] - 2026-03-19

### Changed

- User-facing note: `tab-foundry data build-manifest` and
  `data dagzoo generate-manifest` now reject invalid or non-finite split ratios
  before doing work, the dagzoo wrapper preserves dagzoo-root-relative
  handoff/diagnostics paths, rejects non-directory dagzoo roots before launch,
  preserves upstream non-zero exit codes instead of surfacing a Python
  traceback on generate failures, and now rejects stale or copied handoff
  manifests whose `generate_run_id` or `generated_corpus_id` do not match the
  scanned generated corpus.

## [0.8.2] - 2026-03-19

### Changed

- User-facing note: system-delta sweep metadata and generated research packages
  now record the locked training experiment, config profile, and surface role,
  and the runbook/template now makes `screen_only` rows explicitly
  diagnostic-only while reserving benchmark-facing claims for `benchmark_full`
  rows on the cited locked bundle/control-baseline contract.

- Fixed sweep creation so overriding only `training_experiment` now also
  derives the matching `training_config_profile` and `surface_role` instead of
  inheriting contradictory hybrid-lane metadata from the parent sweep.

- Fixed sweep graph rendering to resolve queue rows from the sweep's actual
  `training_experiment`, and classified `cls_benchmark_linear_simple` plus
  `cls_benchmark_linear_simple_prior` sweeps as `pfn_control` instead of
  generic `custom`.

- Fixed PFN-control sweep creation so catalog rows drop staged-only
  `model.stage`, `model.stage_label`, and `model.module_overrides` payloads
  while preserving portable data, preprocessing, and training overrides.

- Clarified and corrected the checked-in `cls_benchmark_linear_v2`
  control-baseline contract: the canonical medium-bundle baseline id is
  preserved, but it now freezes the existing prior-trained staged `nano_exact`
  anchor instead of reusing the older `cls_benchmark_linear_v1` train surface.

- User-facing note: the repo now consistently targets Python `3.14` across
  `requires-python`, Ruff, mypy, `.python-version`, GitHub Actions, and the
  setup/workflow docs.

- Restored the pre-0.8.2 legacy-sweep fallback for manifests missing
  `training_experiment` and `training_config_profile`: execution and graph
  resolution now keep the old `cls_benchmark_staged_prior` default instead of
  trusting synthetic `anchor_context` names.

## [0.8.1] - 2026-03-18

### Added

- Added `tab-foundry research sweep graph`, a torchview-based architecture
  renderer for sweep anchors and selected sweep rows. The command writes SVG
  outputs plus an `index.md` manifest under
  `outputs/staged_ladder/research/<sweep_id>/architecture_graphs` by default.

### Changed

- User-facing note: repo installs now include the Python graph-rendering
  dependencies needed for sweep architecture graphs. Rendering still requires
  the Graphviz `dot` binary to be installed on the machine.

## [0.8.0] - 2026-03-18

### Changed

- Broad user-facing break: removed the legacy `tabfoundry` architecture family
  from supported build, training, evaluation, checkpoint-loading, export, and
  benchmark paths. The active model-development surface is now
  `tabfoundry_staged`, with `tabfoundry_simple` retained only as the frozen
  exact anchor.

- Broad user-facing break: removed regression from the active train/eval/export
  surface. The repo is now classification-only until regression is rebuilt on
  top of `tabfoundry_staged`.

- The canonical model default now resolves to `model.arch=tabfoundry_staged`,
  and legacy `tabfoundry` checkpoints now fail fast with explicit unsupported
  compatibility errors instead of attempting reconstruction.

- Updated the model, workflow, inference, and roadmap documentation to describe
  the single active staged surface and the classification-only contract.

## [0.7.3] - 2026-03-18

### Added

- Added the CUDA-scale experiment profile `cls_benchmark_staged_prior_cuda_scale` for large-anchor exact-prior benchmarking on the locked prenorm plus row-cls bridge surface.

- Added CUDA capacity, stability, and budget system-delta sweep metadata, queues, matrices, catalog entries, and research tests, including completed `cuda_stability_followup` benchmark registry entries and deferred batch64 backlog tracking.

### Changed

- Changed system-delta execution to use queue-aware wandb run IDs and to replace `model.module_overrides` cleanly for row-specific overrides such as `post_encoder_norm`.

- Changed prior-dump training to retry `torch.OutOfMemoryError` batches with smaller microbatches while preserving aggregated loss, accuracy, and activation-trace logging, and switched the active sweep metadata to `cuda_stability_followup`.

- User-facing workflow break: the packaged CLI now uses nested namespaces for
  all supported commands. Canonical surfaces are `tab-foundry data build-manifest`, `train run`, `train prior simple|staged`, `eval checkpoint`, `export bundle|validate`, `bench ...`, and `research sweep ...`.
  The thin Python workflow wrappers under `scripts/` have been removed.

- User-facing note: research metadata now includes the CUDA sweep family, exact-prior prior-dump runs may automatically fall back to smaller microbatches after OOM, and the package version is now `0.7.3`.

## [0.7.2] - 2026-03-18

### Added

- Added OpenML discovery mode to the benchmark-bundle builder, including direct task-list filtering, one-per-dataset deduping with `10-fold Crossvalidation` preference, validation-time rejection reporting, and CLI controls for `--discover-from-openml`, `--min-instances`, and `--min-task-count`.

- Added the opt-in repo-tracked benchmark artifact `src/tab_foundry/bench/nanotabpfn_openml_binary_large_no_missing_v1.json` for larger binary no-missing comparisons, while leaving the existing missing-permitting large bundle in place for missingness-focused sweeps.

### Changed

- User-facing note: the benchmark bundle builder now supports direct OpenML discovery for supervised-classification bundles, and the package version is now `0.7.2`.

## [0.7.1] - 2026-03-18

### Added

- Added system-delta queue and result-card coverage for log-loss-first classification benchmarking, including persisted `final_log_loss`, `final_brier_score`, and anchor delta fields such as `delta_final_log_loss`.

### Changed

- Changed classification curve ranking so benchmark best-step selection now follows `log_loss` before ROC AUC, keeping ROC AUC as a diagnostic rather than the primary promotion metric.

- Refreshed the active `input_norm_followup` and `input_norm_none_followup` sweep state around the rerun anchor `sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2`, and recorded the unclipped `none` follow-up rerun `sd_input_norm_none_followup_01_dpnb_input_norm_none_batch64_sqrt_v2` against that log-loss-capable anchor.

- Tightened wandb-backed research execution so the new benchmarking reruns are captured and promotable under the current benchmark contract instead of relying on the earlier ROC-only sweep state.

## [0.7.0] - 2026-03-18

### Added

- Added task-aware benchmark evaluation and reporting so benchmark runs now compute classification Brier scores and regression CRPS, average pinball loss, and `PICP 90` alongside existing classification diagnostics.

- Added regression-capable benchmark bundle support through `selection.task_type`, regression task loading, and tab-foundry checkpoint evaluation, while keeping the external nanoTabPFN comparison path classification-only for now.

- Added task-aware persisted benchmark metadata fields in comparison summaries, benchmark registry entries, and control baselines, including optional `final_brier_score`, `final_crps`, `final_avg_pinball_loss`, and `final_picp_90`.

### Changed

- Changed the benchmark objective policy so classification bundles target `final_log_loss` and regression bundles target `final_crps`, with ROC AUC retained as a classification diagnostic and tie-breaker instead of the primary sweep score.

- Changed benchmark reporting surfaces to render the available task-family metric set instead of assuming ROC-only evaluation, including system-delta and instability-audit outputs.

- User-facing note: benchmark bundle selection now accepts `selection.task_type`, regression task rows omit `n_classes`, and persisted benchmark summaries, registries, and control baselines may now include the new task-aware metric fields.

## [0.6.11] - 2026-03-17

### Added

- Added two new staged/shared input-normalization modes: `train_zscore_tanh` and `train_robust_tanh`. Both use train-only statistics and then apply a bounded smooth-tail tanh transform instead of a hard clip.

- Added a new draft bridge-baseline sweep at `reference/system_delta_sweeps/input_norm_followup/`, anchored on the promoted `dpnb_row_cls_cls2_linear_warmup_decay` surface and covering unclipped z-score, winsorized z-score, smooth-tail normalization, and bounded batch-size sidecars.

- Added tracked exact-prior batch-size metadata under `training_surface_record.json`, including `prior_dump_batch_size`, `prior_dump_lr_scale_rule`, `prior_dump_batch_reference_size`, and `effective_lr_scale_factor`.

### Changed

- Exact-prior prior-dump training now resolves `training.prior_dump_batch_size`, `training.prior_dump_lr_scale_rule`, and `training.prior_dump_batch_reference_size`, and applies the resulting scale factor to both `optimizer.min_lr` and schedule `lr_max` before training.

- User-facing note: exact-prior training configs now accept the new `training.prior_dump_*` keys, and export/model validation paths now accept `train_zscore_tanh` and `train_robust_tanh` anywhere staged/shared `input_normalization` values are validated.

## [0.6.10] - 2026-03-17

### Added

- Added the staged missingness-aware tokenizer `scalar_per_feature_nan_mask`, preserving raw NaN/Inf signals through benchmark evaluation and converting each feature token into `value + missingness bit` without changing token count.

- Added exact-prior training support for `training.prior_dump_non_finite_policy=skip`, allowing local prior-dump runs to skip non-finite source batches without spending optimizer steps while recording additive skipped-batch telemetry.

- Added focused training-dynamics diagnostics to exact-prior telemetry and wandb reporting: clipped-step fractions, `feature_encoder` versus `direct_head` gradient-balance windows, and `post_feature_encoder` / `pre_transformer` activation windows.

- Added expanded research and benchmark coverage for the re-anchored `stability_followup` bridge sweep, prior-dump skip handling, plain `adamw` exact-prior training, and wandb-visible diagnostics.

- Added synthetic prior missingness injection for exact prior-dump training via `training.overrides.prior_missingness`, with additive telemetry and wandb summary reporting under `missingness.synthetic_prior`.

- Added the non-active `missingness_followup` draft sweep plus focused research tests covering the large-bundle missingness tokenizer rows and fixed-normalization sweep contract.

- Kept a repo-tracked binary-only dagzoo pilot config at `reference/system_delta_sweeps/stability_followup/dagzoo_binary_pilot.yaml` so the sibling `../dagzoo` checkout can still materialize that pilot when needed.

### Changed

- Retargeted `stability_followup` from the earlier five-row warmup/clip-plus-dagzoo draft toward a `delta_prenorm_block`-anchored 12-row bridge queue covering schedule shape, LR, warmup length, clip, weight decay, optimizer family, one bounded `row_cls` reopen, and one promotion-gate `row_cls + warm10` confirmation row.

- Promoted `dpnb_row_cls_cls2_linear_warmup_decay` as the new `cls_benchmark_staged_prior` default bridge-surface recipe: `nano_exact` plus prenorm table blocks, `row_cls` readout with `tfrow_cls_tokens=2`, linear warmup decay, and wandb-visible activation diagnostics.

- Exact-prior training now emits the new clipped-step, module-balance, activation-window, and skipped-batch diagnostics directly to wandb via per-step logs and flattened final summaries, so queue interpretation no longer depends on one-off offline scripts.

- Staged-prior wandb runs now derive sweep-specific run names from `runtime.output_dir` when the config still uses the generic default name, so uploaded runs mirror the `reference/` queue ids automatically.

- Standalone staged checkpoint evaluation now keeps benchmark normalization internal for the missingness-aware tokenizer path so raw missing values survive to tokenization instead of being erased by external preprocessing.

- Updated focused research coverage so `stability_followup` asserts the retargeted 12-row bridge queue and `missingness_followup` is registered as a separate non-active draft sweep.

## [0.6.9] - 2026-03-17

### Added

- Completed the `stability_ladder` sweep (4 rungs) isolating the marginal
  contribution of each stability mechanism on prenorm blocks trained against
  the nanoTabPFN prior dump.

- Rung A (baseline) shows an early-step loss spike to 2.23 and a max gradient
  norm of 13.16, confirming that prenorm blocks without warmup are unstable in
  the first few steps.

- Rung B (cosine warmup 5%) eliminates the spike entirely: max train loss stays
  at the initial 0.76, max gradient norm drops to 2.75 (4.8x reduction), max
  loss delta drops from 1.39 to 0.22 (6.3x), and loss variance halves from
  0.0031 to 0.0015. Warmup is the dominant stabilizer.

- Rungs C (dropout 0.1) and D (input clip 10.0) show negligible marginal
  stability effect on clean prior-dump data at 2500 steps, confirming that
  dropout and input clipping are defense-in-depth mechanisms rather than
  primary stability drivers for this training surface.

### Changed

- Promoted prenorm blocks with cosine warmup as the new default training
  surface for prior-dump experiments. `compact_binary_prior.yaml` now sets
  `training.apply_schedule=true` with a single-stage cosine schedule
  (`warmup_ratio=0.05`, `lr_schedule=cosine`) and the training surface label
  changed from `prior_constant_lr` to `prior_cosine_warmup`.

## [0.6.8] - 2026-03-17

### Added

- Three new input normalization modes: `train_rankgauss` (quantile-to-Gaussian
  via erfinv), `train_robust` (median/IQR scaling), and `train_winsorize_zscore`
  (1st/99th percentile clip then z-score). Both numpy and torch paths implemented.

- Four `delta_preproc_norm_*` catalog entries in the `input_normalization` family
  covering all four non-trivial normalization strategies.

- `binary_md_v4` system delta sweep (8 rows) testing input normalization modes
  crossed with RMSNorm and LayerNorm post-encoder norms against the v3 anchor.

- Unit and property tests for the new normalization modes, including rank-gauss
  boundedness, robust median/IQR verification, and winsorize percentile checks.

### Changed

- Extended `InputNormalizationMode` and `SUPPORTED_INPUT_NORMALIZATION_MODES` to
  include the three new modes.

## [0.6.7] - 2026-03-16

### Changed

- Removed bias from all linear layers behind pre-norm in QASS attention
  projections, QASS scaler MLPs, QASS transformer feedforward layers, and the
  shared feature encoder, reducing redundant parameters that LayerNorm/RMSNorm
  already subsumes.

- Retained explicit `bias=True` on the pre-norm cell block feedforward layers
  where the bias sits after normalization and contributes to representational
  capacity.

- Added the `delta_global_rmsnorm` row to the `binary_md_v3` system delta
  sweep queue, testing global RMSNorm combined with bias removal against the
  promoted shared+LayerNorm anchor.

## [0.6.6] - 2026-03-16

### Changed

- Standalone checkpoint evaluation now uses the shared internal Weights &
  Biases helper when `logging.use_wandb=true`, logging the existing computed
  `eval/loss` plus task-specific `eval/acc` or `eval/rmse` scalars and a
  compact summary payload without changing CLI flags or evaluation result
  schemas.

- Main trainer wandb summaries now retain the final already-computed training
  loss state, including additive `metrics/final_train_loss`,
  `metrics/final_train_loss_ema`, and task-specific
  `metrics/final_train_acc` or `metrics/final_train_rmse`, without changing
  `train_history.jsonl` or `TrainResult`.

## [0.6.5] - 2026-03-16

### Changed

- Training runs now use a shared internal Weights & Biases helper module, and
  both the main trainer and exact-prior trainer emit richer additive wandb
  scalar telemetry and end-of-run summaries without changing the logging config
  surface or uploading artifact files.

- `delta_shared_feature_norm` catalog defaults now preserve the traced training
  surface used by the canonical `binary_md_v2` queue, so curated subset sweeps
  created via `create_sweep(..., delta_refs=...)` reproduce the intended shared
  activation-baseline row instead of silently dropping `runtime.trace_activations`.

- Staged activation tracing now records detached fp32 per-element RMS values
  instead of size-dependent whole-tensor L2 norms, keeping
  `gradient_history.jsonl.activation_norms` and
  `telemetry.json.gradient_summary.activations` comparable across varying
  padded prior-dump batch shapes without changing their schema.

## [0.6.4] - 2026-03-16

### Changed

- Added optional staged `model.module_overrides.post_encoder_norm` support with
  `none`, `layernorm`, and `rmsnorm` variants, wired immediately before the
  staged transformer stack. User-facing note: staged
  `training_surface_record.json` model metadata now includes additive
  `module_selection.post_encoder_norm` and
  `module_hyperparameters.post_encoder_norm` fields.

- Prior-dump staged reruns can now opt into additive activation telemetry with
  `runtime.trace_activations`. When enabled, `gradient_history.jsonl` gains an
  optional `activation_norms` object and `telemetry.json.gradient_summary`
  gains an additive `activations` section summarizing traced forward-pass norm
  checkpoints.

- Sweep tooling now supports curated follow-up sweeps via explicit ordered
  delta subsets, and the active system-delta sweep has advanced from the
  completed `binary_md_v1` queue to the new four-row diagnostic follow-up
  sweep `binary_md_v2`.

## [0.6.3] - 2026-03-16

### Changed

- Prior-dump queue reruns now emit additive instability-debug artifacts:
  `gradient_history.jsonl` for module-level gradient traces and
  `telemetry.json` for run summaries, checkpoint snapshots, missingness
  diagnostics, and failure context. `train_history.jsonl` also gained additive
  `train_loss_delta`, `train_loss_ema`, `grad_clip_threshold`, and
  `grad_clip_triggered` fields.

- `comparison_summary.json["artifacts"]` now surfaces additive
  `gradient_history_jsonl` and `telemetry_json` pointers when those run-level
  artifacts are present, without changing the benchmark registry schema.

- Added an instability-audit entry point under
  `tab_foundry.bench.instability_audit` that scans the existing first-pass
  `outputs/staged_ladder/sd_binary_md_v1_*/train/train_history.jsonl` runs,
  joins available benchmark summaries and result cards, and writes ranked audit
  reports under `outputs/staged_ladder/reports/`.

## [0.6.2] - 2026-03-16

### Changed

- Added explicit no-missing mainline guardrails for benchmark-facing manifests,
  manifest-backed datasets, prior-dump training inputs, and benchmark bundle
  loading. `tab-foundry build-manifest` now accepts
  `--missing-value-policy {allow_any,forbid_any}`, manifest parquet files
  persist missingness metadata, and the mainline benchmark helpers reject
  bundles or inputs that allow NaN/Inf unless a future research path opts in
  explicitly.

- Restored explicit large-bundle benchmark compatibility for compare and bounce
  diagnosis entry points by auto-opting into missing-valued inputs when the
  explicitly selected bundle metadata allows them. User-facing note:
  `training_surface_record.json` now emits
  `data.manifest.characteristics.all_records_no_missing = null` for legacy
  manifests that predate missingness annotations instead of incorrectly
  reporting `true`.

- Mainline bounce diagnosis no longer defaults to the larger
  `nanotabpfn_openml_binary_large_v1.json` confirmation bundle because that
  bundle permits missing-valued inputs. The default diagnosis path now stays on
  the primary no-missing bundle unless a separate confirmation bundle is passed
  explicitly.

- Added a dedicated benchmark-bounce diagnosis runner and script that benchmark
  completed runs on both the primary and a larger confirmation bundle, writes
  task-bootstrap checkpoint traces, and classifies likely causes such as
  benchmark noise, checkpoint aliasing, optimization instability, and
  concentrated per-dataset tradeoffs.

- `comparison_summary.json` now carries additive checkpoint-level benchmark
  diagnostics for tab-foundry runs, including best-to-final drift, per-task
  deltas, task-bootstrap confidence intervals, explicit failed-checkpoint
  metadata, and the mainline no-missing benchmark-bundle policy used to produce
  the summary.

- Added a repo-tracked large binary confirmation benchmark bundle for diagnosis
  work under `src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json`.
  User-facing note: this bundle intentionally relaxes the medium bundle's
  `max_features` and `max_missing_pct` filters to `20` and `5.0` so the pinned
  TabArena source yields a meaningfully larger confirmation surface.

## [0.6.1] - 2026-03-15

### Changed

- Added row-CLS `tfrow_norm` support with a shared `RMSNorm` implementation so
  staged row-pool experiments can compare LayerNorm versus RMSNorm without
  changing the rest of the model surface.

- Added `model.norm_type` as a broader staged/global norm-family switch so
  queue rows can test RMSNorm across the active table-block, column-encoder,
  and context-encoder paths independently of the row-pool-specific
  `model.tfrow_norm` override.

- Exact-prior training now supports opt-in single-stage LR schedules for the
  system-delta queue, including linear decay and warmup-plus-linear-decay while
  preserving the prior constant-LR path by default.

- `training_surface_record.json`, benchmark registry surface labels, and
  system-delta sweep materialization/rendering now support an optional
  `training` surface dimension, and the binary medium queue now includes
  explicit RMSNorm and LR-schedule rows.

- Clarified the benchmark dependency boundary by adding optional install
  extras for benchmark helpers and Muon while keeping repo-local `uv sync`
  behavior unchanged for developers.

- Rewrote the development navigation and dependency-map docs to match the
  current three-family model surface, live `preprocessing` and `research`
  packages, and the generated active-sweep alias behavior.

- Added audit guardrails for repo-root Markdown paths, local links, and
  module-graph drift so future doc changes fail fast when the observed package
  graph diverges from `docs/development/module-dependency-map.md`.

- Removed the orphaned benchmark-run registry helper
  `load_benchmark_run_entry()` and the unused legacy benchmark bundle
  filename constant.

- Cleaned up repeated unused benchmark-test lambda parameters so the repo now
  passes the current `vulture` threshold cleanly.

- `resolve_data_surface()` and `resolve_preprocessing_surface()` now treat
  explicit `null`/`None` config values as unset and fall back to their intended
  defaults instead of coercing them into `"none"`, `False`, or a `TypeError`.

- Added Hypothesis-based property tests for benchmark bundle normalization,
  data/preprocessing surface resolution, model-spec resolution, manifest helper
  determinism, and training/runtime resolution invariants under
  `tests/property/`.

- `build_stage_configs()` now rejects non-positive stage step counts instead of
  silently accepting invalid stage payloads.

## [0.6.0] - 2026-03-14

### Changed

- Refactored the system-delta workflow into a multi-sweep campaign model with a
  reusable delta catalog in `reference/system_delta_catalog.yaml`, per-sweep
  canonical sources under `reference/system_delta_sweeps/<sweep_id>/`, and an
  active-sweep index in `reference/system_delta_sweeps/index.yaml`.

- `scripts/system_delta_queue.py` and `src/tab_foundry/research/system_delta.py`
  are now sweep-aware. The tool gained `list-sweeps`, `show-active`,
  `set-active`, and `create-sweep`, and existing queue commands now accept an
  optional `--sweep-id`.

- New sweep creation now materializes queue instances from the delta catalog,
  applies machine-checked applicability guards against the anchor context, and
  renders a sweep-specific matrix plus active-sweep aliases.

- Benchmark registry records now support additive sweep metadata:
  `sweep_id`, `delta_id`, `parent_sweep_id`, `queue_order`, and `run_kind`.

- New system-delta research artifacts are sweep-namespaced under
  `outputs/staged_ladder/research/<sweep_id>/<delta_id>/...`, and rendered
  matrix references now point at those namespaced result-card paths.

- `program.md` now defines the active-sweep source-of-truth hierarchy and
  requires new complexity passes to create a new sweep instead of mutating an
  existing completed sweep.

- User-facing workflow break: `reference/system_delta_queue.yaml` and
  `reference/system_delta_matrix.md` are no longer the canonical editable
  sources. They are generated active-sweep aliases, so their default contents
  now depend on `reference/system_delta_sweeps/index.yaml`.

## [0.5.1] - 2026-03-14

### Changed

- Staged surface resolution now derives normalization mode from the effective
  feature encoder instead of the base ladder stage, so anchor-only shared
  feature-encoder deltas use the intended external train/test normalization
  path.

- `resolve_staged_surface()` now rejects tokenizer overrides that cannot affect
  execution while the effective feature encoder remains `nano`, preventing
  no-op benchmark rows from being recorded as real ablations.

- Checkpoint evaluation now reuses persisted `preprocessing` config when it is
  present in the saved checkpoint payload, keeping non-default runtime
  preprocessing surfaces reproducible across train and eval.

- `comparison_summary.json` now records the actual derived
  `training_surface_record.json` artifact path instead of a speculative
  benchmark-directory placeholder.

## [0.5.0] - 2026-03-14

### Changed

- Refactored `tabfoundry_staged` so staged runs resolve through an explicit
  atomic surface instead of hidden `_build_*` branch constants. Row-pool,
  column-encoder, and context-encoder capacity knobs now come from surfaced
  config (`tfrow_*`, `tfcol_*`, `tficl_*`) rather than hard-coded values.

- Added additive staged model config fields `model.stage_label` and
  `model.module_overrides`. These are user-facing config-surface changes under
  `src/tab_foundry` and are intended for anchor-only system-delta experiments.

- Added additive data and preprocessing surface settings:
  `data.surface_label`, `data.surface_overrides`, `data.filter_policy`,
  `data.dagzoo_provenance`, and the new root `preprocessing` config group with
  explicit runtime preprocessing overrides.

- Training and benchmark-facing runs now emit or derive a new
  `training_surface_record.json` artifact that records the effective model
  surface, data surface, manifest fingerprint and characteristics, dagzoo
  provenance references when present, and preprocessing surface.

- Benchmark comparison summaries and benchmark-run registry entries now carry
  additive training-surface metadata, including
  `artifacts.training_surface_record_path` and surfaced model/data/preprocessing
  labels.

- Added the anchor-only system-delta workflow contract:
  `reference/system_delta_queue.yaml`,
  `reference/system_delta_matrix.md`,
  `reference/system_delta_campaign_template.md`, and
  `scripts/system_delta_queue.py`. `program.md` now references this queue/matrix
  workflow instead of the old stage-promotion loop.

- User-facing workflow break: the active research contract is now an
  interpretation-first anchor-only sweep across model, data, and preprocessing
  dimensions. `result_card.md` and `training_surface_record.json` are required
  evidence artifacts for completed rows, and underperformance alone is not
  sufficient for `reject`.

## [0.4.0] - 2026-03-14

### Changed

- Added a named OpenML task-source registry under `src/tab_foundry/bench/`
  plus an additive `--task-source` flag to
  `scripts/build_openml_benchmark_bundle.py`, preserving the legacy
  `tabarena_v0_1` source while adding the broader pinned
  `binary_expanded_v1` source for the canonical medium binary surface.

- Added the repo-tracked
  `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json` bundle and
  changed `default_benchmark_bundle_path()` to return that 10-task binary
  surface by default.

- Rolled the canonical frozen control baseline id from
  `cls_benchmark_linear_v1` to `cls_benchmark_linear_v2` without rewriting the
  historical `v1` registry entry.

- User-facing benchmark surface break: the canonical benchmark bundle path now
  defaults to `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
  instead of `src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json`; the
  canonical comparison surface therefore changes, and new benchmark results are
  not directly comparable to historical `v1` entries unless the bundle path and
  control-baseline id are matched explicitly.

- Redefined `tab-foundry-export-v3` as a single-manifest bundle containing
  `manifest.json` plus `weights.safetensors`. The prior v3 sidecar-based bundle
  layout is obsolete, and both those older sidecar bundles and earlier
  single-manifest v3 bundles without `manifest_sha256` must be regenerated.

- Embedded inference and preprocessing policy metadata directly into
  `manifest.json`; `inference_config.json` and `preprocessor_state.json` are no
  longer emitted for v3 bundles.

- Changed the v3 preprocessing contract from dataset-specific fitted state to
  policy-only metadata. Bundled v3 exports no longer persist `feature_ids`,
  per-feature `fill_values`, or classification `label_values`.

- Removed `tab-foundry build-preprocessor-state` and dropped
  `tab-foundry export --preprocessor-state` from the supported CLI surface.

- Reference execution now derives preprocessing from the incoming runtime
  support set, keeping export consumption aligned with dataset loading instead
  of reusing export-time fitted values.

- Export validation now rejects unsupported `manifest.model.input_normalization`
  values before model construction.

- Benchmark checkpoint evaluation now honors the saved
  `model.input_normalization` exactly. Checkpoints saved with
  `input_normalization="none"` no longer get an implicit train-zscore pass in
  the external benchmark wrapper, which changes the reproduced ROC AUC curve
  for the canonical `cls_benchmark_linear_v1` control baseline.

- Refactored tabfoundry classification conditioning so the column/row backbone
  is feature-only and label-conditioned reasoning happens in the ICL stage.
  The many-class path now reruns ICL conditioning per mixed-radix digit view
  instead of averaging feature-identical column outputs.

- `tficl` attention masking now preserves each test token's own state by
  allowing test-token self-attention while still blocking test-to-test
  cross-attention.

- Added an additive `tabfoundry_simple` classification debug architecture plus
  the `experiment=cls_benchmark_linear_simple` benchmark profile for
  nanoTabPFN-style ablation work on the current 3-task control benchmark.

- Fixed many-class hierarchical path target clamping to respect the configured
  `many_class_base` instead of hardcoding base-10 behavior.

- `TabFoundryStagedClassifier` now rejects unsupported
  `many_class_train_mode` and `input_normalization` values during
  construction, matching the eager validation behavior already enforced by the
  canonical model spec and the existing tabfoundry model families.

## [0.3.2] - 2026-03-13

### Changed

- `scripts/benchmark_nanotabpfn.py` now accepts an additive
  `--benchmark-bundle-path` override so staged benchmark runs can target a
  repo-tracked non-default OpenML bundle while keeping the binary 3-task bundle
  as the default compare path.
- Added `scripts/build_openml_benchmark_bundle.py` plus the repo-tracked
  `src/tab_foundry/bench/nanotabpfn_openml_classification_small_v1.json`
  companion bundle, which widens the nanoTabPFN notebook's TabArena v0.1 task
  set from binary-only to small multiclass without changing the other bundle
  selection thresholds or runtime drift-check contract.
- Prior-dump training now rejects staged `model.stage="many_class"` configs up
  front instead of failing on the first batch when the staged model does not
  expose direct `forward_batched()` tensor logits.
- `tab-foundry-export-v2` validation now rejects bundles whose
  `manifest.json` and `inference_config.json` disagree on `model_arch` or
  staged `model_stage`, preventing contradictory model identity metadata from
  loading successfully.

## [0.3.1] - 2026-03-13

### Changed

- `experiment=cls_benchmark_linear` now writes `train_history.jsonl` under the
  run output directory by default, so the benchmark comparison flow can consume
  plain training outputs rather than only smoke-harness layouts.
- Added a repo-tracked canonical control baseline registry at
  `src/tab_foundry/bench/control_baselines_v1.json` plus a promotion helper in
  `scripts/freeze_control_baseline.py` that validates a completed run and
  `comparison_summary.json` before freezing the baseline metadata.
- `comparison_summary.json` now supports an additive `control_baseline` object
  copied from the frozen registry when benchmark comparisons are run with
  `--control-baseline-id`.
- `tab-foundry train` now fails fast when `runtime.output_dir` already contains
  a non-empty history JSONL or checkpoint `.pt` artifacts, preventing benchmark
  reruns from mixing stale training outputs into later comparisons.

## [0.3.0] - 2026-03-13

### Added

- Added a shared preprocessing package under `src/tab_foundry/preprocessing/`
  for fitted train-mean imputation and classification-label remapping.
- Added a reference-only executable consumer in
  `tab_foundry.export.loader_ref` plus pinned conformance fixtures for one
  classification bundle and one regression bundle.
- Export bundles now default to the new `tab-foundry-export-v3` schema. This is
  a user-facing artifact-contract change, and v3 manifests now require
  `manifest_sha256` to protect embedded metadata against post-export edits.
- `manifest.model` now persists `input_normalization`, and the model
  reconstruction path round-trips that field across export and load.
- `manifest.json` now embeds the v3 inference and preprocessing policy sections
  directly instead of emitting `inference_config.json` and
  `preprocessor_state.json` sidecars.
- `tab-foundry-export-v2` remains validator-readable during migration, but the
  executable reference consumer in this repo is intentionally v3-only.
- Training checkpoints no longer persist fitted preprocessing state. Checkpoints
  remain resume/eval artifacts, while v3 export consumes an explicit external
  runtime-derived preprocessing contract encoded in `manifest.json`.
- Classification label remapping and unseen-test-label filtering now still run
  when `PackedParquetTaskDataset(..., impute_missing=False)` leaves feature
  values unimputed.

## [0.2.0] - 2026-03-13

### Added

- Added an internal architecture reference at
  `docs/development/model-architecture.md` that documents the
  current tabfoundry transformer stack and its task heads.
- Added `docs/development/model-config.md` as the dedicated reference for model
  settings, defaults, and resolution precedence across train/eval/export/load.

### Changed

- Renamed the primary model family from `tabiclv2` to `tabfoundry`. Python
  imports now use `tab_foundry.model.architectures.tabfoundry` with
  `TabFoundryClassifier` and `TabFoundryRegressor`; the old module path and
  class names were removed.
- Export bundles now use `manifest.model.arch="tabfoundry"` and
  `inference_config.model_arch="tabfoundry"` under the new
  `tab-foundry-export-v2` schema. Existing `tab-foundry-export-v1` bundles are
  intentionally unsupported and must be regenerated.
- Reconciled `feature_group_size` defaults so omitted model settings now resolve
  to the canonical per-feature default of `1` across Hydra config composition,
  model spec fallback resolution, model construction, and repo-owned
  benchmark/smoke workflows.
- Checkpoint-backed eval/export/load no longer infer omitted
  `feature_group_size` values from legacy grouped-token weights. Legacy
  checkpoints without an explicit `feature_group_size` must now be regenerated
  or loaded with an explicit override that matches the saved weights.
- Updated internal planning, navigation, and contract docs to describe the
  current family as `tabfoundry` and treat `TabICLv2` as an external reference
  rather than the repo's persistent model identity.

## [0.1.3] - 2026-03-13

### Changed

- Pinned the canonical nanoTabPFN/OpenML benchmark bundle in
  `src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json` and made the
  benchmark load path fail fast on selection-threshold, task-list, and
  task-metadata drift.
- Updated benchmark comparison runs so `benchmark_tasks.json` now persists the
  exact pinned benchmark bundle used for the run, and
  `comparison_summary.json` now includes additive `benchmark_bundle` metadata
  with name, version, source path, task count, and task IDs.
- Tightened the benchmark bundle schema so `selection` now requires the raw
  OpenML thresholds for row count, task type, feature count, class count,
  missingness, and minority-class percentage; older ad hoc bundle files
  without those fields now fail to load.
- Normalized `TF-RD-LEGACY-002` to implemented in the canonical roadmap and marked the
  TF-RD-LEGACY-001 kickoff on the tracker around pinned benchmark-input discipline.

## [0.1.2] - 2026-03-13

### Changed

- Centralized canonical model-build spec resolution in
  `tab_foundry.model.spec` so training, evaluation, checkpoint loading, and
  export manifest validation share one defaulting and coercion path.
- Consolidated duplicated smoke-harness manifest/config helpers into
  `tab_foundry.bench.smoke_common` while preserving the existing dagzoo and
  Iris workflow behavior.
- Removed package-root convenience re-exports from `tab_foundry.model`,
  `tab_foundry.data`, and `tab_foundry.export`. Direct Python imports should
  now target concrete submodules. This is a Python API cleanup only; CLI flags
  and export bundle or dataset artifact contracts did not change.
- Updated contributor and development docs to reflect the current role-based
  package layout and the repo's actual local verification commands.

## [0.1.1] - 2026-03-12

### Added

- Added repo-managed `pre-commit` hooks for `ruff`, `mypy`, and `mdformat`,
  plus local and CI documentation for the shared quality workflow.
- Added an Iris-backed smoke harness that generates manifest-compatible packed
  tasks, trains a real checkpoint through the existing pipeline, evaluates that
  checkpoint, and writes `telemetry.json` plus `summary.md` artifacts.
- Added a GitHub Actions `test` workflow with `quality-and-unit` and
  `iris-smoke` jobs, plus a helper script to configure `main` branch protection
  around those checks.

### Changed

- Refactored shared benchmark/smoke artifact helpers so history loading,
  checkpoint-snapshot timing, JSON persistence, and loss-curve rendering live in
  one bench module instead of duplicated harness code.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Normalized `TF-RD-002` to implemented in the canonical roadmap and marked the
  TF-RD-001 kickoff on the tracker around pinned benchmark-input discipline.

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

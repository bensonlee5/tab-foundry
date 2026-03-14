# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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

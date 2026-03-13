# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

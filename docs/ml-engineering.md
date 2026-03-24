# ML Engineering And Infra

Use this guide when you need the repo's operational view: artifacts, ownership
boundaries, and verification paths.

## Overview

`tab-foundry` is where the training-side work happens. It turns data
descriptions and configs into training [runs](glossary.md#run-directory),
[checkpoints](glossary.md#checkpoint), benchmark results, and
[export bundles](glossary.md#export-bundle).

If you care most about files, artifacts, contracts, and validation paths, start
here instead of the research-only docs.

Use these alongside this guide:

- general start page: [docs/getting-started.md](getting-started.md)
- repo overview: [docs/what-is-tab-foundry.md](what-is-tab-foundry.md)
- workflows: [docs/workflows.md](workflows.md)
- inference/export contract: [docs/inference.md](inference.md)
- contribution workflow: [CONTRIBUTING.md](../CONTRIBUTING.md)

## What This Repo Owns

This repo owns:

- corpus-backed data selection, manifest inspection, and dataset provenance
- model training and evaluation
- benchmark comparison and sweep evidence
- export-bundle production and validation

This repo does not own:

- long-lived production serving
- downstream runtime API ownership
- generic production inference policy beyond the reference consumer contract

## Artifacts That Matter

- [manifest](glossary.md#manifest):
  - describes the concrete data/tasks used by training and evaluation
- [run directory](glossary.md#run-directory):
  - contains histories, checkpoints, summaries, and telemetry for one training
    run
- [checkpoint](glossary.md#checkpoint):
  - saved model state used for evaluation, comparison, and export
- [benchmark bundle](glossary.md#benchmark-bundle):
  - pinned task set used to compare runs consistently
- [export bundle](glossary.md#export-bundle):
  - packaged inference artifact handed off to downstream runtime ownership

## Files And Paths To Know

- [docs/workflows.md](workflows.md):
  canonical commands and artifact expectations
- [docs/inference.md](inference.md):
  export schema and runtime handoff boundary
- [src/tab_foundry/bench/benchmark_run_registry_v1.json](../src/tab_foundry/bench/benchmark_run_registry_v1.json):
  historical benchmark-facing system of record
- `outputs/...`:
  local run, checkpoint, benchmark, and sweep artifacts

## Common Operational Flows

### Materialize And Train A Corpus-Backed Surface

```bash
.venv/bin/tab-foundry data corpus materialize \
  --recipe tf_rd_013_current_corpus_default_v1 \
  --dagzoo-root ../dagzoo \
  --force
.venv/bin/tab-foundry train run \
  experiment=cls_benchmark_staged_corpus \
  data.corpus_ref=tf_rd_013_current_corpus_default_v1
```

### Inspect One Corpus, Manifest, Or Run

```bash
.venv/bin/tab-foundry data corpus inspect \
  --corpus-ref tf_rd_013_current_corpus_default_v1
.venv/bin/tab-foundry data manifest-inspect \
  --manifest data/manifests/default.parquet \
  --experiment cls_smoke \
  --override data.manifest_path=data/manifests/default.parquet
.venv/bin/tab-foundry dev run-inspect --run-dir outputs/cls_smoke
```

### Evaluate Or Export One Checkpoint

```bash
.venv/bin/tab-foundry eval checkpoint \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  experiment=cls_smoke

.venv/bin/tab-foundry export bundle \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  --out-dir outputs/exports/cls_smoke_v3

.venv/bin/tab-foundry export validate \
  --bundle-dir outputs/exports/cls_smoke_v3
```

### Verify A Change Safely

```bash
./scripts/dev review-base
./scripts/dev verify affected
./scripts/dev verify paths src/tab_foundry/export/contracts.py
```

## How To Read The Repo As An Engineer

If you need the minimal mental model:

- data enters through corpus refs or explicit manifest overrides
- training produces run directories and checkpoints
- benchmarking compares selected runs against pinned bundles
- export turns checkpoints into handoff artifacts
- sweeps explain why one change was tested and how it performed

## Where To Go Next

- [docs/workflows.md](workflows.md): operational commands and artifact flow
- [docs/inference.md](inference.md): exact export-bundle contract
- [docs/research-contributors.md](research-contributors.md): if you also need
  the research-side framing

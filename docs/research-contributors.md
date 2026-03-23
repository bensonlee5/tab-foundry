# Research Contributors

Use this guide when you are working on architecture, sweeps, synthetic data, or
broader model-capability questions in `tab-foundry`.

## Overview

The research workflow in this repo is designed to answer one question at a
time. It uses [sweeps](glossary.md#sweep) to compare isolated changes against a
locked [anchor](glossary.md#anchor), so the team can tell whether a model,
data, or preprocessing change actually helped.

Most research contributors come here to answer four common questions:

1. what architecture is active here?
1. how do sweeps work without breaking attribution?
1. where does synthetic data fit relative to real-data ladders?
1. how should broader model capability work be framed?

Use these alongside this guide:

- shared vocabulary: [docs/glossary.md](glossary.md)
- general orientation: [docs/getting-started.md](getting-started.md)
- contribution workflow: [CONTRIBUTING.md](../CONTRIBUTING.md)
- operational commands: [docs/workflows.md](workflows.md)
- sweep contract: [program.md](../program.md)
- architecture reference: [docs/development/model-architecture.md](development/model-architecture.md)
- roadmap: [docs/development/roadmap.md](development/roadmap.md)

## Repo Mental Model

`tab-foundry` has one frozen control lane and one active development lane.

- Frozen control lane:
  - `tabfoundry_simple`
  - `tabfoundry_staged` with `stage=nano_exact`
  - used for benchmark trust and PFN-style comparison
- Active development lane:
  - `tabfoundry_staged`
  - row-first classification target inspired by TabICLv2
  - the only live surface for new architecture work

Use the canon this way:

- [docs/development/model-architecture.md](development/model-architecture.md):
  what the current model surface is
- [docs/workflows.md](workflows.md): how to run commands and what artifacts to
  expect
- [program.md](../program.md): how the active sweep is allowed to change
- [docs/development/roadmap.md](development/roadmap.md): which architectural
  or capability questions are actually next
- [reference/README.md](../reference/README.md): where literature and evidence
  live

## Task 1: Understand The Architecture

Use this path when the question is:

- what is frozen vs active?
- where does a subsystem live?
- what is the current forward path?

Read in this order:

1. [docs/development/design-decisions.md](development/design-decisions.md)
1. [docs/development/model-architecture.md](development/model-architecture.md)
1. [docs/development/codebase-navigation.md](development/codebase-navigation.md)
1. [docs/development/module-dependency-map.md](development/module-dependency-map.md)

Inspect-first commands:

```bash
.venv/bin/tab-foundry dev resolve-config experiment=cls_smoke
.venv/bin/tab-foundry dev forward-check experiment=cls_smoke
.venv/bin/tab-foundry dev diff-config \
  --left experiment=cls_smoke \
  --right experiment=cls_smoke \
  --right model.stage=qass_context
./scripts/dev verify paths src/tab_foundry/model/factory.py
```

Artifacts and files to read:

- resolved configs
- forward-check summaries
- [src/tab_foundry/model/architectures/tabfoundry_staged/model.py](../src/tab_foundry/model/architectures/tabfoundry_staged/model.py)
- [src/tab_foundry/model/architectures/tabfoundry_staged/resolved.py](../src/tab_foundry/model/architectures/tabfoundry_staged/resolved.py)

Common mistakes:

- treating `nano_exact` bridge or diagnostic surfaces as the long-term target
- inferring architecture policy from a historical sweep row instead of from the
  roadmap and design docs
- changing `bench/` or `research/` code when the real change belongs in the
  model package

## Task 2: Inspect Or Run Sweeps

Use this path when the question is:

- what is the active sweep?
- what row should run next?
- what artifacts make a row interpretable?

Read in this order:

1. [program.md](../program.md)
1. [docs/workflows.md](workflows.md)
1. [reference/system_delta_campaign_template.md](../reference/system_delta_campaign_template.md)
1. [docs/development/roadmap.md](development/roadmap.md)

Inspect-first commands:

```bash
.venv/bin/tab-foundry research sweep list
.venv/bin/tab-foundry research sweep next
.venv/bin/tab-foundry research sweep summarize --include-screened
.venv/bin/tab-foundry research sweep inspect --order <order> --sweep-id <sweep_id>
.venv/bin/tab-foundry research sweep diff \
  --order <order> \
  --against-order <anchor_order> \
  --sweep-id <sweep_id>
.venv/bin/tab-foundry research sweep graph --anchor
```

Expected research-package artifacts:

- `research_card.md`
- `campaign.yaml`
- `result_card.md` for `benchmark_full` rows
- `training_surface_record.json`
- `train_history.jsonl`
- `gradient_history.jsonl`
- `telemetry.json`

Common mistakes:

- treating [docs/development/roadmap.md](development/roadmap.md) as the sweep
  contract
- mutating a completed sweep instead of starting a new one
- benchmarking directly when the row is only `screen_only`
- changing more than one dimension family in a single row

## Task 3: Work On Synthetic Data

Use this path when the question is:

- how does `dagzoo` relate to the repo?
- when should synthetic data change the training surface?
- how is synthetic work separated from curated real-data ladders?

Read in this order:

1. [docs/development/dataset-curation.md](development/dataset-curation.md)
1. [docs/development/roadmap.md](development/roadmap.md)
1. [reference/system_delta_sweeps/tf_rd_013_shape_aware_dagzoo_v1/support/README.md](../reference/system_delta_sweeps/tf_rd_013_shape_aware_dagzoo_v1/support/README.md)
1. [docs/workflows.md](workflows.md)

Key rule:

- `dagzoo` is the synthetic-data lane
- OpenML and vetted external datasets are the curated real-data comparator
  lanes
- synthetic corpora do not remove the license-review requirement for real-data
  ladders

Inspect-first commands:

```bash
.venv/bin/tab-foundry data manifest-inspect \
  --manifest data/manifests/default.parquet \
  --experiment cls_smoke \
  --override data.manifest_path=data/manifests/default.parquet
.venv/bin/tab-foundry dev resolve-config experiment=cls_benchmark_staged
.venv/bin/tab-foundry research sweep inspect \
  --sweep-id tf_rd_013_shape_aware_dagzoo_v1 \
  --order 1
```

Common mistakes:

- treating `dagzoo` as interchangeable with curated real-data benchmark ladders
- adding a new data loader path when the manifest-backed surface already solves
  the workflow
- discussing TF-RD-013 without naming which corpus provenance or manifest
  surface is under review

## Task 4: Propose Broader Model Capability

Use this path when the question is:

- how should many-class, regression, inference handoff, or scaling work start?
- which roadmap item owns a breadth proposal?

Relevant roadmap lanes:

- many-class promotion: `TF-RD-010`
- regression rebuild: `TF-RD-015`
- inference handoff and later modalities: `TF-RD-012`
- scaling-law measurement: `TF-RD-009`

Read in this order:

1. [docs/development/roadmap.md](development/roadmap.md)
1. [docs/development/model-architecture.md](development/model-architecture.md)
1. [reference/evidence.md](../reference/evidence.md)
1. [reference/papers.md](../reference/papers.md)

Default framing:

- update the roadmap if the work changes repo priorities or gates
- use a sweep row if the question is an attributable change against the current
  anchor
- use a bounded prototype only when the existing surface cannot express the
  question cleanly

Common mistakes:

- treating breadth work as independent of the current anchor and adequacy gates
- introducing a second live model family instead of extending
  `tabfoundry_staged`
- skipping the roadmap and opening an implementation path that is not yet in
  scope

## First Week Checklist

If you are brand new, this sequence gives you the highest context return with
the least churn:

1. Read [docs/glossary.md](glossary.md).
1. Read [docs/development/design-decisions.md](development/design-decisions.md).
1. Read [docs/development/model-architecture.md](development/model-architecture.md).
1. Run one config resolve and one forward-check command.
1. Read [program.md](../program.md) and inspect the active sweep.
1. Read [CONTRIBUTING.md](../CONTRIBUTING.md) before editing code or sweep
   state.

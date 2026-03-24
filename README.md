# tab-foundry

Training, comparing, and exporting tabular ML models, with direct routes into
research, operations, and architecture docs.

## Start Here

Use these entry points when you want the shortest path to the current
architecture, workflows, and research surfaces.

| Goal | Start here | Then go deeper |
| ---- | ---------- | -------------- |
| Understand what this repo is | [docs/what-is-tab-foundry.md](docs/what-is-tab-foundry.md) | [docs/getting-started.md](docs/getting-started.md) |
| Get oriented quickly | [docs/getting-started.md](docs/getting-started.md) | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Contribute to research | [docs/research-contributors.md](docs/research-contributors.md) | [program.md](program.md) and [docs/workflows.md](docs/workflows.md) |
| Work as ML engineering / infra | [docs/ml-engineering.md](docs/ml-engineering.md) | [docs/inference.md](docs/inference.md) and [docs/workflows.md](docs/workflows.md) |
| Understand the active architecture | [docs/getting-started.md](docs/getting-started.md) | [docs/development/model-architecture.md](docs/development/model-architecture.md) |
| Think through synthetic data and dagzoo | [docs/getting-started.md](docs/getting-started.md) | [docs/development/dataset-curation.md](docs/development/dataset-curation.md) |
| Plan broader model capability work | [docs/research-contributors.md](docs/research-contributors.md) | [docs/development/roadmap.md](docs/development/roadmap.md) |
| Learn repo vocabulary | [docs/glossary.md](docs/glossary.md) | [reference/README.md](reference/README.md) |

Check out the docs site for more guides, references, and development policies:
[bensonlee5.github.io/tab-foundry](https://bensonlee5.github.io/tab-foundry/).

## Environment

- Python `3.14` (pinned in `.python-version`)
- `uv` workflow for sync, tooling, and commands

## Setup

```bash
./scripts/dev bootstrap
```

`./scripts/dev bootstrap` wraps the canonical repo-local setup:

```bash
uv sync
pre-commit install
```

After setup, activate the virtual environment so you can run commands directly
without the `uv run` prefix:

```bash
source .venv/bin/activate
```

Repo-local `uv sync` includes the benchmark helper dependencies plus Muon
through the dev environment. For a minimal non-dev install, opt into the extra
surfaces explicitly:

```bash
uv sync --no-dev --extra benchmark --extra muon
```

## CLI Navigation

`tab-foundry` is the canonical packaged CLI for data, dev, train, eval,
export, bench, and research workflows. Treat `./scripts/dev` as a repo-local
convenience wrapper only for bootstrap, doctor, ready, verification, and Iris
smoke.

The only remaining non-packaged Python entrypoints are the standalone internal
benchmark helpers under `scripts/bench/`; use them only when a runbook such as
`program.md` calls for them explicitly.

For command discovery and execution, prefer `.venv/bin/tab-foundry ...` or an
activated `.venv`. Use help in this order:

```bash
.venv/bin/tab-foundry --help
.venv/bin/tab-foundry <group> --help
.venv/bin/tab-foundry <group> <command> --help
```

Use [codebase navigation](docs/development/codebase-navigation.md) for the full
namespace inventory and [workflows](docs/workflows.md) for operational
runbooks.

Sanity-check the repo-local environment:

```bash
./scripts/dev doctor
```

Summarize the current diff against `origin/main` and run the smallest safe
verification slice:

```bash
./scripts/dev ready --base-ref origin/main
./scripts/dev review-base
./scripts/dev verify affected
./scripts/dev verify paths src/tab_foundry/model/architectures/tabfoundry_staged/subsystems.py
```

Run the full local quality gate:

```bash
./scripts/dev verify full
```

Inspect one resolved config or run a forward-only construction smoke check:

```bash
tab-foundry dev resolve-config experiment=cls_smoke
tab-foundry dev forward-check experiment=cls_smoke
tab-foundry dev diff-config --left experiment=cls_smoke --right experiment=cls_smoke --right model.stage=many_class
```

Summarize one run's instability telemetry or one sweep's local results:

```bash
tab-foundry dev health-check --run-dir outputs/cls_smoke
tab-foundry dev run-inspect --run-dir outputs/cls_smoke
tab-foundry dev export-check --checkpoint outputs/cls_smoke/checkpoints/best.pt
tab-foundry data manifest-inspect --manifest data/manifests/default.parquet --experiment cls_smoke --override data.manifest_path=data/manifests/default.parquet
tab-foundry research sweep summarize --include-screened
tab-foundry research sweep list --sweep-id binary_md_v1
tab-foundry research sweep inspect --order 6 --sweep-id binary_md_v1
tab-foundry research sweep diff --order 7 --against-order 6 --sweep-id binary_md_v1
```

## Quickstart

Build a manifest:

```bash
export DAGZOO_DATA_ROOT="$HOME/dev/dagzoo/data"
tab-foundry data build-manifest \
  --data-root "${DAGZOO_DATA_ROOT:-$HOME/dev/dagzoo/data}" \
  --out-manifest data/manifests/default.parquet
```

Train a smoke profile:

```bash
tab-foundry train run experiment=cls_smoke
```

Evaluate a checkpoint:

```bash
tab-foundry eval checkpoint \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  experiment=cls_smoke
```

Export and validate an inference bundle:

```bash
tab-foundry export bundle \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  --out-dir outputs/exports/cls_smoke_v3

tab-foundry export validate \
  --bundle-dir outputs/exports/cls_smoke_v3
```

Run the Iris smoke harness:

```bash
./scripts/dev smoke iris
```

Repo-local `uv sync` includes Muon. If you are using a minimal install without
the `muon` extra, override the optimizer explicitly:

```bash
tab-foundry train run experiment=cls_smoke optimizer=adamw
```

## Docs

- Published docs site: <https://bensonlee5.github.io/tab-foundry/>
- `docs/getting-started.md`: researcher onboarding path for architecture, sweeps, synthetic data, and model breadth
- `docs/what-is-tab-foundry.md`: repo overview and entry-point guide
- `docs/research-contributors.md`: research workflow onboarding path
- `docs/ml-engineering.md`: artifact and workflow path for ML engineering / infra readers
- `docs/glossary.md`: shared vocabulary for sweep and architecture work
- `CONTRIBUTING.md`: contribution workflow for research contributors
- `docs/workflows.md`: setup, manifest build, train/eval/export, smoke flows, tuning, benchmarking, and CI
- `docs/inference.md`: export bundle schema and validation contract
- `docs/development/roadmap.md`: canonical planning state and ranked roadmap
- `docs/development/design-decisions.md`: architecture direction, repo-structure policy, and compatibility guidance
- `docs/development/model-architecture.md`: detailed architecture reference for the current staged/simple model surfaces
- `docs/development/architecture-deltas.md`: diagram-first comparison of the active row-first target direction against TabPFN and TabICLv2 reference lineages
- `docs/development/model-config.md`: model configuration reference, defaults, and resolution rules
- `docs/development/codebase-navigation.md`: current package layout and workflow entry surfaces
- `docs/development/module-dependency-map.md`: maintained baseline dependency view for repo evolution
- `reference/README.md`: index for literature notes, evidence maps, and future adjacent-repo summaries
- `reference/papers.md`: curated papers, typed-column-encoder references, and external baseline borrowing rules
- `reference/evidence.md`: roadmap-to-reference mapping and evidence notes
- `site/README.md`: local Hugo build, sync, and Pages publishing workflow

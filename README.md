# tab-foundry

Training, benchmarking, and exporting tabular ML models, with research and
workflow guidance documented in the repo docs and published docs site.

The primary reader path is the docs site:
[bensonlee5.github.io/tab-foundry](https://bensonlee5.github.io/tab-foundry/).

## Start Here

Use the docs site first, then fall back to the local Markdown docs when you
are working in the repo:

- [docs/what-is-tab-foundry.md](docs/what-is-tab-foundry.md): short repo
  overview
- [docs/getting-started.md](docs/getting-started.md): orientation and reader
  paths
- [docs/research-contributors.md](docs/research-contributors.md): research and
  sweep workflow
- [docs/ml-engineering.md](docs/ml-engineering.md): artifact and infra-facing
  workflow
- [docs/workflows.md](docs/workflows.md): canonical commands and operational
  runbooks

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

## Finding Commands

`tab-foundry` is the canonical packaged CLI for data, dev, train, eval,
export, bench, and research workflows. Treat `./scripts/dev` as a repo-local
convenience wrapper only for bootstrap, doctor, ready, verification, and Iris
smoke.

The remaining repo-local entrypoints outside the packaged CLI are
`./scripts/dev`, a small set of shell helpers under `scripts/`, the standalone
internal benchmark helpers under `scripts/bench/`, and the TF-RD-013 support
materializer at `scripts/materialize_tf_rd_013_support.py`. Use those only
when a runbook such as `program.md` or a checked-in support README calls for
them explicitly.

After installation, use `tab-foundry ...` for command discovery and execution.
Use `--help` in this order:

```bash
tab-foundry --help
tab-foundry <group> --help
tab-foundry <group> <command> --help
```

CLI tree:

- `tab-foundry`
  - `data`: manifests, corpora, and dataset inspection
    - `build-manifest`: build a manifest from one or more data roots
    - `manifest-inspect`: inspect one manifest against an experiment surface
    - `dagzoo generate-manifest`: generate dagzoo data and emit a manifest
    - `corpus list-recipes`: list available corpus recipes
    - `corpus materialize`: materialize one named corpus recipe
    - `corpus inspect`: inspect one materialized corpus
    - `corpus compare`: compare two corpus records
    - `corpus results`: summarize benchmark results tied to one corpus
  - `dev`: developer inspection and verification helpers
    - `resolve-config`: render the resolved Hydra config
    - `forward-check`: build the model and run a forward-only smoke check
    - `diff-config`: diff two resolved config surfaces
    - `export-check`: validate exportability from one checkpoint
    - `health-check`: summarize training telemetry health for one run
    - `run-inspect`: inspect one run directory and its artifacts
  - `train`: training entrypoints
    - `run`: train from Hydra config overrides
    - `prior simple`: train the exact-prior simple benchmark family
    - `prior staged`: train the exact-prior staged benchmark family
  - `eval`: checkpoint evaluation
    - `checkpoint`: evaluate one checkpoint on a selected split
  - `export`: inference bundle workflows
    - `bundle`: export one checkpoint as an inference bundle
    - `validate`: validate an exported inference bundle
  - `bench`: smoke, benchmark, tuning, and registry workflows
    - `smoke iris`: run the Iris smoke harness
    - `smoke dagzoo`: run the dagzoo smoke harness
    - `tune`: run the internal benchmark tuning sweep
    - `compare`: compare one run against external baselines
    - `env bootstrap`: bootstrap sibling benchmark environments
    - `bundle build-openml`: build an OpenML benchmark bundle
    - `registry register-run`: register a benchmark-facing run
    - `registry freeze-baseline`: freeze a control baseline
    - `diagnose bounce`: run the benchmark bounce diagnosis flow
  - `research`: system-delta sweep workflows
    - `sweep list-sweeps`: list committed sweeps
    - `sweep show-active`: show the active sweep alias target
    - `sweep set-active`: set the active sweep alias target
    - `sweep list`: list rows in one sweep
    - `sweep next`: show the next runnable row
    - `sweep render`: render the sweep matrix Markdown
    - `sweep validate`: validate one sweep's queue and artifacts
    - `sweep create-sweep`: create a new sweep from anchor metadata
    - `sweep execute`: execute selected sweep rows
    - `sweep graph`: render architecture graphs for sweep targets
    - `sweep promote`: promote a completed row to the sweep anchor
    - `sweep summarize`: summarize local sweep results
    - `sweep inspect`: inspect one materialized sweep row
    - `sweep diff`: diff one sweep row against another row or the anchor

Repo-local sanity check:

```bash
./scripts/dev doctor
```

For the full namespace inventory, use
[docs/development/codebase-navigation.md](docs/development/codebase-navigation.md).

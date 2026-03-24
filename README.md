# tab-foundry

Training, benchmarking, and exporting tabular ML models, with research and
workflow guidance documented in the repo docs and published docs site.

> Docs-first: start with the published docs at
> [bensonlee5.github.io/tab-foundry](https://bensonlee5.github.io/tab-foundry/)
> for the fastest route to the current workflows, architecture, and research
> context.

## Start Here

Use the docs site first, then fall back to the local Markdown docs when you
are working in the repo.

| If you want to... | Start here | Then go deeper |
| --- | --- | --- |
| Understand what this repo does | [docs/what-is-tab-foundry.md](docs/what-is-tab-foundry.md) | [docs/getting-started.md](docs/getting-started.md) |
| Get oriented quickly | [docs/getting-started.md](docs/getting-started.md) | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Work on research or sweeps | [docs/research-contributors.md](docs/research-contributors.md) | [program.md](program.md) and [docs/workflows.md](docs/workflows.md) |
| Work on artifacts or infra | [docs/ml-engineering.md](docs/ml-engineering.md) | [docs/inference.md](docs/inference.md) and [docs/workflows.md](docs/workflows.md) |
| Find exact command syntax | [docs/workflows.md](docs/workflows.md) | [docs/development/codebase-navigation.md](docs/development/codebase-navigation.md) |

## Environment

- Python `3.14` (pinned in `.python-version`)
- `uv` workflow for sync, tooling, and commands

## Setup

```bash
./scripts/dev bootstrap
```

`./scripts/dev bootstrap` wraps the canonical repo-local setup (`uv sync` plus
`pre-commit install`). For deeper setup and workflow details, use
[docs/workflows.md](docs/workflows.md).

## Finding Commands

`tab-foundry` is the canonical packaged CLI for data, dev, train, eval,
export, bench, and research workflows. Treat `./scripts/dev` as a repo-local
convenience wrapper only for bootstrap, doctor, ready, verification, and Iris
smoke.

### Entry Points

| Surface | Use it for |
| --- | --- |
| `tab-foundry` | Packaged CLI for data, dev, train, eval, export, bench, and research workflows |
| `./scripts/dev` | Repo-local bootstrap, doctor, ready, verification, and Iris smoke |
| `scripts/bench/` | Standalone internal benchmark helpers when a runbook calls for them explicitly |
| `scripts/materialize_tf_rd_013_support.py` | TF-RD-013 support materialization for committed support bundles |

After installation, use `tab-foundry ...` for command discovery and execution.
Use `--help` in this order:

```bash
tab-foundry --help
tab-foundry <group> --help
tab-foundry <group> <command> --help
```

### Top-Level Namespaces

| Namespace | Purpose | Read next |
| --- | --- | --- |
| `data` | Manifests, corpora, and dataset inspection | [docs/workflows.md](docs/workflows.md) |
| `dev` | Config resolution, forward checks, export checks, and run inspection | [docs/workflows.md](docs/workflows.md) |
| `train` | Training entrypoints, including prior workflows | [docs/workflows.md](docs/workflows.md) |
| `eval` | Checkpoint evaluation | [docs/workflows.md](docs/workflows.md) |
| `export` | Inference bundle export and validation | [docs/workflows.md](docs/workflows.md) and [docs/inference.md](docs/inference.md) |
| `bench` | Smoke, benchmarking, tuning, and registry flows | [docs/workflows.md](docs/workflows.md) |
| `research` | System-delta sweep management and execution | [docs/research-contributors.md](docs/research-contributors.md), [program.md](program.md), and [docs/workflows.md](docs/workflows.md) |

Repo-local sanity check:

```bash
./scripts/dev doctor
```

<details>
<summary>Full CLI tree</summary>

```text
tab-foundry
├── data                          data workflows
│   ├── manifest-inspect            inspect a manifest and preflight compatibility
│   └── corpus
│       ├── list-recipes            list tracked corpus recipes
│       ├── materialize             materialize a corpus recipe under outputs/corpora/
│       ├── inspect                 inspect a materialized corpus record
│       ├── compare                 diff two materialized corpus records
│       └── results                 list benchmark runs linked to a corpus
├── dev                           developer inspection and diagnostics
│   ├── resolve-config              compose and print the resolved config surface
│   ├── forward-check               build a model and run a synthetic forward smoke
│   ├── diff-config                 compare two resolved config surfaces
│   ├── export-check                export a checkpoint, validate, and run a smoke
│   ├── health-check                summarize run telemetry and instability signals
│   ├── run-inspect                 inspect a run directory and its artifacts
│   └── data
│       ├── build-manifest          build a manifest parquet from packed shard outputs
│       └── generate-manifest       generate a dagzoo corpus and emit a manifest
├── train                         training workflows
│   ├── run                         train from Hydra config overrides
│   └── prior
│       ├── simple                  train the exact-prior simple benchmark family
│       └── staged                  train the exact-prior staged benchmark family
├── eval                          evaluation workflows
│   └── checkpoint                  evaluate a checkpoint on a selected split
├── export                        export workflows
│   ├── bundle                      export a checkpoint as an inference bundle
│   └── validate                    validate an exported inference bundle
├── bench                         benchmark workflows
│   ├── smoke
│   │   ├── iris                    run the Iris smoke harness
│   │   └── dagzoo                  run the dagzoo smoke harness
│   ├── tune                        run the internal benchmark tuning sweep
│   ├── compare                     compare a run against external baselines
│   ├── env
│   │   └── bootstrap               bootstrap sibling benchmark environments
│   ├── bundle
│   │   └── build-openml            build an OpenML benchmark bundle
│   ├── registry
│   │   ├── register-run            register a benchmark run
│   │   └── freeze-baseline         freeze a control baseline
│   └── diagnose
│       └── bounce                  run the benchmark bounce diagnosis flow
└── research                      research workflows
    └── sweep
        ├── list-sweeps             list known sweeps
        ├── show-active             print the active sweep id
        ├── set-active              set the active sweep and regenerate aliases
        ├── create-sweep            bootstrap a new sweep from the delta catalog
        ├── list                    list queue rows in order
        ├── next                    print the next ready row
        ├── render                  render the sweep matrix as Markdown
        ├── validate                validate completed rows for a sweep
        ├── execute                 execute selected sweep rows
        ├── graph                   render architecture graphs for sweep targets
        ├── promote                 promote a completed run to the sweep anchor
        ├── summarize               summarize local sweep results into one table
        ├── inspect                 inspect a materialized sweep row
        └── diff                    diff a sweep row against the anchor or another row
```

</details>

For the full namespace inventory, use
[docs/development/codebase-navigation.md](docs/development/codebase-navigation.md).

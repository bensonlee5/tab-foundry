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
| `train` | Training entrypoints, with `train run` as the default surface and `train legacy-prior` as the exact-prior legacy lane | [docs/workflows.md](docs/workflows.md) |
| `eval` | Checkpoint evaluation | [docs/workflows.md](docs/workflows.md) |
| `export` | Inference bundle export and validation | [docs/workflows.md](docs/workflows.md) and [docs/inference.md](docs/inference.md) |
| `bench` | Smoke, benchmarking, tuning, and registry flows | [docs/workflows.md](docs/workflows.md) |
| `research` | System-delta sweep management and execution | [docs/research-contributors.md](docs/research-contributors.md), [program.md](program.md), and [docs/workflows.md](docs/workflows.md) |

Repo-local sanity check:

```bash
./scripts/dev doctor
```

For the canonical leaf-command inventory, use
[docs/development/codebase-navigation.md](docs/development/codebase-navigation.md).

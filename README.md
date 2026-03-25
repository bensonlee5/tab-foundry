# tab-foundry

A tabular foundation model that generates the data it learns from.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-bensonlee5.github.io%2Ftab--foundry-blue)](https://bensonlee5.github.io/tab-foundry/)

Most tabular foundation models learn from a fixed corpus and stop. You get a
`.predict()` call and a benchmark number, but no control over the data the
model trained on, the architecture it uses, or the training loop that produced
it.

**tab-foundry** takes a different approach. It uses
[dagzoo](https://github.com/bensonlee5/dagzoo) to generate synthetic tabular
datasets, trains a modular staged model on them, benchmarks against real-world
tasks, and exports inference bundles you can deploy. You control the full
pipeline: what data gets generated, which architecture stages are active, how
training runs, and what gets exported.

## How It Works

```
dagzoo           manifest         staged model       benchmark        export
(generate)  -->  (prepare)   -->  (train)       -->  (evaluate)  -->  (bundle)
   |                                                      |
   +------------- curriculum feedback (planned) ----------+
```

1. **Generate** synthetic tabular datasets with dagzoo, or bring your own
   real-data manifests
1. **Train** a modular staged foundation model with swappable architecture
   components
1. **Benchmark** against pinned OpenML evaluation bundles with tracked
   baselines
1. **Export** inference bundles for downstream deployment
1. **(Planned)** Close the loop: the model tells dagzoo what harder data it
   needs next

## Quick Start

```bash
# Clone and bootstrap
git clone https://github.com/bensonlee5/tab-foundry.git
cd tab-foundry
./scripts/dev bootstrap

# Run a smoke training loop
tab-foundry train run experiment=cls_smoke

# Evaluate the checkpoint
tab-foundry eval checkpoint \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  experiment=cls_smoke
```

For full setup details, see [docs/getting-started.md](docs/getting-started.md).

## What Makes This Different

- **Full pipeline control.** Data generation, architecture selection, training,
  benchmarking, and export are all in one repo. You own the entire stack, not
  just the prediction API.

- **Synthetic data engine.** Dagzoo generates tabular datasets with controlled
  shape, complexity, and regime coverage. You decide what the model trains on
  rather than hoping a fixed corpus covers your use case.

- **Modular staged architecture.** The model is built from explicit stages: cell
  blocks, row pooling, column encoding, context encoding, and class heads. Swap
  any subsystem independently and measure the effect in isolation.

## What Works Today

- **Staged row-first architecture** inspired by TabICL v2, with a frozen
  nanoTabPFN control lane for trusted comparison
- **Dagzoo integration** for synthetic corpus generation, manifests, and
  materialization
- **OpenML benchmarking** against pinned binary and multiclass evaluation
  bundles with a tracked benchmark registry
- **Research sweep framework** for systematic architecture and data-surface
  experiments with full attribution
- **Export pipeline** for packaging inference bundles
- **Evidence-backed decisions** every architecture choice has a pinned
  benchmark, sweep result, and research card

## What We're Building

- **Active learning loop** where the model requests harder synthetic data from
  dagzoo based on its weaknesses
- **Pluggable data sources** with a unified interface for synthetic and real
  datasets
- **Curriculum control** so users can design training progressions instead of
  filtering data
- **Distributed training** across checkpoints contributed by different users
- **Perpetually evolving model** that improves as the community contributes
  compute and data

## Architecture at a Glance

The active model family (`tabfoundry_staged`) decomposes a tabular foundation
model into explicit subsystems:

```
input table --> shared normalization + shifted-grouped tokenizer
            --> label-token target conditioning
            --> prenorm cell blocks (test-self attention)
            --> (optional) column encoder (TFCol)
            --> row-CLS pooling
            --> QASS context encoder
            --> class head
```

A frozen nanoTabPFN control lane (`tabfoundry_simple`) preserves benchmark
comparability. For the full architecture reference, see
[docs/development/model-architecture.md](docs/development/model-architecture.md).

## Find Your Path

| If you want to... | Start here | Then go deeper |
| --- | --- | --- |
| Understand what this repo does | [What is tab-foundry?](docs/what-is-tab-foundry.md) | [Getting started](docs/getting-started.md) |
| Run research sweeps | [Research contributors](docs/research-contributors.md) | [Research program](program.md) |
| Work on artifacts or infra | [ML engineering](docs/ml-engineering.md) | [Inference & export](docs/inference.md) |

## Docs and Resources

- [Published docs site](https://bensonlee5.github.io/tab-foundry/) for the
  fastest route to workflows, architecture, and research context
- [Roadmap](docs/development/roadmap.md) for what's active, planned, and
  completed
- [Architecture reference](docs/development/model-architecture.md) for the full
  model surface
- [Workflows](docs/workflows.md) for exact command syntax and artifact
  expectations
- [Glossary](docs/glossary.md) for shared vocabulary

## License and Contributing

tab-foundry is released under the [MIT License](LICENSE).

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development workflow, code standards, and how to run the test suite.

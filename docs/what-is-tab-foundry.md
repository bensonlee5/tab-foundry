# What Is tab-foundry?

Start here when you want the shortest useful explanation of what
`tab-foundry` owns, what it produces, and where to go next in the docs.

## Overview

`tab-foundry` is a tabular foundation model that generates its own training
data, trains on it, and predicts on new tasks.

It uses [dagzoo](https://github.com/bensonlee5/dagzoo) to generate synthetic
tabular datasets with controlled shape, complexity, and regime coverage. A
modular staged model trains on those datasets, benchmarks against real-world
evaluation bundles, and exports inference bundles for deployment. You control
the full pipeline: what data gets generated, which architecture stages are
active, how training runs, and what gets exported.

This is not a general-purpose serving system. It owns:

- synthetic data generation and manifest preparation
- model training with swappable architecture components
- systematic benchmarking with tracked baselines and research evidence
- inference artifact export for downstream runtime code

## What Goes In

The main inputs are:

- [dagzoo](https://github.com/bensonlee5/dagzoo) corpus recipes or
  real-data [manifests](glossary.md#manifest) describing the data/tasks to
  train on
- configuration values that define the model architecture and training recipe
- pinned benchmark bundles used for evaluation
- sweep metadata that defines which research change is being tested

## What Comes Out

The main outputs are:

- synthetic training datasets materialized from dagzoo corpus recipes
- [run directories](glossary.md#run-directory) with logs, histories, and
  summaries
- [checkpoints](glossary.md#checkpoint) that store trained model state
- benchmark comparisons against pinned real-world baselines
- [export bundles](glossary.md#export-bundle) for downstream inference handoff
- research artifacts explaining what changed and what evidence was collected

## Who This Repo Is For

These entry points cover the most common ways people approach the repo:

- repo orientation:
  - use [docs/getting-started.md](getting-started.md)
- research contributions:
  - use [docs/research-contributors.md](research-contributors.md)
- ML engineering / infra:
  - use [docs/ml-engineering.md](ml-engineering.md)

## What It Does Not Try To Do

`tab-foundry` does not try to be:

- the final production serving layer
- a generic dashboard for all model operations
- a general machine learning tutorial

It assumes the training and research side is here, while long-lived runtime
ownership can live elsewhere.

## If You Want To Go Deeper

- [docs/research-contributors.md](research-contributors.md): research path
- [docs/ml-engineering.md](ml-engineering.md): artifact and workflow path
- [docs/workflows.md](workflows.md): canonical commands and artifacts
- [docs/inference.md](inference.md): export and inference handoff contract

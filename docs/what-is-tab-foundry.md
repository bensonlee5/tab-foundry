# What Is tab-foundry?

This page provides a concise overview of what the repository does, what it
produces, and how to route into the rest of the docs.

## Overview

`tab-foundry` is a repository for training, comparing, and studying tabular
machine learning models.

It is not a general-purpose serving system. Instead, it is the place where the
team:

- prepares and inspects the training/evaluation data surface
- trains and compares model variants
- records research evidence about what changed and why
- exports inference artifacts that downstream runtime code can consume

If you think of the ML lifecycle in stages, this repo owns the training and
research side of the process, plus the packaging of inference artifacts.

## What Goes In

The main inputs are:

- [manifests](glossary.md#manifest) describing the data/tasks to train on
- configuration values that define the model and training recipe
- pinned benchmark bundles used for comparison
- sweep metadata that defines which research change is being tested

## What Comes Out

The main outputs are:

- [run directories](glossary.md#run-directory) with logs, histories, and
  summaries
- [checkpoints](glossary.md#checkpoint) that store trained model state
- benchmark comparisons against pinned baselines
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

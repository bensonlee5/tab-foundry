# Getting Started

Use this page for a fast orientation to `tab-foundry`.

## Overview

`tab-foundry` is a training and research repository for tabular models.
It takes curated data descriptions, training configs, and benchmark settings,
then produces things like [manifests](glossary.md#manifest), training
[run directories](glossary.md#run-directory),
[checkpoints](glossary.md#checkpoint), metrics, and
[export bundles](glossary.md#export-bundle).

Most readers should first decide which of these questions they are trying to
answer:

1. what does this repo do?
1. how do I contribute to the research side?
1. how do the artifacts and workflows fit together operationally?

Related docs:

- repo overview: [docs/what-is-tab-foundry.md](what-is-tab-foundry.md)
- research path: [docs/research-contributors.md](research-contributors.md)
- ML engineering / infra path: [docs/ml-engineering.md](ml-engineering.md)
- shared vocabulary: [docs/glossary.md](glossary.md)
- contribution workflow: [CONTRIBUTING.md](../CONTRIBUTING.md)

## Choose Your Path

### What This Repo Does

Start here for a concise overview of the repo, what it produces, and how the
main contributor paths fit together.

- [docs/what-is-tab-foundry.md](what-is-tab-foundry.md)

### Research Contributors

Start here if you need to understand the active architecture, sweeps, synthetic
data work, and how to frame broader model capability changes.

- [docs/research-contributors.md](research-contributors.md)

### ML Engineering / Infra

Start here if you care most about operational artifacts, verification paths,
exports, and where this repo hands off to downstream runtime ownership.

- [docs/ml-engineering.md](ml-engineering.md)

## Core Concepts

These terms appear everywhere in the docs:

- [model](glossary.md#model): the learned tabular network family being trained
  and compared
- [manifest](glossary.md#manifest): the dataset/task description the training
  and evaluation flows consume
- [sweep](glossary.md#sweep): a bounded research campaign with one anchor and a
  queue of isolated changes
- [checkpoint](glossary.md#checkpoint): a saved training state that can be
  evaluated or exported
- [export bundle](glossary.md#export-bundle): the packaged inference artifact
  this repo produces for downstream runtime use

## What The Repo Produces

At a high level, `tab-foundry` produces:

- training runs with metrics and histories
- checkpoints for evaluation and comparison
- benchmark results against pinned bundles
- export bundles for downstream inference ownership
- sweep artifacts that explain why a research change was tried and how it
  performed

## If You Only Have 15 Minutes

1. Read [docs/what-is-tab-foundry.md](what-is-tab-foundry.md).
1. Skim [docs/glossary.md](glossary.md).
1. Choose one path:
   - [docs/research-contributors.md](research-contributors.md)
   - [docs/ml-engineering.md](ml-engineering.md)
1. Use [CONTRIBUTING.md](../CONTRIBUTING.md) before editing code or sweep state.

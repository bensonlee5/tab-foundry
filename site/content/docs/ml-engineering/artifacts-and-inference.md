---
title: "Artifacts & Inference"
linkTitle: "Artifacts & Inference"
description: "Start here for manifests, runs, checkpoints, export bundles, and the runtime handoff boundary."
weight: 10
aliases:
  - /docs/artifacts-and-inference/
---

Use this page when the question is about the files and contracts this repo
owns.

`tab-foundry` turns manifests, corpora, and training configs into run
directories, checkpoints, benchmark results, and export bundles. This repo
stops at producing validated handoff artifacts; downstream runtime systems own
long-lived serving.

## Start With

- [ML Engineering]({{< relref "/docs/ml-engineering/_index.md" >}}): the
  fastest operational route through repo ownership, artifacts, and verification
  flows
- [Workflows]({{< relref "/docs/ml-engineering/workflows.md" >}}): command
  syntax and artifact expectations
- [Inference Contract]({{< relref "/docs/ml-engineering/inference.md" >}}):
  export-bundle schema and runtime handoff boundary
- [Getting Started]({{< relref "/docs/getting-started/_index.md" >}}): repo
  orientation and shared terminology

## Key Questions

- What does a training run emit?
- What is the difference between a manifest, a checkpoint, and an export
  bundle?
- How do I inspect one run or validate one bundle?
- What does this repo own, and what belongs in downstream runtime systems?

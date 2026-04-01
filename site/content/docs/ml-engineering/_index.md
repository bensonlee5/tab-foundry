---
title: "ML Engineering And Infra"
linkTitle: "ML Engineering"
description: "Operational route through workflows, artifacts, export bundles, and runtime handoff boundaries."
weight: 20
no_list: true
---

Use this route when the question is about operational artifacts, runbooks,
validation, or export/runtime handoff.

## Core Entry Points

- [Artifacts & Inference]({{< relref "/docs/ml-engineering/artifacts-and-inference.md" >}}):
  artifact mental model and ownership boundary.
- [Workflows]({{< relref "/docs/ml-engineering/workflows.md" >}}): command
  syntax and artifact expectations.
- [Inference Contract]({{< relref "/docs/ml-engineering/inference.md" >}}):
  export-bundle schema and runtime handoff details.
- [Codebase Navigation]({{< relref "/docs/development/codebase-navigation.md" >}}):
  package ownership for CLI, benchmark, export, and support code.

## Use This Route When

- you care about manifests, corpora, runs, checkpoints, or benchmark outputs
- you need the shortest path from an artifact on disk to the command that
  produces or validates it
- you are changing verification, packaging, benchmarking, or export wiring

## Pair With

- [Getting Started]({{< relref "/docs/getting-started/_index.md" >}}) if you
  need repo orientation first
- [Research]({{< relref "/docs/research-contributors/_index.md" >}}) if the
  operational change is tied to an active sweep or architecture question

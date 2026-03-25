---
title: "Artifacts & Inference"
linkTitle: "Artifacts & Inference"
description: "Start here for manifests, runs, checkpoints, export bundles, and runtime handoff boundaries."
weight: 10
aliases:
  - /docs/artifacts-and-inference/
---

______________________________________________________________________

## Overview

Use this page when the question is about the files and contracts this repo
owns.

`tab-foundry` takes data descriptions and training configs, then produces
\[run directories\]({{< relref "/docs/getting-started/glossary.md" >}}#run-directory),
\[checkpoints\]({{< relref "/docs/getting-started/glossary.md" >}}#checkpoint),
benchmark results, and \[export bundles\]({{< relref "/docs/getting-started/glossary.md" >}}#export-bundle).
This repo stops at producing validated handoff artifacts; downstream runtime
systems own long-lived serving.

## Where To Start

- \[ML Engineering & Infra\]({{< relref "/docs/ml-engineering/_index.md" >}}): the
  fastest operational route through repo ownership, artifacts, and verification
  flows.
- \[Workflows\]({{< relref "/docs/ml-engineering/workflows.md" >}}): command syntax and artifact
  expectations.
- \[Inference Contract\]({{< relref "/docs/ml-engineering/inference.md" >}}): export-bundle
  schema and runtime handoff boundary.
- \[Getting Started\]({{< relref "/docs/getting-started/_index.md" >}}): general repo
  orientation and glossary-linked terminology.

## Key Questions This Page Helps Answer

- What does a training run emit?
- What is the difference between a manifest, a checkpoint, and an export bundle?
- How do I inspect one run or validate one bundle?
- What does this repo own, and what belongs in downstream runtime systems?

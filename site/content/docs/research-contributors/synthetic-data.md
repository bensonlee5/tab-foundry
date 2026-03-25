---
title: "Synthetic Data"
linkTitle: "Synthetic Data"
description: "How dagzoo-backed synthetic corpora fit into training, comparison, and roadmap work."
weight: 30
aliases:
  - /docs/synthetic-data/
---

______________________________________________________________________

## Overview

`dagzoo` is the repo's synthetic-data lane. It is not the same thing as the
curated real-data benchmark ladders used for benchmark comparison.

Use this page when you need to answer:

- where synthetic corpora fit into the repo
- how synthetic-data work differs from real-data ladder work
- which docs govern the current data-source contract

## Where To Start

Read these pages together:

- \[Research Contributors\]({{< relref "/docs/research-contributors/_index.md" >}}):
  synthetic-data path and inspect-first commands.
- \[Dataset Curation\]({{< relref "/docs/development/dataset-curation.md" >}}):
  boundary between synthetic and curated real-data surfaces.
- \[Roadmap\]({{< relref "/docs/development/roadmap.md" >}}): the TF-RD-013 and
  training-surface sequencing context.
- \[TF-RD-013 Support Notes\]({{< relref "/docs/reference/tf-rd-013-support.md" >}}):
  current dagzoo support bundle and materialization assumptions.

The core distinction:

- `dagzoo`: synthetic training-surface and provenance work
- OpenML and vetted external datasets: curated real-data comparator ladders

---
title: "Architecture"
linkTitle: "Architecture"
description: "High-level entry point into the active tab-foundry model surface and the supporting development docs."
weight: 15
aliases:
  - /docs/architecture/
---

## Overview

Use this page when you want the shortest path from "what model is this repo
training?" to the docs and files that actually answer it.

The quick mental model is:

- one model family is frozen for trust and comparison
- one staged family is active for new architecture work
- the deeper design and code-layout canon lives under Development docs

If you want a broader repo overview first, start with
[What Is tab-foundry?]({{< relref "/docs/getting-started/what-is-tab-foundry.md" >}}) or
[Getting Started]({{< relref "/docs/getting-started/_index.md" >}}) first.

## Where To Start

Start with these pages:

- [Research Contributors]({{< relref "/docs/research-contributors/_index.md" >}}):
  research-first route through the active architecture and sweep surfaces.
- [Model Architecture]({{< relref "/docs/development/model-architecture.md" >}}):
  the current staged/simple architecture reference.
- [Design Decisions]({{< relref "/docs/development/design-decisions.md" >}}):
  enduring architecture direction and repo structure policy.
- [Codebase Navigation]({{< relref "/docs/development/codebase-navigation.md" >}}):
  where workflows and modules live.
- [Module Dependency Map]({{< relref "/docs/development/module-dependency-map.md" >}}):
  the maintained import graph and hotspot boundaries.

Then inspect the model surface directly:

- `src/tab_foundry/model/architectures/tabfoundry_staged/`
- `src/tab_foundry/model/factory.py`
- `src/tab_foundry/model/spec.py`

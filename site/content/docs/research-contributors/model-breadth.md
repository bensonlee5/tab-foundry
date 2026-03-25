---
title: "Model Breadth"
linkTitle: "Model Breadth"
description: "Entry point for many-class, regression, scaling, and later capability-expansion questions."
weight: 40
aliases:
  - /docs/model-breadth/
---

______________________________________________________________________

## Overview

Use this page when the question is about expanding the model beyond the current
classification anchor.

In this repo, "model breadth" means capability expansion such as many-class
support, regression, scaling, or later inference-related surfaces. These
changes are roadmap-governed, not ad hoc feature work.

If you need the contributor framing first, start with
[Research Contributors]({{< relref "/docs/research-contributors/_index.md" >}}).

## Where To Start

Primary roadmap lanes:

- `TF-RD-010`: many-class promotion
- `TF-RD-015`: regression rebuild
- `TF-RD-012`: inference handoff and later modalities
- `TF-RD-009`: scaling-law measurement

Start with:

- [Roadmap]({{< relref "/docs/development/roadmap.md" >}})
- [Model Architecture]({{< relref "/docs/development/model-architecture.md" >}})
- [Evidence Mapping]({{< relref "/docs/reference/evidence.md" >}})
- [Papers And References]({{< relref "/docs/reference/papers.md" >}})

Default rule: breadth proposals should extend the active `tabfoundry_staged`
surface and should not create a second live model family.

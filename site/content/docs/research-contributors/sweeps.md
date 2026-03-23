---
title: "Sweeps"
linkTitle: "Sweeps"
description: "Start here for the active sweep contract, queue discipline, and inspect-first workflow."
weight: 10
aliases:
- /docs/sweeps/
---

## Overview

Use this page when the question is about the active [sweep]({{< relref "/docs/getting-started/glossary.md" >}}#sweep),
the current [anchor]({{< relref "/docs/getting-started/glossary.md" >}}#anchor),
or the next allowed change.

A sweep is the repo's way of testing one research change at a time without
losing attribution. If you need the full research workflow, pair this page with
[Research Contributors]({{< relref "/docs/research-contributors/_index.md" >}}).

## Where To Start

Primary docs:

- [Sweep Contract]({{< relref "/docs/research-contributors/sweep-contract.md" >}}): queue discipline,
  locked surface, artifact requirements, and execution loop.
- [Workflows]({{< relref "/docs/ml-engineering/workflows.md" >}}): command syntax for inspect,
  execute, promote, render, and validate.
- [Roadmap]({{< relref "/docs/development/roadmap.md" >}}): which questions are
  actually next.
- [Reference Index]({{< relref "/docs/reference/_index.md" >}}): evidence and
  literature context.

Inspect-first commands:

```bash
.venv/bin/tab-foundry research sweep list
.venv/bin/tab-foundry research sweep next
.venv/bin/tab-foundry research sweep summarize --include-screened
.venv/bin/tab-foundry research sweep inspect --order <order> --sweep-id <sweep_id>
```

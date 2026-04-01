---
title: "Sweeps"
linkTitle: "Sweeps"
description: "Start here for the active sweep contract, queue discipline, and inspect-first workflow."
weight: 10
aliases:
  - /docs/sweeps/
---

Use this page when the question is about the active sweep, the current anchor,
or the next allowed research change.

A sweep is the repo's way of testing one research change at a time without
losing attribution. If you need the broader architecture or synthetic-data
context, pair this page with
[Research]({{< relref "/docs/research-contributors/_index.md" >}}).

## Start With

- [Sweep Contract]({{< relref "/docs/research-contributors/sweep-contract.md" >}}):
  queue discipline, locked surface, artifact requirements, and execution loop
- [Workflows]({{< relref "/docs/ml-engineering/workflows.md" >}}): command
  syntax for inspect, execute, promote, render, and validate
- [Roadmap]({{< relref "/docs/development/roadmap.md" >}}): which questions are
  actually next
- [References]({{< relref "/docs/reference" >}}): evidence and literature
  context

## Inspect-First Commands

```bash
.venv/bin/tab-foundry research sweep list-sweeps
.venv/bin/tab-foundry research sweep list --sweep-id <sweep_id>
.venv/bin/tab-foundry research sweep next --sweep-id <sweep_id>
.venv/bin/tab-foundry research sweep summarize --sweep-id <sweep_id> --include-screened
.venv/bin/tab-foundry research sweep inspect --order <order> --sweep-id <sweep_id>
```

---
title: "Documentation"
linkTitle: "Docs"
description: "Start here to choose the right docs path for repo overview, research work, or ML engineering."
weight: 1
no_list: true
---

## Start Here

Start with the question you are trying to answer, then take the shortest path
into the repo.

<dl class="path-list mb-5">
  <dt><a href="{{< relref "/docs/getting-started/what-is-tab-foundry.md" >}}">What This Repo Does &rarr;</a></dt>
  <dd>Repo overview before diving into training, sweeps, or runtime artifacts.</dd>

  <dt><a href="{{< relref "/docs/research-contributors/_index.md" >}}">Research Contributors &rarr;</a></dt>
  <dd>Architecture questions, sweep rules, synthetic-data work, and broader model capability proposals.</dd>

  <dt><a href="{{< relref "/docs/ml-engineering/_index.md" >}}">ML Engineering &amp; Infra &rarr;</a></dt>
  <dd>Artifact contracts, operational flows, export bundles, and inference handoff boundaries.</dd>
</dl>

## Explore By Topic

These topic pages are the next layer down once you know which surface you need.

<div class="row row-cols-1 row-cols-md-2 g-3 mb-4">
  <div class="col">
    <div class="card card--topic h-100">
      <div class="card-body position-relative">
        <h3 class="h5 card-title mb-1">
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/development/architecture.md" >}}">Architecture</a>
        </h3>
        <p class="card-text text-body-secondary mb-0">Active model surface, frozen control lane, and code layout.</p>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="card card--topic h-100">
      <div class="card-body position-relative">
        <h3 class="h5 card-title mb-1">
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/research-contributors/sweeps.md" >}}">Sweeps</a>
        </h3>
        <p class="card-text text-body-secondary mb-0">Anchor rules, queue discipline, and inspect-first commands.</p>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="card card--topic h-100">
      <div class="card-body position-relative">
        <h3 class="h5 card-title mb-1">
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/research-contributors/synthetic-data.md" >}}">Synthetic Data</a>
        </h3>
        <p class="card-text text-body-secondary mb-0">How <code>dagzoo</code> relates to real-data ladders and manifest surfaces.</p>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="card card--topic h-100">
      <div class="card-body position-relative">
        <h3 class="h5 card-title mb-1">
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/ml-engineering/artifacts-and-inference.md" >}}">Artifacts &amp; Inference</a>
        </h3>
        <p class="card-text text-body-secondary mb-0">Manifests, run directories, checkpoints, and export bundles.</p>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="card card--topic h-100">
      <div class="card-body position-relative">
        <h3 class="h5 card-title mb-1">
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/research-contributors/model-breadth.md" >}}">Model Breadth</a>
        </h3>
        <p class="card-text text-body-secondary mb-0">Many-class, regression, scaling, and later capability expansion.</p>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="card card--topic h-100">
      <div class="card-body position-relative">
        <h3 class="h5 card-title mb-1">
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/reference/_index.md" >}}">References</a>
        </h3>
        <p class="card-text text-body-secondary mb-0">Papers, evidence maps, and repo-local research notes.</p>
      </div>
    </div>
  </div>
</div>

## Canonical Deep References

- \[Workflows\]({{< relref "/docs/ml-engineering/workflows.md" >}}): command syntax and artifact
  expectations.
- \[Inference Contract\]({{< relref "/docs/ml-engineering/inference.md" >}}): export bundle
  schema and inference handoff contract.
- \[Sweep Contract\]({{< relref "/docs/research-contributors/sweep-contract.md" >}}): active
  system-delta execution rules.
- \[Development Docs\]({{< relref "/docs/development/_index.md" >}}): roadmap,
  architecture, design decisions, config, and module boundaries.
- \[References\]({{< relref "/docs/reference/_index.md" >}}): papers, evidence,
  and repo-local research notes.
- \[Contributing\]({{< relref "/docs/getting-started/contributing.md" >}}): how to choose the
  right unit of work and prepare a branch for review.

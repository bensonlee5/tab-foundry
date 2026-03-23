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

<div class="row row-cols-1 row-cols-md-2 row-cols-xl-3 g-4 mb-5">
  <div class="col d-flex">
    <div class="card card--audience h-100 shadow-sm">
      <div class="card-body position-relative">
        <h3 class="h4 card-title">
          <i class="fas fa-microscope text-primary me-2"></i>
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/getting-started/what-is-tab-foundry.md" >}}">What This Repo Does</a>
        </h3>
        <p class="card-text mb-0">
          Use this route for a repo overview before diving into training,
          sweeps, or runtime artifacts.
        </p>
      </div>
    </div>
  </div>
  <div class="col d-flex">
    <div class="card card--audience h-100 shadow-sm">
      <div class="card-body position-relative">
        <h3 class="h4 card-title">
          <i class="fas fa-flask text-primary me-2"></i>
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/research-contributors/_index.md" >}}">Research Contributors</a>
        </h3>
        <p class="card-text mb-0">
          Follow this path for architecture questions, sweep rules,
          synthetic-data work, and broader model capability proposals.
        </p>
      </div>
    </div>
  </div>
  <div class="col d-flex">
    <div class="card card--audience h-100 shadow-sm">
      <div class="card-body position-relative">
        <h3 class="h4 card-title">
          <i class="fas fa-server text-primary me-2"></i>
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/ml-engineering/_index.md" >}}">ML Engineering &amp; Infra</a>
        </h3>
        <p class="card-text mb-0">
          Use this route for artifact contracts, operational flows, export
          bundles, and inference handoff boundaries.
        </p>
      </div>
    </div>
  </div>
</div>

## Explore By Topic

These topic pages are the next layer down once you know which surface you need.

<div class="row row-cols-1 row-cols-md-2 row-cols-xl-3 g-4 mb-4">
  <div class="col d-flex">
    <div class="card card--topic h-100 shadow-sm">
      <div class="card-body position-relative">
        <h3 class="h5 card-title">
          <i class="fas fa-sitemap text-secondary me-2"></i>
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/development/architecture.md" >}}">Architecture</a>
        </h3>
        <p class="card-text mb-0">
          Active model surface, frozen control lane, and code layout.
        </p>
      </div>
    </div>
  </div>
  <div class="col d-flex">
    <div class="card card--topic h-100 shadow-sm">
      <div class="card-body position-relative">
        <h3 class="h5 card-title">
          <i class="fas fa-list-ol text-secondary me-2"></i>
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/research-contributors/sweeps.md" >}}">Sweeps</a>
        </h3>
        <p class="card-text mb-0">
          Anchor rules, queue discipline, and inspect-first commands.
        </p>
      </div>
    </div>
  </div>
  <div class="col d-flex">
    <div class="card card--topic h-100 shadow-sm">
      <div class="card-body position-relative">
        <h3 class="h5 card-title">
          <i class="fas fa-database text-secondary me-2"></i>
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/research-contributors/synthetic-data.md" >}}">Synthetic Data</a>
        </h3>
        <p class="card-text mb-0">
          How <code>dagzoo</code> relates to real-data ladders and manifest
          surfaces.
        </p>
      </div>
    </div>
  </div>
  <div class="col d-flex">
    <div class="card card--topic h-100 shadow-sm">
      <div class="card-body position-relative">
        <h3 class="h5 card-title">
          <i class="fas fa-box-archive text-secondary me-2"></i>
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/ml-engineering/artifacts-and-inference.md" >}}">Artifacts &amp; Inference</a>
        </h3>
        <p class="card-text mb-0">
          Manifests, run directories, checkpoints, export bundles, and
          validation paths.
        </p>
      </div>
    </div>
  </div>
  <div class="col d-flex">
    <div class="card card--topic h-100 shadow-sm">
      <div class="card-body position-relative">
        <h3 class="h5 card-title">
          <i class="fas fa-expand text-secondary me-2"></i>
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/research-contributors/model-breadth.md" >}}">Model Breadth</a>
        </h3>
        <p class="card-text mb-0">
          Many-class, regression, scaling, and later capability expansion.
        </p>
      </div>
    </div>
  </div>
  <div class="col d-flex">
    <div class="card card--topic h-100 shadow-sm">
      <div class="card-body position-relative">
        <h3 class="h5 card-title">
          <i class="fas fa-book text-secondary me-2"></i>
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/reference/_index.md" >}}">References</a>
        </h3>
        <p class="card-text mb-0">
          Papers, evidence maps, and repo-local research notes.
        </p>
      </div>
    </div>
  </div>
</div>

## Canonical Deep References

- [Workflows]({{< relref "/docs/ml-engineering/workflows.md" >}}): command syntax and artifact
  expectations.
- [Inference Contract]({{< relref "/docs/ml-engineering/inference.md" >}}): export bundle
  schema and inference handoff contract.
- [Sweep Contract]({{< relref "/docs/research-contributors/sweep-contract.md" >}}): active
  system-delta execution rules.
- [Development Docs]({{< relref "/docs/development/_index.md" >}}): roadmap,
  architecture, design decisions, config, and module boundaries.
- [References]({{< relref "/docs/reference/_index.md" >}}): papers, evidence,
  and repo-local research notes.
- [Contributing]({{< relref "/docs/getting-started/contributing.md" >}}): how to choose the
  right unit of work and prepare a branch for review.

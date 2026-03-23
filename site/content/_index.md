---
title: "tab-foundry"
linkTitle: "Home"
---

{{< blocks/cover title="tab-foundry" image_anchor="top" height="med" color="dark" >}}

<p class="lead mt-3">Docs, workflows, and research paths for training, comparing, and exporting tabular ML models.</p>
<a class="btn btn-lg btn-outline-light me-3 mb-4" href="{{< relref "/docs" >}}">
  Read the Docs
</a>
<a class="btn btn-lg btn-secondary me-3 mb-4" href="https://github.com/bensonlee5/tab-foundry">
  GitHub <i class="fab fa-github ms-2"></i>
</a>
{{< /blocks/cover >}}

## Choose Your Path

Start with the question you are trying to answer, then follow the shortest
route into the repo.

<div class="row row-cols-1 row-cols-md-2 row-cols-xl-3 g-4 mb-5">
  <div class="col d-flex">
    <div class="card card--audience h-100 shadow-sm">
      <div class="card-body position-relative">
        <h3 class="h4 card-title">
          <i class="fas fa-microscope text-primary me-2"></i>
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/getting-started/what-is-tab-foundry.md" >}}">What This Repo Does</a>
        </h3>
        <p class="card-text mb-0">
          Start here for an overview of what <code>tab-foundry</code> does,
          what it produces, and where it fits in the ML lifecycle.
        </p>
      </div>
    </div>
  </div>
  <div class="col d-flex">
    <div class="card card--audience h-100 shadow-sm">
      <div class="card-body position-relative">
        <h3 class="h4 card-title">
          <i class="fas fa-flask text-primary me-2"></i>
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/research-contributors/_index.md" >}}">Contribute Research</a>
        </h3>
        <p class="card-text mb-0">
          Use this path for architecture work, sweep discipline, synthetic-data
          questions, and proposals to broaden model capability.
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
          Follow this route if you care most about manifests, runs,
          checkpoints, export bundles, verification paths, and inference
          handoff boundaries.
        </p>
      </div>
    </div>
  </div>
</div>

## Explore By Topic

Once you know your audience path, use these pages to go deeper into one
surface at a time.

<div class="row row-cols-1 row-cols-md-2 row-cols-xl-3 g-4 mb-5">
  <div class="col d-flex">
    <div class="card card--topic h-100 shadow-sm">
      <div class="card-body position-relative">
        <h3 class="h5 card-title">
          <i class="fas fa-sitemap text-secondary me-2"></i>
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/development/architecture.md" >}}">Architecture</a>
        </h3>
        <p class="card-text mb-0">
          Learn which model family is active, which one is frozen for
          comparison, and where the code and architecture canon live.
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
          Understand the anchor-only workflow, the next runnable row, and the
          artifacts needed to make research results attributable.
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
          See how <code>dagzoo</code> fits into the repo and how synthetic data
          differs from curated real-data benchmark ladders.
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
          Focus on manifests, run directories, checkpoints, export bundles, and
          what this repo owns before downstream runtime systems take over.
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
          Track the roadmap lanes for many-class work, regression, scaling, and
          broader capability expansion.
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
          Browse papers, evidence maps, and repo-local notes that shape
          architecture and data decisions.
        </p>
      </div>
    </div>
  </div>
</div>

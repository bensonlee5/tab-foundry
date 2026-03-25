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

<dl class="path-list mb-5">
  <dt><a href="{{< relref "/docs/getting-started/what-is-tab-foundry.md" >}}">What This Repo Does &rarr;</a></dt>
  <dd>Overview of what <code>tab-foundry</code> does, what it produces, and where it fits in the ML lifecycle.</dd>

  <dt><a href="{{< relref "/docs/research-contributors/_index.md" >}}">Contribute Research &rarr;</a></dt>
  <dd>Architecture work, sweep discipline, synthetic-data questions, and proposals to broaden model capability.</dd>

  <dt><a href="{{< relref "/docs/ml-engineering/_index.md" >}}">ML Engineering &amp; Infra &rarr;</a></dt>
  <dd>Manifests, runs, checkpoints, export bundles, verification paths, and inference handoff boundaries.</dd>
</dl>

## Explore By Topic

Once you know your audience path, use these pages to go deeper into one
surface at a time.

<div class="row row-cols-1 row-cols-md-2 g-3 mb-5">
  <div class="col">
    <div class="card card--topic h-100">
      <div class="card-body position-relative">
        <h3 class="h5 card-title mb-1">
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/development/architecture.md" >}}">Architecture</a>
        </h3>
        <p class="card-text text-body-secondary mb-0">
          Active model surface, frozen control lane, and code layout.
        </p>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="card card--topic h-100">
      <div class="card-body position-relative">
        <h3 class="h5 card-title mb-1">
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/research-contributors/sweeps.md" >}}">Sweeps</a>
        </h3>
        <p class="card-text text-body-secondary mb-0">
          Anchor rules, queue discipline, and inspect-first commands.
        </p>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="card card--topic h-100">
      <div class="card-body position-relative">
        <h3 class="h5 card-title mb-1">
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/research-contributors/synthetic-data.md" >}}">Synthetic Data</a>
        </h3>
        <p class="card-text text-body-secondary mb-0">
          How <code>dagzoo</code> relates to real-data ladders and manifest surfaces.
        </p>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="card card--topic h-100">
      <div class="card-body position-relative">
        <h3 class="h5 card-title mb-1">
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/ml-engineering/artifacts-and-inference.md" >}}">Artifacts &amp; Inference</a>
        </h3>
        <p class="card-text text-body-secondary mb-0">
          Manifests, run directories, checkpoints, and export bundles.
        </p>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="card card--topic h-100">
      <div class="card-body position-relative">
        <h3 class="h5 card-title mb-1">
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/research-contributors/model-breadth.md" >}}">Model Breadth</a>
        </h3>
        <p class="card-text text-body-secondary mb-0">
          Many-class, regression, scaling, and later capability expansion.
        </p>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="card card--topic h-100">
      <div class="card-body position-relative">
        <h3 class="h5 card-title mb-1">
          <a class="stretched-link text-decoration-none" href="{{< relref "/docs/reference/_index.md" >}}">References</a>
        </h3>
        <p class="card-text text-body-secondary mb-0">
          Papers, evidence maps, and repo-local research notes.
        </p>
      </div>
    </div>
  </div>
</div>

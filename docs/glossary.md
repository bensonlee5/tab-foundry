# Glossary

Use this glossary when architecture, sweep, artifact, and workflow terms start
to carry too much repo-specific meaning.

## Anchor

The locked comparison run for a sweep. New rows are judged against the anchor
unless the queue row explicitly declares a different preserved surface.

## Architecture-Screen Surface

The canonical benchmark-facing staged surface used for future architecture work.
In current docs this is the regular `cls_benchmark_staged` lane, distinct from
hybrid diagnostic rows.

## Benchmark Bundle

A pinned benchmark task set used to compare runs consistently. It defines the
shared comparison surface for benchmark-facing claims.

## Benchmark-Full

A sweep execution policy that continues past screening, runs the pinned
benchmark bundle, registers the run, and requires a `result_card.md`.

## Checkpoint

A saved training state that can be evaluated, compared, resumed, or exported.

## Control Lane

The frozen PFN-style comparison lane:

- `tabfoundry_simple`
- `tabfoundry_staged` with `stage=nano_exact`

Use it for trust and comparison, not as the default landing zone for new
architecture work.

## Dagzoo Synthetic-Data Lane

The synthetic-data source lane used when `tab-foundry` reasons about generated
corpora. It is separate from curated real-data benchmark ladders.

## Delta

One named change package in the system-delta workflow. A delta is the logical
unit a queue row applies against the anchor.

## Export Bundle

A packaged inference artifact produced by this repo for downstream runtime use.
In the current contract it contains a manifest and weights file.

## Hybrid Diagnostic Lane

A staged surface that holds some PFN-adjacent assumptions fixed while isolating
one question. Useful for diagnosis, but not automatically benchmark-facing
promotion evidence.

## Inference Handoff

The boundary where this repo stops at producing validated export artifacts and
another repo or runtime system takes over serving or long-lived inference
ownership.

## Manifest

The concrete data/task description consumed by training, evaluation, and
inspection flows.

## Manifest-Backed Data

The canonical concrete data surface in the repo today. Training, evaluation,
and inspection operate on manifests rather than on ad hoc loader paths.

## Model

The learned tabular network family being trained, evaluated, and compared in
the repo.

## Model Breadth

Capability expansion beyond the settled classification anchor, such as
many-class support, regression, later modalities, inference handoff, or
scaling-law breadth.

## Prior Training

Training against a prior-data surface instead of only replaying benchmark tasks.
In this repo it usually refers to the prior-dump or manifest-backed staged
training surfaces used to compare architectures under a fixed recipe.

## Row

One queue entry in a sweep. A row isolates one declared dimension family and
records the description, rationale, artifacts, metrics, interpretation, and
next action.

## Run Directory

The output directory for one training run. It usually contains histories,
summaries, checkpoints, and other artifacts produced during the run.

## Screen-Only

A sweep execution policy for diagnostic rows that stop after screening metrics.
These rows do not register benchmark runs and do not create a `result_card.md`.

## Sweep

A bounded research campaign with one active anchor, one queue, and one rendered
matrix. Completed sweeps remain historical evidence rather than being mutated in
place.

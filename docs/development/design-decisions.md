# Design Decisions

This document records the current high-level decisions shaping the repo.

## Decision: Prioritize scaling predictability ahead of mode breadth

Status: accepted for roadmap direction

Reason:

- the primary goal of the project is Chinchilla-like predictability on scalability
- the repo needs a stable scaling-oriented control family before it broadens into many prediction modes
- without this prioritization, the backlog risks treating benchmark tweaks, regression work, and tertiary modalities as peers

Consequence:

- Goal 1 work is scaling predictability first
- classification remains the anchor workload until regression is promoted into the same framework
- regression comes before missing-data prediction
- many-class, time series, text embeddings, and similar modalities are explicitly tertiary
- external benchmark results remain a constraint and feedback signal, not the top-level project objective

## Decision: Mirror dagzoo's role-based structure

Status: accepted for roadmap direction

Reason:

- the current package root mixes core library code with workflow helpers
- `dagzoo` has a cleaner separation between CLI, benchmark tooling, domain packages, config, and development docs
- mirroring that structure reduces future reorg cost as the repo grows

Consequence:

- new roadmap work should bias toward `cli/`, `bench/`, domain packages, and `docs/development/`
- future workflow helpers should not land as new top-level package modules by default

## Decision: Use `bench/` as the neutral home for harness tooling

Status: accepted for roadmap direction

Reason:

- the repo needs a neutral home for smoke flows, benchmark utilities, plotting helpers, and tuning harnesses
- `bench/` matches the precedent in `dagzoo`
- it avoids conflating correctness checks with the broader benchmark and harness surface

Consequence:

- smoke, Iris, nanoTabPFN comparison, env bootstrap, and similar workflow code should converge under `bench/`

## Decision: Treat `TabICLv2` as a legacy compatibility concept

Status: accepted for roadmap direction

Reason:

- the current implementation is a starting point, not the settled architecture identity for the repo
- future work needs space for multiple families and external-baseline-inspired variants
- current export and checkpoint compatibility still refer to `tabiclv2`

Consequence:

- new internal architecture work should use neutral family naming
- export/inference compatibility can continue to mention `tabiclv2` until a separate migration is planned

## Decision: Make QASS optional by construction

Status: accepted for roadmap direction

Reason:

- the user wants clean comparisons against non-QASS alternatives
- future architecture work should not assume QASS is mandatory

Consequence:

- shared components and family configs should support both QASS and non-QASS paths under the same training/eval stack

## Decision: Prepare for prior and source modularity early

Status: accepted for roadmap direction

Reason:

- the user wants freedom to experiment not only with architecture but eventually with priors and task sources
- the current manifest-backed path is useful, but it should not become a hidden hard constraint on the repo shape

Consequence:

- roadmap work should add a neutral source interface under `data/`
- manifest remains the only concrete source initially
- future prior/source experiments should plug into the same training and benchmarking framework rather than creating one-off paths

## Decision: Build for external baseline borrowing and scaling-law work

Status: accepted for roadmap direction

Reason:

- the repo should intentionally absorb ideas from `nanoTabPFN`, `nanochat`, and adjacent transformer repos
- the desired end state includes predictable scaling rather than ad hoc model growth

Consequence:

- roadmap tickets should include named baseline families/configs, scaling instrumentation, and Chinchilla-style planning support

## Decision: Add a reference area before major architecture changes

Status: accepted for roadmap direction

Reason:

- the intended workflow is literature-first, similar to `~/dev/dagzoo/reference`

Consequence:

- major architecture tickets should cite curated references from `reference/` before implementation begins

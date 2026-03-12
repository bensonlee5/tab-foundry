# Architecture Strategy

This document captures the intended architectural direction for `tab-foundry`.

It should be read together with:

- `docs/development/roadmap.md`
- `docs/development/codebase-navigation.md`
- `docs/development/design-decisions.md`
- `reference/PAPERS.md`

## Direction

The repository should not ossify around `TabICLv2` as the final architecture identity.

`TabICLv2` is the current starting point, but the architecture should evolve through:

- modular building blocks
- explicit baseline comparisons against adjacent repos
- literature-guided exploration
- scaling-law-driven planning rather than one-off model guesses

The primary project objective is predictable scaling behavior. Broader prediction-mode coverage comes after that, and tertiary modalities are intentionally deferred further still.

## Design Goals

### 0. Scaling predictability first

The first design goal is Chinchilla-like predictability on scalability.

That means:

- classification remains the anchor workload for near-term architecture work
- architecture and training changes should be judged first by whether they improve controllable scaling behavior
- benchmark gains matter, but they are a constraint and feedback signal rather than the top-level identity of the project

Only after the scaling-oriented foundation is stable should the roadmap prioritize regression, then missing-data prediction, and only later broader modalities such as many-class expansion, time series, or text-embedding-conditioned inputs

### 1. Neutral architecture naming

The repo should move toward neutral model naming and architecture registration.

The goal is:

- internal code should not assume `TabICLv2` is the long-term architecture name
- multiple model families should be able to coexist behind a shared builder/factory surface
- the current `tabiclv2` export contract should be treated as a compatibility constraint, not as the final taxonomy

### 2. Modular model construction

The model stack should be decomposable so components can be swapped without forking the repo.

Examples:

- feature tokenization choices
- target conditioning strategy
- row/column encoder choices
- QASS versus non-QASS attention
- backbone depth and width schedules
- readout and head choices

This should support clean comparisons instead of architecture edits that bundle multiple ideas together.

### 3. External baseline ingestion

The repo should be able to absorb ideas from adjacent codebases without rewriting the whole stack each time.

Important sources of inspiration include:

- `nanoTabPFN`
- `karpathy/nanochat`
- more general transformer repos for optimizer, LR schedule, model sizing, and training infrastructure ideas

The practical goal is to turn these into named baseline configurations or modular blocks that can be compared directly.

Default transfer rule:

- the repo should resemble `nanochat` in compact-transformer recipe, FFN choice, residual/pre-norm hygiene, optimizer partitioning, and sizing discipline where those ideas do not depend on sequence order
- the repo should not inherit language-sequence assumptions by default
- row and column structure should preferentially draw from set- and permutation-aware references rather than from positional language-model machinery such as RoPE

### 4. Dagzoo-style repo organization

The repo should move toward the same kind of neutral role-based structure used in `dagzoo`.

That means:

- workflow tooling should have a neutral home such as `bench/`
- architecture families should be separated from reusable components
- development docs should converge under `docs/development/`
- literature and evidence notes should have a first-class `reference/` area

This is meant to reduce future reorg cost as architecture and prior experimentation expands.

### 5. Scaling-law readiness

The ideal end state is a model family that can be scaled predictably.

That means the repo should support:

- consistent size sweeps
- clean accounting of train compute and parameter count
- comparison across depth/width choices
- the ability to fit Chinchilla-like scaling trends for this model family

The aspiration is not necessarily to copy `nanochat`, but to make scaling predictable in the same spirit.

### 6. Literature-first construction

Architecture changes should begin with a literature and reference pass, similar in spirit to `~/dagzoo/reference`.

That means:

- collect papers and repo references before major refactors
- summarize why a candidate mechanism is relevant
- record what signal would count as success or failure before coding

## What This Means For The Backlog

The architecture epic should include foundation work before later ablations:

- literature search and reference curation
- neutral naming and architecture registry design
- pluggable QASS and non-QASS infrastructure
- external baseline configurations inspired by adjacent repos
- scaling-law measurement and planning support

Only after that foundation is in place should the repo lean heavily into architecture-specific ablation work.

Prediction-mode expansion should follow after the scaling-oriented architecture platform is stable:

1. classification as the anchor mode
1. regression
1. missing-data prediction
1. tertiary modalities later

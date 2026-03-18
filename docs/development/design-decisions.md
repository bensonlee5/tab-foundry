# Design Decisions

This document captures enduring architecture direction, repo-structure policy,
and compatibility guidance for `tab-foundry`.

Related docs:

- quickstart: `README.md`
- workflow runbooks: `docs/workflows.md`
- canonical roadmap: `docs/development/roadmap.md`
- architecture reference: `docs/development/model-architecture.md`
- codebase navigation: `docs/development/codebase-navigation.md`
- maintained dependency map: `docs/development/module-dependency-map.md`
- reference index: `reference/README.md`
- curated references: `reference/papers.md`
- evidence mapping: `reference/evidence.md`

## Direction

`tab-foundry` should not ossify around the current `tabfoundry` family as the
final architecture identity.

The current `tabfoundry` family started from the TabICLv2 paper and adjacent
references, but the repo should evolve through:

- modular building blocks
- explicit baseline comparisons against adjacent repos
- literature-guided exploration
- scaling-law-driven planning rather than one-off model guesses

The primary project objective is predictable scaling behavior. Broader
prediction-mode coverage comes after that, and tertiary modalities are
intentionally deferred further still.

## Prioritization Lens

- Scaling predictability comes first.
- Classification remains the anchor workload until the scaling-oriented control
  family is stable.
- Regression comes next, then missing-data prediction.
- Many-class expansion beyond maintenance, time series, text-conditioned inputs,
  and similar modalities are explicitly later work.
- Benchmark gains matter, but they are a constraint and feedback signal rather
  than the top-level identity of the project.

## Enduring Decisions

### Neutral Architecture Naming

- Internal code should not assume `tabfoundry` is the long-term architecture
  name.
- Multiple model families should be able to coexist behind a shared builder or
  factory surface.
- The current `tabfoundry` export contract is a user-facing boundary, not the
  final internal taxonomy.

### Modular Model Construction

The model stack should be decomposable enough to compare changes without
forking the repo. Core swap points include:

- feature tokenization choices
- target conditioning strategy
- row and column encoder choices
- QASS versus non-QASS attention
- backbone depth and width schedules
- readout and head choices

### QASS Remains Optional

- Shared components and family configs should support both QASS and non-QASS
  paths.
- Architecture work should not assume QASS is structurally mandatory.
- Comparisons should run through the same training and evaluation stack.

### Role-Based Repo Structure

The repo should keep the same role-based direction already started in code:

- workflow tooling in `bench/`
- user-facing command surfaces in `cli/`
- Python workflow entrypoints should live under the packaged CLI rather than
  reappearing under `scripts/`
- `scripts/` should stay limited to shell helpers and audit tooling rather than
  becoming a second home for benchmark logic
- reusable data, model, training, and export packages separated by role
- canonical planning and repo-shape docs under `docs/development/`
- stable operational docs such as `docs/workflows.md` and `docs/inference.md`
  at the top level
- literature and evidence notes under `reference/`, indexed from
  `reference/README.md`

### External Baseline Borrowing And Literature-First Construction

- Major architecture changes should begin from curated references rather than
  ad hoc intuition.
- Borrow compact-transformer recipe ideas from `nanochat` when they do not
  depend on sequence order.
- Prefer set- and permutation-aware references for row and column structure.
- Treat language-sequence positional machinery as low priority by default.
- Turn adjacent repo ideas into named baselines or modular options rather than
  one-off edits.

### Prior And Source Modularity

- Manifest-backed data is the canonical concrete source today.
- The repo should still prepare early for future source and prior
  experimentation.
- Future prior or source work should plug into the same training and
  benchmarking framework rather than creating parallel paths.

### Scaling-Law Readiness

The end state should support:

- consistent size sweeps
- clean accounting of train compute and parameter count
- comparison across depth and width choices
- the ability to fit Chinchilla-like scaling trends for the model family

## Package Roles And Target Boundaries

| Path | Intended role |
| ---- | ------------- |
| `src/tab_foundry/` | Stable top-level package namespace for CLI entrypoints, shared config resolution, and small cross-cutting helpers. |
| `src/tab_foundry/cli/` and `src/tab_foundry/cli/groups/` | User-facing command surfaces and nested argument parsing. |
| `src/tab_foundry/bench/` | Smoke harnesses, benchmark utilities, comparison flows, plotting helpers, env bootstrap, and internal research harnesses. |
| `src/tab_foundry/data/` and `src/tab_foundry/data/sources/` | Data package namespaces for reusable dataset abstractions, loaders, and registered task sources. Direct imports should target modules such as `tab_foundry.data.manifest`, `tab_foundry.data.dataset`, and `tab_foundry.data.factory`. |
| `src/tab_foundry/model/` | Model package namespace. Direct imports should target submodules such as `tab_foundry.model.factory`, `tab_foundry.model.spec`, or family modules under `tab_foundry.model.architectures`. |
| `src/tab_foundry/model/components/` | Reusable architectural pieces such as tokenization blocks, attention blocks, many-class helpers, and QASS primitives. |
| `src/tab_foundry/model/architectures/` | Full model-family implementations assembled from reusable components. |
| `src/tab_foundry/training/` | Family-agnostic training infrastructure such as schedules, optimizers, trainer loops, checkpointing, and history logging. |
| `src/tab_foundry/export/` | Export package namespace for bundle contracts and compatibility handling. Direct imports should target modules such as `tab_foundry.export.contracts`, `tab_foundry.export.exporter`, and `tab_foundry.export.loader_ref`. |
| `docs/development/` | Canonical planning, architecture rationale, codebase navigation, and dependency mapping for internal repo evolution. |
| `docs/workflows.md` and `docs/inference.md` | Stable operational and contract docs that should remain easy to link for users and downstream repos. |
| `tests/` | Coverage organized by role today across `tests/model/`, `tests/training/`, `tests/data/`, `tests/export/`, `tests/runtime/`, `tests/config/`, `tests/smoke/`, and `tests/benchmark/`. |

## Dependency Direction

The intended dependency direction is:

```text
cli/groups -> cli
cli/groups -> bench/research
cli/groups -> training/data/export
bench -> training/data/model/export
research -> bench/model/config
training -> model/data
export -> model
model/architectures -> model/components
data/sources -> data
```

Notes:

- `bench/` may depend on core library packages, but core library packages
  should not depend on `bench/`.
- `cli/groups/` may orchestrate both `bench/`, `research/`, and core packages.
- `src/tab_foundry/model/` is a namespace package. Direct imports should target
  stable submodules like `model.factory` and `model.spec`, while
  `model/components` and `model/architectures` carry the internal split.
- Current export compatibility constraints around `tabfoundry` are part of the
  boundary, but should not re-anchor internal architecture structure.

## Naming And Compatibility Guidance

- Prefer neutral family ids over architecture names that assume permanence.
- Use `tabfoundry` for the current family rather than a paper-derived,
  versioned name.
- Export and inference compatibility now follow the current family id; any
  future rename must be planned as an explicit schema migration.
- Optimize naming for future coexistence of multiple model families, data
  sources, and baseline-inspired variants.

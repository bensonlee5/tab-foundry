# Architecture

This document captures the intended architectural direction, repo structure, and enduring design decisions for `tab-foundry`.

Related docs:

- quickstart: `README.md`
- workflow runbooks: `docs/workflows.md`
- roadmap: `docs/roadmap.md`
- curated references: `reference/papers.md`
- evidence mapping: `reference/evidence.md`

## Direction

`tab-foundry` should not ossify around `TabICLv2` as the final architecture identity.

`TabICLv2` is the current starting point, but the repo should evolve through:

- modular building blocks
- explicit baseline comparisons against adjacent repos
- literature-guided exploration
- scaling-law-driven planning rather than one-off model guesses

The primary project objective is predictable scaling behavior. Broader prediction-mode coverage comes after that, and tertiary modalities are intentionally deferred further still.

## Prioritization Lens

- Scaling predictability comes first.
- Classification remains the anchor workload until the scaling-oriented control family is stable.
- Regression comes next, then missing-data prediction.
- Many-class expansion beyond maintenance, time series, text-conditioned inputs, and similar modalities are explicitly later work.
- Benchmark gains matter, but they are a constraint and feedback signal rather than the top-level identity of the project.

## Enduring Decisions

### Neutral Architecture Naming

- Internal code should not assume `TabICLv2` is the long-term architecture name.
- Multiple model families should be able to coexist behind a shared builder or factory surface.
- The current `tabiclv2` export contract is a compatibility constraint, not the final internal taxonomy.

### Modular Model Construction

The model stack should be decomposable enough to compare changes without forking the repo. Core swap points include:

- feature tokenization choices
- target conditioning strategy
- row and column encoder choices
- QASS versus non-QASS attention
- backbone depth and width schedules
- readout and head choices

### QASS Remains Optional

- Shared components and family configs should support both QASS and non-QASS paths.
- Architecture work should not assume QASS is structurally mandatory.
- Comparisons should run through the same training and evaluation stack.

### Role-Based Repo Structure

The repo should keep the same role-based direction already started in code:

- workflow tooling in `bench/`
- user-facing command surfaces in `cli/`
- reusable data, model, training, and export packages separated by role
- roadmap and operational docs under `docs/`
- literature and evidence notes under `reference/`

### External Baseline Borrowing And Literature-First Construction

- Major architecture changes should begin from curated references rather than ad hoc intuition.
- Borrow compact-transformer recipe ideas from `nanochat` when they do not depend on sequence order.
- Prefer set- and permutation-aware references for row and column structure.
- Treat language-sequence positional machinery as low priority by default.
- Turn adjacent repo ideas into named baselines or modular options rather than one-off edits.

### Prior And Source Modularity

- Manifest-backed data is the canonical concrete source today.
- The repo should still prepare early for future source and prior experimentation.
- Future prior or source work should plug into the same training and benchmarking framework rather than creating parallel paths.

### Scaling-Law Readiness

The end state should support:

- consistent size sweeps
- clean accounting of train compute and parameter count
- comparison across depth and width choices
- the ability to fit Chinchilla-like scaling trends for the model family

## Package Roles

| Path | Intended role |
| ---- | ------------- |
| `src/tab_foundry/` | Stable shared surfaces such as CLI entrypoints, shared config resolution, and small cross-cutting helpers. |
| `src/tab_foundry/cli/` and `src/tab_foundry/cli/commands/` | User-facing command surfaces and argument parsing. |
| `src/tab_foundry/bench/` | Smoke harnesses, benchmark utilities, comparison flows, plotting helpers, env bootstrap, and internal research harnesses. |
| `src/tab_foundry/data/` and `src/tab_foundry/data/sources/` | Reusable dataset abstractions, loaders, and registered task sources. |
| `src/tab_foundry/model/components/` | Reusable architectural pieces such as tokenization blocks, attention blocks, heads, and neutral model outputs. |
| `src/tab_foundry/model/architectures/` | Full model-family implementations assembled from reusable components. |
| `src/tab_foundry/training/` | Family-agnostic training infrastructure such as schedules, optimizers, trainer loops, checkpointing, and history logging. |
| `src/tab_foundry/export/` | Bundle/export contracts and compatibility handling. |
| `tests/` | Coverage organized by role: `tests/model/`, `tests/training/`, `tests/data/`, `tests/export/`, `tests/bench/`, and `tests/cli/`. |

## Dependency Direction

The intended dependency direction is:

```text
cli/commands -> cli -> bench
cli/commands -> training/data/model/export
bench -> training/data/model/export
training -> model/data
export -> model
model/architectures -> model/components
data/sources -> data
```

Notes:

- `bench/` may depend on core library packages, but core library packages should not depend on `bench/`.
- `cli/commands/` may orchestrate both `bench/` and core packages.
- `model/components/` should remain reusable and family-agnostic.
- `model/architectures/` should assemble components into named families.
- Current export compatibility constraints around `tabiclv2` are tolerated at the boundary, but should not re-anchor internal architecture structure.

## Naming And Compatibility Guidance

- Prefer neutral family ids over architecture names that assume permanence.
- Treat `tabiclv2` as a compatibility alias, not the preferred internal organizing term.
- Keep export and inference compatibility stable until a dedicated migration is planned.
- Optimize naming for future coexistence of multiple model families, data sources, and baseline-inspired variants.

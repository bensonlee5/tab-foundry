# Codebase Navigation

This document captures the intended package roles for `tab-foundry`.

It is descriptive for now. It defines where new code should trend even before the full structural refactor lands.

## Top-Level Direction

`tab-foundry` should look more like `dagzoo`:

- neutral role-based packages
- clear separation between core library code and workflow tooling
- development docs under `docs/development/`
- literature and external-baseline notes under `reference/`

## Intended Package Roles

### `src/tab_foundry/`

Keep only stable shared surfaces here, such as:

- CLI entrypoints that truly belong at the package boundary
- shared config resolution
- neutral types and small cross-cutting helpers

The package root should not become the long-term home for smoke harnesses, benchmark scripts, or research runners.

### `src/tab_foundry/cli/` and `src/tab_foundry/cli/commands/`

Use these for user-facing command surfaces and argument parsing.

This is the long-term home for workflow entrypoints that should be callable through a unified CLI instead of thin script wrappers.

### `src/tab_foundry/bench/`

Use `bench/` as the neutral home for harness and benchmark tooling, including:

- dagzoo smoke flows
- Iris checks
- nanoTabPFN comparison flows
- checkpoint-backed evaluation helpers
- plotting helpers tied to benchmark outputs
- sibling-env bootstrap
- internal research harnesses that operate on completed runs

This mirrors `dagzoo`'s `bench/` naming and intentionally avoids a more loaded label like `sanity/`.

### `src/tab_foundry/data/` and `src/tab_foundry/data/sources/`

Use `data/` for reusable dataset abstractions and loaders.

Use `data/sources/` for registered task sources. Manifest-backed data is the initial concrete source, but the structure should be ready for future prior/source experimentation.

### `src/tab_foundry/model/components/`

Use this for reusable architectural pieces:

- tokenization blocks
- attention blocks
- QASS and non-QASS-capable primitives
- heads and readouts
- neutral model output types

### `src/tab_foundry/model/architectures/`

Use this for full model-family implementations assembled from reusable components.

The current `TabICLv2` implementation should eventually live here as one family among others.

### `src/tab_foundry/training/`

Use this for family-agnostic training infrastructure:

- schedules
- optimizers
- trainer loops
- checkpointing
- metric/history logging

### `src/tab_foundry/export/`

Use this for bundle/export contracts and compatibility handling.

Current `tabiclv2` naming constraints belong here as legacy compatibility rules until a later migration changes the contract.

## Naming Guidance

- prefer neutral family ids over architecture names that assume permanence
- treat `tabiclv2` as a compatibility alias, not the preferred internal organizing term
- when in doubt, optimize naming for future coexistence of multiple model families

## Test Structure Direction

Tests should mirror the package roles over time:

- `tests/model/`, `tests/training/`, `tests/data/`, `tests/export/` for core library coverage
- `tests/bench/` for smoke, benchmark, and harness flows
- `tests/cli/` for command-level behavior

## Related Docs

- `docs/development/roadmap.md`
- `docs/development/design-decisions.md`
- `docs/ARCHITECTURE_STRATEGY.md`

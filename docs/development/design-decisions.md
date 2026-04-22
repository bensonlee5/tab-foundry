# Design Decisions

Use this page to understand the durable decisions behind the current
architecture, repo structure, and compatibility boundaries.

Use these alongside this page:

- quickstart: `README.md`
- problem formulation: `docs/development/synthetic-prior-mission.md`
- workflow runbooks: `docs/workflows.md`
- canonical roadmap: `docs/development/roadmap.md`
- architecture reference: `docs/development/model-architecture.md`
- codebase navigation: `docs/development/codebase-navigation.md`
- maintained dependency map: `docs/development/module-dependency-map.md`
- reference index: `reference/README.md`
- curated references: `reference/papers.md`
- evidence mapping: `reference/evidence.md`

## Direction

`tab-foundry` should not fragment across multiple live model families.

The active architecture surface is now `grid_sandwich`, with
`tabfoundry_sandwich` retained as the previous carried in-family comparison,
`tabfoundry_simple` retained only as the frozen exact anchor, and
`tabfoundry_staged` retained only as the historical reference line. The repo
should evolve through:

- modular building blocks
- explicit baseline comparisons against adjacent repos
- literature-guided exploration
- scaling-law-driven planning rather than one-off model guesses

Near-term architecture direction is now explicit:

- keep a frozen PFN-style control lane for benchmark trust
- evolve the TF-RD-026 row `10` `grid_sandwich` anchor as the coherent primary
  classification family
- keep `tabfoundry_staged` available only for historical comparison and
  compatibility
- remain free to borrow specific components from TabPFN or other references
  when they fit better than a literal TabICLv2 copy

The primary project objective is predictable scaling behavior. Broader
prediction-mode coverage comes after that, and tertiary modalities are
intentionally deferred further still.

## Prioritization Lens

- Scaling predictability comes first.
- Classification remains the anchor workload until the scaling-oriented control
  family and the row-first classification target are stable.
- Training-surface adequacy and at least one harder post-008 ladder should be
  settled before selective low-level architecture expansion or scaling-law work
  becomes the main next source of evidence.
- Regression is intentionally deferred until it can be rebuilt on top of
  `tabfoundry_staged`.
- Many-class expansion beyond maintenance, time series, text-conditioned inputs,
  and similar modalities are explicitly later work.
- Benchmark gains matter, but they are a constraint and feedback signal rather
  than the top-level identity of the project.

## Enduring Decisions

### Single Active Architecture Surface

- Internal code should optimize for one active model-development surface:
  `grid_sandwich`, currently the two-layer recurrent SwiGLU grid core promoted
  by TF-RD-026 row `10`.
- `tabfoundry_simple` remains only as the frozen compatibility anchor.
- `tabfoundry_staged` remains only as a historical comparison or
  compatibility surface.
- New feature work should not create a second live family unless it is planned
  as an explicit replacement of `grid_sandwich`.

### PFN Control, Row-First Target

- `tabfoundry_simple` and `stage=nano_exact` are the frozen PFN-style control
  lane.
- `grid_sandwich` is the active architecture target.
- `tabfoundry_sandwich` remains the previous carried in-family comparison
  surface.
- `tabfoundry_staged` remains the historical row-first reference line rather
  than the active development destination.
- Architecture promotion should prefer coherent grid-preserving surfaces over piling
  structurally unrelated deltas onto the PFN control path or extending the
  staged reference line indefinitely.

### Modular Model Construction

The model stack should be decomposable enough to compare changes without
forking the repo. Core swap points include:

- feature tokenization choices
- target conditioning strategy
- row and column encoder choices
- QASS versus non-QASS attention
- backbone depth and width schedules
- readout and head choices
- bounded low-level follow-up such as norm placement or family, initialization
  choices, and scaler capacity only after harder-surface evidence shows they
  are decision-relevant
- coherent staged-surface promotion rather than override accumulation

### QASS Remains Optional

- Shared components and family configs should support both QASS and non-QASS
  paths.
- Architecture work should not assume QASS is structurally mandatory, even on a
  TabICLv2-inspired path.
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

### Single-Source Docs With Hugo Publishing

- Canonical docs stay in repo Markdown: `README.md`, `CONTRIBUTING.md`,
  `docs/`, `reference/`, and `program.md`.
- The Hugo app under `site/` is a published navigation layer, not a second
  policy surface.
- Generated site inputs under `site/.generated/` should be treated as build
  artifacts owned by the sync script.
- Contributors should edit canonical Markdown and let the sync/build workflow
  publish it.

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

## Operational Boundary Notes

Use the live owner docs for implementation-time boundaries:

- [Codebase Navigation](codebase-navigation.md) owns package and entrypoint
  routing.
- [Module Dependency Map](module-dependency-map.md) owns the current top-level
  graph plus dependency-direction policy.
- [Workflows](../workflows.md) owns command examples and artifact expectations.
- [Inference Contract](../inference.md) owns the export/runtime handoff
  contract.

## Naming And Compatibility Guidance

- Prefer family ids that reflect current ownership and scope.
- `grid_sandwich` is the active development family, `tabfoundry_sandwich` is
  the previous carried in-family comparison family, `tabfoundry_simple` is the
  frozen anchor, and `tabfoundry_staged` is the historical reference family.
- Export and inference compatibility changes still require explicit schema
  migration planning.
- Optimize naming for clear role separation, not for keeping retired families
  alive indefinitely.

# Roadmap

This is the canonical roadmap for `tab-foundry`.

The roadmap is intentionally ordered around project goals rather than around whatever is easiest to implement next.

## Project Goals In Priority Order

### Goal 1: Scaling Predictability

The primary goal of the project is Chinchilla-like predictability on scalability.

That means:

- building model families whose quality improves in a more predictable way as size and train budget change
- making architecture and training knobs measurable rather than ad hoc
- using short-run benchmarking as a constraint and feedback signal, not as the project identity

### Goal 2: Core Prediction Modes

After the scaling-oriented foundation is stable, the next goal is broader core prediction coverage in this order:

1. classification remains the anchor mode
2. regression is the next mode to flesh out
3. missing-data prediction follows regression

### Goal 3: Extended Tasks And Modalities

Only after Goal 1 and Goal 2 are in good shape should the roadmap prioritize additional task families and modalities such as:

- many-class work beyond maintenance
- time-series extensions
- text-embedding-conditioned tabular inputs
- similar future extensions

The current direction is therefore to make the repository easier to evolve in three ways at once:

- organize it more like `dagzoo`, with neutral role-based package boundaries
- treat the current `TabICLv2` implementation as one starting family, not the final architecture identity
- make future model-family, baseline, and prior/source experimentation cheap instead of repo-wide

## Working Constraints

- use benchmark quality against `nanoTabPFN` as a constraint and selection signal
- keep `tab-foundry` training in the short-run class, roughly 5-6 minutes for benchmark-facing work
- use internal sweeps to prune candidates and use OpenML benchmarking only for shortlisted runs
- keep current export/inference compatibility intact until a dedicated contract-migration effort is planned

## Current Control

The current benchmark-facing control is `experiment=cls_benchmark_linear`.

It remains the control until a later ticket explicitly promotes a replacement.

## Goal 1: Scaling Predictability

The near-term roadmap is dominated by work that improves the repo's ability to reason about predictable scaling.

The sequence is:

1. trustworthy experiment infrastructure
2. repo/package foundation
3. modular architecture platform
4. tuning for more predictable behavior
5. focused ablations in service of scaling predictability

### Epic 1: Trustworthy Experiment Platform

Goal: make experiment results reproducible, comparable, and reviewable so scaling behavior can be trusted.

Linear issues:

- `BL-152` Epic: Trustworthy Experiment Platform
- `BL-155` Freeze the canonical `cls_benchmark_linear` control baseline
- `BL-156` Pin benchmark inputs and add benchmark drift checks
- `BL-157` Add repeated-run stability evaluation for tab-foundry
- `BL-158` Add checkpoint-selection diagnostics to benchmark summaries
- `BL-159` Add experiment leaderboard output for research runs

Expected outcome:

- one frozen control run
- one pinned benchmark input bundle
- repeated-run statistics for tab-foundry
- one leaderboard that can accept baseline, sweep, and architecture rows

### Epic 2: Repo Foundation And Dagzoo-Style Organization

Goal: reorganize the repository around durable roles and extension points before deeper architecture work.

This epic mirrors the `dagzoo` pattern: keep core/domain packages focused, give workflow tooling a neutral home, and make development docs and references first-class.

Linear issues:

- `BL-175` Epic: Repo Foundation And Dagzoo-Style Organization
- `BL-177` Adopt dagzoo-style package boundaries
- `BL-178` Create a neutral `bench` home for harnesses and benchmark tooling
- `BL-179` Mirror dagzoo's docs layout under `docs/development/`
- `BL-180` Add a `reference/` area for literature and external-baseline notes

Target structure to capture and later implement:

- `src/tab_foundry/cli/`
- `src/tab_foundry/cli/commands/`
- `src/tab_foundry/bench/`
- `src/tab_foundry/data/`
- `src/tab_foundry/data/sources/`
- `src/tab_foundry/model/components/`
- `src/tab_foundry/model/architectures/`
- `src/tab_foundry/training/`
- `src/tab_foundry/export/`
- `docs/development/`
- `reference/`

Expected outcome:

- the package root contains only stable shared surfaces
- smoke, Iris, nanoTabPFN comparison, env bootstrap, plotting, and tuning live under `bench/`
- the repo is organized by role and extension point rather than by historical growth

### Epic 3: Modular Architecture Platform

Goal: make architecture experimentation cheap and structured, without centering the repo on `TabICLv2`, so scaling can be reasoned about deliberately.

Linear issues:

- `BL-154` Epic: Modular Architecture Platform
- `BL-170` Conduct literature search and curate architecture references
- `BL-171` Introduce neutral architecture naming and registry infrastructure
- `BL-181` Split reusable model components from architecture families
- `BL-172` Build modular QASS and non-QASS backbone infrastructure
- `BL-182` Add a source and prior interface for future task-source experimentation
- `BL-173` Add external baseline configs inspired by adjacent repos
- `BL-174` Add scaling-law measurement and Chinchilla-style planning support

Rule for this epic:

- architecture infrastructure comes before later ablations
- QASS remains optional rather than structurally mandatory
- external ideas from `nanoTabPFN`, `nanochat`, and adjacent transformer repos land as named baselines or modular options
- `tabiclv2` is treated as a legacy compatibility identifier, not the long-term internal taxonomy

Expected outcome:

- one literature-backed architecture foundation
- one neutral model-family registry
- one shared component surface that supports QASS and non-QASS variants
- one source/prior interface ready for future data work
- one scaling-law-ready measurement path

### Epic 4: Hyperparameter Tuning On The Current Architecture

Goal: exhaust schedule, optimizer, batching, and size tuning in service of more predictable scaling behavior before changing the model design.

Linear issues:

- `BL-153` Epic: Hyperparameter Tuning On Current Architecture
- `BL-160` Expand the schedule sweep space for current architecture tuning
- `BL-161` Sweep optimizer families under a fixed short-run budget
- `BL-162` Sweep batching and clipping for stability
- `BL-164` Sweep model size within the current architecture
- `BL-163` Promote the tuned baseline and update the comparison control

Rule for this epic:

- each tuning ticket must target an explicit architecture family and config surface
- tuning stays separate from OpenML benchmarking until candidates are shortlisted

Expected outcome:

- a tuned shortlist selected by internal metrics
- confirmatory benchmark runs only for top candidates
- one explicit decision on whether the control baseline changes

### Epic 5: Architecture Ablations Against The Control Baseline

Goal: test specific inductive-bias hypotheses only after the foundation epic is in place, and only insofar as they help identify a more predictable scaling path.

Linear issues:

- `BL-165` Ablate feature tokenization granularity
- `BL-166` Ablate target conditioning strategy
- `BL-167` Ablate repeated asymmetric row attention
- `BL-169` Ablate QASS and encoder simplification
- `BL-168` Make the post-ablation architecture decision

Rules for this epic:

- each ticket changes only one core inductive bias at a time
- each ticket must name the target family and whether it touches shared components or a family-specific implementation
- ablations are benchmarked against the frozen or promoted control baseline, not evaluated in isolation

Expected outcome:

- one architecture path to keep
- one to reject
- one to defer

## Goal 2: Core Prediction Modes

This goal begins only after Goal 1 has produced a stable scaling-oriented control family and experiment stack.

### Epic 6: Core Prediction Modes

Goal: broaden core prediction coverage in a deliberate order while preserving the scaling-oriented framework built in Goal 1.

Linear issues:

- `BL-183` Epic: Core Prediction Modes
- `BL-185` Define shared mode interfaces and evaluation contracts
- `BL-186` Reach regression parity on the common experiment and tuning stack
- `BL-187` Define missing-data prediction tasks and data-path support
- `BL-188` Add missing-data evaluation harnesses and summaries
- `BL-189` Break out leaderboard and artifact reporting by prediction mode

Rules for this epic:

- classification remains the anchor mode until regression reaches parity on the same experiment stack
- regression comes before missing-data prediction
- missing-data work is treated as a second-phase expansion, not part of the first scaling-oriented architecture cycle

Expected outcome:

- one shared prediction-mode contract across at least classification and regression
- regression support on the same sweep and benchmark-adjacent workflow class
- one clear path for adding missing-data prediction without forking the repo

## Goal 3: Extended Tasks And Modalities

This goal is explicitly tertiary.

Many-class support may continue in maintenance mode, but it is not a near-term roadmap driver until Goal 1 and Goal 2 are meaningfully complete.

### Epic 7: Extended Prediction Modes And Modalities

Goal: capture longer-horizon task and modality expansions without letting them displace the primary scaling work.

Linear issues:

- `BL-184` Epic: Extended Prediction Modes And Modalities
- `BL-191` Keep many-class support in maintenance mode until higher-priority goals are stable
- `BL-190` Scope a time-series extension path
- `BL-192` Scope text-embedding-conditioned tabular inputs
- `BL-193` Define readiness gates for future tertiary modalities

Expected outcome:

- tertiary work is visible on the roadmap without competing with Goal 1 or Goal 2
- many-class, time series, text embeddings, and similar extensions have an explicit later home

## Conditional Epic: Prior / Data Alignment

Do not open this epic by default.

Open it only if the best tuned architecture still materially trails `nanoTabPFN` at the same short-run budget, or if earlier evidence shows that data/prior mismatch is the dominant bottleneck.

If opened, it should cover:

- benchmark-aligned synthetic `dagzoo` configs
- multi-pass row-ladder corpus generation
- train-time sampling controls for oversized synthetic tasks
- direct prior/data comparison studies under the same architecture family and benchmark budget

## Standard Research Contract

Every tuning or architecture ticket should produce:

- one reproducible config or code path
- one run artifact set
- one leaderboard row
- one short written conclusion: keep, reject, or defer

Every confirmatory benchmark ticket should enforce:

- the same train-time budget class
- the same benchmark input bundle
- both `best_*` and `final_*` metrics
- explicit comparison against the frozen control baseline

The benchmark remains a constraint and evaluation surface, not the primary statement of project success.

Related docs:

- `docs/development/codebase-navigation.md`
- `docs/development/design-decisions.md`
- `docs/development/module-dependency-map.md`
- `docs/ARCHITECTURE_STRATEGY.md`
- `reference/PAPERS.md`

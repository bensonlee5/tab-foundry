# Mission-Aligned Roadmap (2026Q1)

This is the canonical roadmap for `tab-foundry`.

It mirrors the planning format used in `dagzoo`, but the content is specific to
`tab-foundry`'s current mission:

- predictable scaling for short-run compact tabular transformers
- benchmark-facing experiment discipline without making the benchmark the repo's
  identity
- modular architecture evolution instead of hard-coding the repo around one
  family
- broader prediction-mode coverage only after the scaling-oriented control path
  is stable

Related docs:

- design decisions and repo structure: `docs/development/design-decisions.md`
- codebase navigation: `docs/development/codebase-navigation.md`
- maintained dependency map: `docs/development/module-dependency-map.md`
- workflow runbooks: `docs/workflows.md`
- inference/export contract: `docs/inference.md`
- reference index: `reference/README.md`
- evidence appendix: `reference/evidence.md`

## Status Labels

- `implemented`: available in current code and wired into the canonical
  workflow surface.
- `partial`: meaningful building blocks exist, but the roadmap claim is not yet
  fully satisfied.
- `planned`: clearly scoped and prioritized, but not implemented end to end.
- `research`: exploratory, gated, or intentionally deferred to a later lane.
- `retired`: historical item that no longer represents the active roadmap.

## Canonical Planning Metadata

`docs/development/roadmap.md` is the single source of truth for planning state
in `tab-foundry`. Every active item is tracked here with:

- status and milestone lane
- canonical local roadmap ID
- priority rank
- BL issue mapping as the active tracker surface
- current state, dependencies, and exit criteria

If any other document disagrees with this file, this file is authoritative.

`tab-foundry` uses local roadmap IDs in the form `TF-RD-###`. BL-prefixed
issues remain the canonical tracker links. Rank `0` is reserved for completed
items retained for traceability.

## Scaling Predictability Prioritization Lens

Roadmap ranking is currently optimized for predictable scaling in the short-run
training regime:

- primary: build a benchmark-facing control family whose quality moves in a more
  interpretable way as model size, train budget, and recipe change
- implemented baseline: manifest-backed training/eval, short-run smoke
  workflows, and benchmark-adjacent comparison tooling already exist and should
  be used as the planning baseline
- deferred: many-class expansion beyond maintenance, time-series work,
  text-conditioned inputs, and prior/data alignment should not displace the
  scaling-first agenda unless earlier evidence makes them the dominant
  bottleneck

## Canonical Priority Queue

Lower rank means higher priority. Rank `0` is reserved for completed items kept
for traceability.

| Rank | Roadmap ID | Item | Status | Milestone | Tracker Links |
| ---- | ---------- | ---- | ------ | --------- | ------------- |
| 0 | TF-RD-002 | Repo foundation and dagzoo-style organization | implemented | Implemented | `BL-175 -> BL-177 -> BL-178 -> BL-179 -> BL-180` |
| 1 | TF-RD-001 | Canonical control baseline and experiment trust | partial | Now | `BL-152 -> BL-155 -> BL-156 -> BL-157 -> BL-158 -> BL-159` |
| 2 | TF-RD-012 | Shared preprocessing and export-contract fidelity | planned | Now | `no active issue yet; open BL preprocessing chain after TF-RD-001` |
| 3 | TF-RD-003 | Literature-first references and external baseline borrowing | partial | Now | `BL-154 -> BL-170 -> BL-173` |
| 4 | TF-RD-004 | Neutral architecture registry and shared component surfaces | planned | Now | `BL-154 -> BL-171 -> BL-181 -> BL-182` |
| 5 | TF-RD-005 | Modular QASS and non-QASS backbone infrastructure | planned | Now | `BL-154 -> BL-172` |
| 6 | TF-RD-006 | Scaling-law measurement and Chinchilla-style planning | planned | Now | `BL-154 -> BL-174` |
| 7 | TF-RD-007 | Current-architecture tuning and control promotion | partial | Now | `BL-153 -> BL-160 -> BL-161 -> BL-162 -> BL-164 -> BL-163` |
| 8 | TF-RD-008 | Architecture ablations against the control baseline | planned | Next | `BL-165 -> BL-166 -> BL-167 -> BL-169 -> BL-168` |
| 9 | TF-RD-013 | Separate-repo inference handoff and conformance | planned | Next | `no active issue yet; open BL inference chain after TF-RD-012` |
| 10 | TF-RD-009 | Core prediction mode parity | partial | Next | `BL-183 -> BL-185 -> BL-186 -> BL-187 -> BL-188 -> BL-189` |
| 11 | TF-RD-010 | Extended prediction modes and modalities | research | Later | `BL-184 -> BL-191 -> BL-190 -> BL-192 -> BL-193` |
| 12 | TF-RD-011 | Conditional prior/data alignment when architecture is not the bottleneck | research | Later | `no active issue yet; gated follow-on after TF-RD-007 and TF-RD-008` |

## Current Capability Matrix

| Roadmap Objective / Pillar Claim | Current State | Evidence in Repo | Gap | Roadmap IDs |
| --- | --- | --- | --- | --- |
| Manifest-backed packed-shard training on `dagzoo` outputs | `implemented` | Manifest build flow, manifest-backed train/eval CLI, export/load contract, and packed-shard dataset handling are present in the canonical workflow | Source and prior modularity beyond the manifest path is not yet implemented | `TF-RD-004`, `TF-RD-011` |
| Short-run reproducible experiment loop with smoke coverage | `partial` | GitHub Actions quality gate, Iris smoke, dagzoo smoke, persisted run artifacts, and pinned benchmark-bundle discipline are implemented | Repeated-run stability evaluation, checkpoint-selection diagnostics, and leaderboard output are incomplete | `TF-RD-001`, `TF-RD-007` |
| Benchmark-facing comparison against `nanoTabPFN` | `partial` | Env bootstrap and comparison harnesses exist, and the benchmark remains the selection surface for shortlist validation | Canonical promoted external-baseline configs and benchmark-trust discipline are not complete | `TF-RD-001`, `TF-RD-003`, `TF-RD-007` |
| Scaling-oriented architecture planning | `partial` | Architecture guidance, literature references, and evidence mapping are now first-class planning artifacts | Neutral registry, modular backbones, and canonical scaling-law measurement are not complete | `TF-RD-003`, `TF-RD-004`, `TF-RD-005`, `TF-RD-006` |
| Export-grade preprocessing fidelity | `partial` | Shared preprocessing code now exists and v3 bundles persist fitted feature fill values, label values, and model `input_normalization` | The fitted preprocessing surface is not yet promoted as the settled producer contract across every downstream workflow and tracker lane | `TF-RD-012` |
| Separate-runtime handoff readiness | `partial` | The repo now ships a v3 reference consumer plus golden bundle conformance fixtures while keeping runtime ownership out of this package | Separate-repo adoption, broader conformance reuse, and downstream stabilization are still unfinished | `TF-RD-013` |
| Core prediction mode coverage | `partial` | Classification and regression flows exist in the common train/eval stack | Shared mode contracts, missing-data task support, and mode-specific reporting are incomplete | `TF-RD-009` |
| Extended modality readiness | `partial` | Many-class support exists, and time-series plus text-conditioned inputs are explicitly deferred rather than ignored | No scoped time-series path, text-conditioned path, or readiness gates exist yet | `TF-RD-010` |
| Prior/source extensibility beyond the current manifest path | `partial` | Neutral package structure now exists for future extension and the roadmap explicitly preserves space for source/prior work | No concrete non-manifest source/prior interface or alignment study path exists yet | `TF-RD-004`, `TF-RD-011` |

## Current Implementation Baseline

This roadmap does not duplicate the current codebase map or every public
surface. Use the canonical docs instead:

- design decisions and repo structure: `docs/development/design-decisions.md`
- codebase navigation: `docs/development/codebase-navigation.md`
- maintained dependency map: `docs/development/module-dependency-map.md`
- workflow runbooks: `docs/workflows.md`
- reference set and evidence mapping: `reference/papers.md`,
  `reference/evidence.md`, and `reference/README.md`

### Current Control and Compatibility

- The benchmark-facing control remains `experiment=cls_benchmark_linear` until a
  later ticket explicitly promotes a replacement.
- The manifest-backed data path is the canonical training/evaluation baseline.
- Export and inference compatibility now use the `tabfoundry` contract across
  `tab-foundry-export-v2` and `tab-foundry-export-v3`.
- Bundle validation exists, `preprocessor_state.json` now has a fitted-state v3
  path in addition to the older policy-only v2 path, and executable inference
  consumption remains reference-only in this repo.
- Benchmark-facing work remains in the short-run class rather than drifting into
  long-training infrastructure by default.

### Research Workflows and Benchmark-Adjacent Tooling

- Iris smoke provides a fast repo-local manifest-backed end-to-end run.
- dagzoo smoke exercises a sibling `dagzoo` checkout and the packed-shard path.
- `nanoTabPFN` comparison tooling exists for benchmark-facing checkpoint
  comparison.
- Internal tuning harnesses exist and are intended to shortlist candidates
  before broader benchmark confirmation.

### Validation and Quality Gates

- Local quality tooling is standardized around `mdformat`, `ruff`, `mypy`, and
  `pytest`.
- GitHub Actions requires both `quality-and-unit` and `iris-smoke`.
- Smoke and tuning flows persist artifacts such as checkpoints, train-history
  logs, telemetry, and markdown summaries for review.

## Roadmap Items

### TF-RD-001: Canonical Control Baseline and Experiment Trust

- Status: `partial`
- Milestone: `Now`
- Goal alignment: `Goal 1: Scaling Predictability`
- Pillar alignment: `experiment trust`
- Goal: make benchmark-facing experiment results reproducible, comparable, and
  reviewable enough that scaling conclusions can be trusted.
- Linear tracking:
  `BL-152 -> BL-155 -> BL-156 -> BL-157 -> BL-158 -> BL-159`
- Repo touchpoints:
  - `.github/workflows/test.yml`
  - `src/tab_foundry/bench/dagzoo_smoke.py`
  - `src/tab_foundry/bench/iris_smoke.py`
  - `src/tab_foundry/bench/checkpoint.py`
  - `scripts/configure_repo_protection.sh`
- Current state:
  - The benchmark-facing control is named and documented as
    `experiment=cls_benchmark_linear`.
  - CI quality gates and smoke workflows now exist and produce persisted
    artifacts.
  - The canonical benchmark bundle is now pinned in-repo, and the comparison
    flow fails fast on benchmark-input drift.
  - Repeated-run trust, checkpoint-selection diagnostics, and leaderboard
    output are not yet fully codified.
- Exit criteria:
  - One frozen canonical control run is documented and preserved.
  - Benchmark inputs are pinned and drift-checked.
  - Repeated-run stability evaluation exists for the benchmark-facing workflow.
  - Checkpoint-selection diagnostics appear in benchmark summaries.
  - Research and confirmatory runs can land in one canonical leaderboard-style
    output.

### TF-RD-012: Shared Preprocessing and Export-Contract Fidelity

- Status: `planned`
- Milestone: `Now`
- Goal alignment: `Goal 2: Core Prediction Modes`
- Pillar alignment: `export fidelity`
- Goal: replace stubbed preprocessing metadata with one shared fitted
  preprocessing surface used by dataset loading, evaluation, export, and
  reference consumption.
- Linear tracking:
  `no active issue yet; open BL preprocessing chain after TF-RD-001`
- Repo touchpoints:
  - `src/tab_foundry/preprocessing/`
  - `src/tab_foundry/data/dataset.py`
  - `src/tab_foundry/training/trainer.py`
  - `src/tab_foundry/export/contracts.py`
  - `src/tab_foundry/export/exporter.py`
- Current state:
  - Shared preprocessing code now handles mean imputation and classification
    label remapping in the packed-task data path.
  - `tab-foundry-export-v3` now persists ordered feature ids, per-feature fill
    values, classification label values when present, and model
    `input_normalization`.
  - The broader producer contract and downstream rollout are not yet stabilized
    under one tracked BL chain.
- Exit criteria:
  - Train, eval, export, and reference consumption use one shared
    preprocessing codepath.
  - v3 bundles contain sufficient fitted state for downstream execution and
    conformance testing.
  - `input_normalization` round-trips through export and load.
  - Classification and regression both have golden v3 fixtures.

### TF-RD-002: Repo Foundation and Dagzoo-Style Organization

- Status: `implemented`
- Milestone: `Implemented`
- Goal alignment: `Goal 1: Scaling Predictability`
- Pillar alignment: `repo foundation`
- Goal: reorganize the repository around durable roles and extension points
  before deeper architecture work.
- Linear tracking: `BL-175 -> BL-177 -> BL-178 -> BL-179 -> BL-180`
- Repo touchpoints:
  - `src/tab_foundry/cli/`
  - `src/tab_foundry/cli/commands/`
  - `src/tab_foundry/bench/`
  - `src/tab_foundry/data/sources/`
  - `src/tab_foundry/model/`
  - `src/tab_foundry/model/components/`
  - `src/tab_foundry/model/architectures/`
  - `docs/development/`
  - `docs/workflows.md`
  - `reference/`
- Completion evidence (March 12, 2026):
  - Neutral `bench/` and `cli/commands/` homes now exist and act as the
    canonical destinations for workflow orchestration and harness logic.
  - Reusable model components now live separately from family-specific
    assemblies under `src/tab_foundry/model/components/` and
    `src/tab_foundry/model/architectures/`.
  - Canonical planning and repo-structure docs now live under
    `docs/development/`, and `reference/` is the stable literature and
    evidence surface.
  - The repo is now organized around durable role-based package boundaries,
    and future extensions can land without another large package reorg.
- Exit criteria:
  - The package root is role-based rather than historical.
  - Workflow helpers converge into neutral homes instead of one-off modules.
  - Development docs and references remain canonical and discoverable.
  - Future extensions can land without another large package reorg.

### TF-RD-003: Literature-First References and External Baseline Borrowing

- Status: `partial`
- Milestone: `Now`
- Goal alignment: `Goal 1: Scaling Predictability`
- Pillar alignment: `modular architecture`
- Goal: make literature and adjacent codebases a first-class input into
  architecture and recipe decisions.
- Linear tracking: `BL-154 -> BL-170 -> BL-173`
- Repo touchpoints:
  - `reference/papers.md`
  - `reference/evidence.md`
  - `docs/development/design-decisions.md`
  - `scripts/bootstrap_benchmark_envs.py`
  - `scripts/benchmark_nanotabpfn.py`
- Current state:
  - Reference documents and evidence mapping now exist and express explicit
    adoption tiers.
  - `nanoTabPFN` comparison tooling exists, and `nanochat` plus set-structured
    references are part of the planning vocabulary.
  - External baseline borrowing is still more documented than instantiated in
    named experiment families.
- Exit criteria:
  - Major architecture changes begin from curated references instead of ad hoc
    intuition.
  - At least one external-baseline-inspired config family exists in the common
    workflow stack.
  - Reference docs remain tied to roadmap IDs and acceptance signals.

### TF-RD-004: Neutral Architecture Registry and Shared Component Surfaces

- Status: `planned`
- Milestone: `Now`
- Goal alignment: `Goal 1: Scaling Predictability`
- Pillar alignment: `modular architecture`
- Goal: separate reusable model components from architecture families and
  introduce neutral family naming.
- Linear tracking: `BL-154 -> BL-171 -> BL-181 -> BL-182`
- Repo touchpoints:
  - `src/tab_foundry/model/factory.py`
  - `src/tab_foundry/data/factory.py`
  - `src/tab_foundry/data/sources/`
  - `src/tab_foundry/export/contracts.py`
  - `docs/development/design-decisions.md`
- Current state:
  - Shared factories and role-based packages exist, and reusable model
    components now live separately from family-specific assembly code.
  - The repo still centers much of the implementation around the `tabfoundry`
    family.
  - The manifest path is canonical, and source modularity is still mostly a
    roadmap direction.
  - Family-neutral registration and naming are still incomplete.
- Exit criteria:
  - Internal architecture naming is neutral rather than tied to a
    paper-derived id.
  - Reusable components are cleanly separated from family-specific assembly.
  - A source/prior interface exists without forking the training/eval stack.
  - Export compatibility broadens only through explicit schema migrations.

### TF-RD-005: Modular QASS and Non-QASS Backbone Infrastructure

- Status: `planned`
- Milestone: `Now`
- Goal alignment: `Goal 1: Scaling Predictability`
- Pillar alignment: `modular architecture`
- Goal: make QASS optional by construction and enable direct comparison against
  non-QASS alternatives in the same workflow stack.
- Linear tracking: `BL-154 -> BL-172`
- Repo touchpoints:
  - `src/tab_foundry/model/components/qass.py`
  - `src/tab_foundry/model/components/blocks.py`
  - `src/tab_foundry/model/architectures/tabfoundry.py`
  - `src/tab_foundry/model/factory.py`
  - `src/tab_foundry/training/evaluate.py`
- Current state:
  - QASS exists as a real mechanism in the current architecture.
  - The train/eval stack can already run the current family, but QASS is not
    yet one swappable option among clearly modular backbones.
  - Literature and architecture docs now explicitly call for QASS to remain
    optional.
- Exit criteria:
  - QASS and non-QASS backbones run through the same training/evaluation stack.
  - Shared components make the comparison structural rather than ad hoc.
  - Contributors can compare backbone choices without rewriting family-specific
    glue.

### TF-RD-006: Scaling-Law Measurement and Chinchilla-Style Planning

- Status: `planned`
- Milestone: `Now`
- Goal alignment: `Goal 1: Scaling Predictability`
- Pillar alignment: `scaling predictability`
- Goal: add one canonical measurement path for compute/parameter accounting and
  fitted scaling trends across model sizes.
- Linear tracking: `BL-154 -> BL-174`
- Repo touchpoints:
  - `src/tab_foundry/bench/tune.py`
  - `src/tab_foundry/bench/compare.py`
  - `src/tab_foundry/training/trainer.py`
  - `docs/development/design-decisions.md`
  - `reference/evidence.md`
- Current state:
  - Chinchilla-like scaling is the repo's primary declared objective.
  - Tuning and benchmark-adjacent tooling exist, but there is no canonical
    scaling-law artifact path yet.
  - Parameter-count and compute-trend reasoning is still mostly conceptual.
- Exit criteria:
  - Size sweeps report comparable compute and parameter metadata.
  - Scaling runs emit artifacts that can support fitted curves.
  - Planning can compare depth/width/recipe choices against explicit scaling
    evidence instead of only end metrics.

### TF-RD-007: Current-Architecture Tuning and Control Promotion

- Status: `partial`
- Milestone: `Now`
- Goal alignment: `Goal 1: Scaling Predictability`
- Pillar alignment: `scaling predictability`
- Goal: exhaust schedule, optimizer, batching, and size tuning on the current
  architecture before drawing architecture conclusions.
- Linear tracking: `BL-153 -> BL-160 -> BL-161 -> BL-162 -> BL-164 -> BL-163`
- Repo touchpoints:
  - `src/tab_foundry/training/optimizer.py`
  - `src/tab_foundry/training/schedule.py`
  - `src/tab_foundry/bench/tune.py`
  - `scripts/tune_tab_foundry.py`
  - `scripts/benchmark_nanotabpfn.py`
- Current state:
  - Internal tuning harnesses exist.
  - Multiple optimizer and schedule choices are present in the training stack.
  - The repo can shortlist candidates internally, but no promoted tuned control
    has replaced the current benchmark-facing baseline yet.
- Exit criteria:
  - A tuned shortlist is produced from internal metrics.
  - Confirmatory benchmark runs are reserved for the finalists.
  - One explicit control-promotion decision is made and documented.
  - Tuning conclusions are comparable across the short-run budget class.

### TF-RD-008: Architecture Ablations Against the Control Baseline

- Status: `planned`
- Milestone: `Next`
- Goal alignment: `Goal 1: Scaling Predictability`
- Pillar alignment: `modular architecture`
- Goal: test a small set of inductive-bias hypotheses only after the control
  path is stable enough to serve as a comparison baseline.
- Linear tracking: `BL-165 -> BL-166 -> BL-167 -> BL-169 -> BL-168`
- Repo touchpoints:
  - `src/tab_foundry/model/components/blocks.py`
  - `src/tab_foundry/model/components/qass.py`
  - `src/tab_foundry/model/architectures/tabfoundry.py`
  - `src/tab_foundry/model/components/many_class.py`
  - `reference/papers.md`
- Current state:
  - Tokenization, target conditioning, row-attention, and simplification
    questions are now literature-backed and explicitly scoped.
  - The repo has not yet resolved these into keep/reject/defer decisions against
    the benchmark-facing control.
  - The most important open architectural questions now have clearer reference
    context than before.
- Exit criteria:
  - Each ablation changes one core hypothesis at a time.
  - Each ablation is evaluated against the frozen or promoted control baseline.
  - The work ends with explicit keep/reject/defer architecture decisions rather
    than open-ended experimentation.

### TF-RD-013: Separate-Repo Inference Handoff and Conformance

- Status: `planned`
- Milestone: `Next`
- Goal alignment: `Goal 2: Core Prediction Modes`
- Pillar alignment: `runtime handoff`
- Goal: make the bundle contract consumable by a separate inference repo
  through a stable reference consumer and conformance suite, without turning
  this repo into the production runtime.
- Linear tracking:
  `no active issue yet; open BL inference chain after TF-RD-012`
- Repo touchpoints:
  - `src/tab_foundry/export/loader_ref.py`
  - `src/tab_foundry/export/contracts.py`
  - `tests/export/test_exporter.py`
  - `tests/export/fixtures/`
  - `docs/inference.md`
- Current state:
  - A v3 reference consumer now loads bundles, applies persisted preprocessing,
    constructs a model batch, and emits classification probabilities or
    regression quantiles.
  - Golden conformance fixtures now exist for one classification path and one
    regression path.
  - Downstream-repo stabilization, broader fixture reuse, and ownership
    boundaries beyond the reference path remain unfinished.
- Exit criteria:
  - The reference consumer runs end to end on v3 bundles only.
  - Downstream repos can reuse the golden fixtures and checksum validation
    surface.
  - Classification and regression both have stable expected outputs.
  - Runtime ownership remains separate-repo by docs and package boundaries.

### TF-RD-009: Core Prediction Mode Parity

- Status: `partial`
- Milestone: `Next`
- Goal alignment: `Goal 2: Core Prediction Modes`
- Pillar alignment: `mode coverage`
- Goal: broaden the common experiment stack from classification-first work into
  stable shared support for regression and missing-data prediction.
- Linear tracking: `BL-183 -> BL-185 -> BL-186 -> BL-187 -> BL-188 -> BL-189`
- Repo touchpoints:
  - `src/tab_foundry/training/evaluate.py`
  - `src/tab_foundry/training/losses.py`
  - `src/tab_foundry/data/dataset.py`
  - `src/tab_foundry/bench/checkpoint.py`
  - `src/tab_foundry/cli/commands/`
- Current state:
  - Classification and regression flows exist.
  - Classification remains the anchor workload.
  - Missing-data task support and prediction-mode-specific summaries are not
    yet complete.
- Exit criteria:
  - A shared mode interface exists across at least classification and
    regression.
  - Regression reaches parity on the common experiment stack.
  - Missing-data tasks and evaluation summaries are implemented.
  - Artifact reporting can break out metrics by prediction mode without
    coupling mode parity work to bundle-schema or runtime-handoff changes.

### TF-RD-010: Extended Prediction Modes and Modalities

- Status: `research`
- Milestone: `Later`
- Goal alignment: `Goal 3: Extended Tasks And Modalities`
- Pillar alignment: `modality readiness`
- Goal: scope tertiary modality work without letting it displace the primary
  scaling agenda.
- Linear tracking: `BL-184 -> BL-191 -> BL-190 -> BL-192 -> BL-193`
- Repo touchpoints:
  - `src/tab_foundry/model/components/many_class.py`
  - `docs/development/design-decisions.md`
  - `reference/papers.md`
  - `reference/evidence.md`
- Current state:
  - Many-class support exists and can remain in maintenance mode.
  - Time-series and text-conditioned inputs are already called out as later
    work.
  - No canonical path exists yet for temporal or text-conditioned extensions.
- Exit criteria:
  - One scoped time-series extension path exists.
  - One scoped text-embedding-conditioned path exists.
  - Readiness gates define when tertiary modalities may compete for priority.
  - Many-class maintenance remains bounded unless explicitly promoted.

### TF-RD-011: Conditional Prior/Data Alignment When Architecture Is Not the Bottleneck

- Status: `research`
- Milestone: `Later`
- Goal alignment: `Goal 1: Scaling Predictability`
- Pillar alignment: `scaling predictability`
- Goal: preserve a deliberate path for prior/data alignment work only if tuned
  architecture and recipe choices still fail to explain the gap to the external
  baseline.
- Linear tracking:
  `no active issue yet; gated follow-on after TF-RD-007 and TF-RD-008`
- Repo touchpoints:
  - `src/tab_foundry/data/sources/`
  - `src/tab_foundry/data/factory.py`
  - `src/tab_foundry/data/manifest.py`
  - `docs/development/design-decisions.md`
  - `reference/papers.md`
- Current state:
  - The roadmap already names prior/source modularity as a desired capability.
  - The manifest path is useful and canonical today.
  - There is no evidence yet that prior/data alignment should preempt the
    architecture and tuning agenda.
- Exit criteria:
  - The item opens only if tuned architecture still materially trails
    `nanoTabPFN`, or earlier evidence shows that prior/data mismatch is the
    dominant bottleneck.
  - If opened, the work covers benchmark-aligned synthetic configs,
    multi-pass corpus generation ideas, and direct prior/data comparison under a
    fixed architecture family and budget.

## Milestone Board

### Implemented

- `TF-RD-002` repo foundation and dagzoo-style organization
  - Completed March 12, 2026 via `BL-175 -> BL-177 -> BL-178 -> BL-179 -> BL-180`

### Now

- `TF-RD-001` canonical control baseline and experiment trust
- `TF-RD-012` shared preprocessing and export-contract fidelity
- `TF-RD-003` literature-first references and external baseline borrowing
- `TF-RD-004` neutral architecture registry and shared component surfaces
- `TF-RD-005` modular QASS and non-QASS backbone infrastructure
- `TF-RD-006` scaling-law measurement and Chinchilla-style planning
- `TF-RD-007` current-architecture tuning and control promotion

### Next

- `TF-RD-008` architecture ablations against the control baseline
- `TF-RD-013` separate-repo inference handoff and conformance
- `TF-RD-009` core prediction mode parity

### Later

- `TF-RD-010` extended prediction modes and modalities
- `TF-RD-011` conditional prior/data alignment when architecture is not the
  bottleneck

## Dependencies and Sequencing

- `TF-RD-001` must mature before deeper benchmark-facing interpretation work can
  be trusted.
- `TF-RD-002` should stay ahead of deeper architecture expansion so repo growth
  does not recreate historical package sprawl.
- `TF-RD-003` should inform architecture work before the repo commits to deeper
  refactors or new named baselines.
- `TF-RD-012` starts immediately after `TF-RD-001` and should land before
  `TF-RD-006` relies on the export-facing preprocessing surface.
- `TF-RD-004` and `TF-RD-005` define the platform on which later comparisons are
  supposed to run.
- `TF-RD-006` depends on having a sufficiently stable architecture and workflow
  surface to make scaling measurements comparable.
- `TF-RD-007` comes before `TF-RD-008`; the current architecture should be tuned
  before architectural ablations are used to explain benchmark gaps.
- `TF-RD-008` and the later architecture decision should complete before Goal 2
  becomes the main driver.
- `TF-RD-013` depends on `TF-RD-012` and should stabilize the separate-runtime
  consumer boundary before `TF-RD-009` expands prediction-mode parity claims.
- `TF-RD-009` begins only after Goal 1 has produced a stable scaling-oriented
  experiment stack.
- `TF-RD-010` remains explicitly tertiary and should not compete with Goal 1 or
  Goal 2 by default.
- `TF-RD-011` stays gated until architecture and tuning evidence strongly
  suggests that prior/data mismatch, not model/recipe choice, is the binding
  bottleneck.

## Guardrails

- The benchmark remains a constraint and evaluation surface, not the primary
  statement of project success.
- Confirmatory benchmark runs should stay in the same train-time budget class as
  the control path.
- Internal sweeps should prune candidates before broader benchmark-facing
  confirmation.
- Export and inference compatibility should remain intact until a separate
  migration effort is explicitly planned.
- Every active item should yield one reproducible config or code path, one run
  artifact set, and one short written keep/reject/defer conclusion.

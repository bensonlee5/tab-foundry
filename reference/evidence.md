# Literature Evidence Mapping

This document links the canonical roadmap in `docs/development/roadmap.md` to
primary references and to the most important repo-local evidence.

Related docs:

- Canonical roadmap: `docs/development/roadmap.md`
- Design decisions and repo structure: `docs/development/design-decisions.md`
- Architecture reference: `docs/development/model-architecture.md`
- Architecture deltas: `docs/development/architecture-deltas.md`
- Paper index: `reference/papers.md`

Conventions:

- Confidence: `high`, `medium`, or `low`
- Lane: roadmap milestone lane supported by this evidence (`Now`, `Next`,
  `Later`)

## Evidence-to-Roadmap Mapping

| Source | Key Claim Used | Roadmap IDs | Lane | Confidence |
|--------|---------------|-------------|------|------------|
| TabPFN v2 (Nature, 2024) | PFN-style cell-table attention remains the frozen control lineage and the main structural comparison point | `TF-RD-001`, `TF-RD-003`, `TF-RD-008` | Now | high |
| nanoTabPFN (repo) | Training recipe, model sizing, and benchmark comparison define the practical PFN control lane | `TF-RD-001`, `TF-RD-008` | Now | high |
| TabICLv2 (2602.11139) | The row-first architecture direction is the main external reference: grouped feature embedding, row embedding, row-level ICL, and optional QASS | `TF-RD-003`, `TF-RD-004`, `TF-RD-005`, `TF-RD-006`, `TF-RD-007`, `TF-RD-008` | Now | high |
| TabICL (2502.05564) | Staged migration and curriculum-style complexity changes should be introduced in a controlled ladder rather than as a single architecture jump | `TF-RD-002`, `TF-RD-003`, `TF-RD-008` | Now | high |
| Deep Sets (1703.06114) | Permutation-aware row and column structure should remain the default lens for new row-first work | `TF-RD-004`, `TF-RD-005`, `TF-RD-006`, `TF-RD-007` | Now | high |
| Set Transformer (1810.00825) | Inducing-point and set-style reasoning are the default reference for column-set work | `TF-RD-006`, `TF-RD-007` | Now | high |
| FT-Transformer (2106.11959) | Per-feature embedding is the baseline tokenization reference that grouped-token work must beat or justify departing from | `TF-RD-003`, `TF-RD-004` | Now | high |
| On Embeddings for Numerical Features in Tabular Deep Learning (2203.05556) | Value embedding choices are a high-leverage architectural lever and should be made on the shared surface, not hidden behind the nano encoder path | `TF-RD-003`, `TF-RD-004` | Now | high |
| A Closer Look at TabPFN v2 (2502.17361) | Architecture robustness claims should be tested structurally and not inferred only from single benchmark endpoints | `TF-RD-002`, `TF-RD-008`, `TF-RD-009` | Now | medium-high |
| SGDR (1608.03983) | Schedule work remains relevant, but should support the promoted architecture rather than substitute for it | `TF-RD-001`, `TF-RD-009` | Next | medium-high |
| muP (2203.03466) | Hyperparameter transfer across model sizes matters once one coherent row-first anchor exists | `TF-RD-009` | Next | high |
| Chinchilla (2203.15556) | Compute-optimal reasoning belongs on the promoted anchor, not on transient hybrid bridge surfaces | `TF-RD-009` | Next | high |
| Kaplan et al. (2001.08361) | Scaling trends should be measured on one coherent family rather than across mixed architectural lines | `TF-RD-009` | Next | high |
| Power Lines (2505.13738) | Small-scale scaling fits need explicit artifact paths and careful interpretation | `TF-RD-009` | Next | medium-high |
| Broken Neural Scaling Laws (2210.14891) | Knees and plateaus must be treated as expected outcomes when the architecture target is still moving | `TF-RD-009` | Next | medium |
| SAINT (2106.01342) | Row/column interaction changes should be benchmarked as explicit structural choices rather than imported as defaults | `TF-RD-005`, `TF-RD-006` | Next | medium |
| Perceiver (2103.03206) | Latent bottlenecks remain a later alternative if row/column token count becomes the limiting factor | `TF-RD-006`, `TF-RD-012` | Later | medium |
| EquiTabPFN (2502.06684) | Label-conditioning choices should stay modular so target handling can evolve after the row-first base stabilizes | `TF-RD-010`, `TF-RD-012` | Later | medium |
| TabDPT (2410.18164) | Prior/source changes are a later scaling lever and should not displace the core architecture migration prematurely | `TF-RD-011`, `TF-RD-012` | Later | medium |
| Sentence-BERT (D19-1410) | Text-conditioned columns should remain an external-embedding later lane rather than a distraction from the classification-first backbone plan | `TF-RD-012` | Later | high |
| nanochat (repo) | Compact-transformer training and residual-layout ideas are valid donors, but only when they preserve the tabular set-structured goal | `TF-RD-002`, `TF-RD-007`, `TF-RD-009` | Now | high |

## Per-Roadmap Evidence Notes

### TF-RD-001: Control Freeze And Experiment Trust

- External evidence:
  - TabPFN v2 and nanoTabPFN define the PFN-style control lineage that should
    stay frozen for benchmark trust
- Repo-local evidence:
  - `tabfoundry_simple` and `stage=nano_exact` already serve as the cleanest PFN
    controls
  - the current large-anchor `nano_exact + prenorm + row_cls` line is useful
    diagnostic evidence, but it is structurally hybrid and should not be treated
    as the promoted target by default
- Success signal:
  - control-lane claims and target-lane claims are clearly separated in research
    docs and result interpretation

### TF-RD-002: Measurement Surfaces For Architecture Migration

- External evidence:
  - TabICL and the TabPFN analysis literature both imply that architecture
    interpretation needs more than final benchmark metrics
  - nanochat is a good recipe reference for clean instrumentation and readable
    training diagnostics
- Repo-local evidence:
  - the exact-prior path already emits module gradients, activation traces, and
    additive telemetry summaries
  - the regular architecture-screen trainer still lacks that parity, so the
    most decision-critical gap is coverage on `cls_benchmark_staged`
- Success signal:
  - row-first rows can be judged on stage-local stability and quality directly
    from the regular architecture-screen lane

### TF-RD-003: Shared-Surface Unlock

- External evidence:
  - TabICLv2 assumes the architecture target is already operating on a
    non-PFN-only embedding surface
  - FT-Transformer and numerical-embedding references support moving feature
    handling out of the frozen nano path before testing richer structure
- Repo-local evidence:
  - tokenizer overrides are ineffective under the nano encoder path
  - the staged recipe ladder already treats `shared_norm` and `prenorm_block` as
    the bridge out of the PFN control lane
- Success signal:
  - TabICL-inspired work happens on a surface where tokenization and later
    modules are actually active

### TF-RD-004: Tokenization Migration

- External evidence:
  - TabICLv2's grouped embedding idea is the main directional reference
  - FT-Transformer is the baseline that grouped tokens must justify departing
    from
  - Deep Sets reinforces that tokenization should preserve a set-structured view
    of rows and columns
- Repo-local evidence:
  - grouped tokens already exist as a staged recipe
  - earlier tokenizer experiments on the nano path were not attributable
- Success signal:
  - grouped tokens are validated as part of the row-first ladder rather than as
    a disconnected ablation

### TF-RD-005: Row-Embedding Unlock

- External evidence:
  - TabICLv2's decisive architectural move is to form row embeddings before the
    final ICL stage
  - SAINT is a useful benchmark-first comparison point for row interaction once
    tokenization is coherent
- Repo-local evidence:
  - `row_cls_pool` exists as a staged step in the intended ladder
  - compact-surface row-CLS evidence was strongly negative, but it was gathered
    on PFN-adjacent surfaces and should be scoped that way
- Success signal:
  - the repo reaches clear separate yes/no answers on useful row embeddings and
    plain row-level context in the intended migration line

### TF-RD-006: Column-Set Integration

- External evidence:
  - Set Transformer is the default reference for column-set reasoning
  - TabICLv2 and related row-first designs make column embedding a first-class
    stage rather than a late add-on
- Repo-local evidence:
  - `column_set` exists as a staged step
  - old TFCol evidence on compact surfaces was stable but near-neutral and
    expensive
- Success signal:
  - TFCol is either promoted with explicit value evidence or deferred with clear
    cost-based reasoning

### TF-RD-007: Row-Level Context And QASS Attribution

- External evidence:
  - TabICLv2 provides the main QASS and row-level ICL reference
  - Deep Sets and Set Transformer imply that QASS should be justified as a
    structural benefit, not presumed mandatory
  - nanochat supports the repo stance that simpler non-QASS alternatives should
    stay easy to compare
- Repo-local evidence:
  - QASS primitives already exist
  - compact-surface QASS rows trained cleanly but did not prove enough value to
    promote the mechanism
- Success signal:
  - the repo can separately answer whether row-level context helps and whether
    QASS helps beyond plain row-level context

### TF-RD-008: Coherent Classification Anchor Promotion

- External evidence:
  - TabICLv2 argues for a coherent row-first architecture, not a permanent
    hybrid of PFN control pieces and row-first readout patches
  - TabPFN analysis work reinforces that robustness claims should be grounded in
    coherent model surfaces
- Repo-local evidence:
  - the staged family contained two final row-first promotion candidates on the
    missing-permitting large bundle: `row_cls + qass + no tfcol` and
    `row_cls + qass + tfcol_heads4`
  - `qass_tfcol_large_missing_validation_v1` closed on a mixed result: the TFCol
    row improved final Brier and ROC AUC, but its final log loss was slightly
    worse than the simpler no-TFCol control
  - TF-RD-008 therefore settled on `row_cls + qass + no tfcol` as the default
    row-first anchor, with `row_cls + qass + tfcol_heads4` retained as a
    calibration-oriented alternative
- Success signal:
  - one named default row-first classification anchor now exists and can serve
    as the target for future architecture and scaling work without erasing the
    retained calibration-oriented alternative

### TF-RD-009: Scaling-Law Measurement On The Promoted Anchor

- External evidence:
  - Chinchilla, Kaplan, Power Lines, and Broken Neural Scaling Laws all require
    a stable family and comparable artifacts
  - muP is only meaningfully useful once one anchor family exists
- Repo-local evidence:
  - tuning and comparison tooling are already present
  - scaling-law artifacts are not yet canonical on the right architecture
- Success signal:
  - scaling curves are fit on the promoted row-first anchor, not on transient
    bridge rows

### TF-RD-010: Many-Class Promotion On The Row-First Base

- External evidence:
  - label-conditioning work such as EquiTabPFN matters more after the backbone
    is coherent
- Repo-local evidence:
  - the staged family already includes `many_class`
  - many-class should inherit the promoted row-first backbone rather than become
    its own architecture lane
- Success signal:
  - many-class is evaluated as an extension of the promoted anchor

### TF-RD-011: Repo-Wide Enablers And Contract Fidelity

- External evidence:
  - TabDPT and related scaling work justify keeping source/prior flexibility in
    view, but not ahead of the architecture target
- Repo-local evidence:
  - manifest-backed training, shared preprocessing work, and v3 export
    foundations already exist
  - corpus provenance and end-to-end contract fidelity still need work
- Success signal:
  - the architecture program can rely on data and contract surfaces that are
    trustworthy enough for promotion decisions

### TF-RD-012: Regression, Inference Handoff, And Later Modalities

- External evidence:
  - text-conditioning references and later modality papers belong to a later
    lane once the classification anchor is stable
- Repo-local evidence:
  - regression is intentionally removed today
  - runtime handoff remains partial and should follow the promoted staged base
- Success signal:
  - later prediction modes and downstream handoff build on the promoted
    row-first base instead of competing with the main architecture program

## Evidence Limits And Assumptions

- This is a planning artifact, not a reproduction benchmark.
- The roadmap is TabICLv2-inspired, not a literal TabICLv2 parity plan.
- Repo-local negative evidence for row-CLS, TFCol, and QASS should be scoped to
  the surfaces on which it was gathered, especially the compact `nano_exact`
  line.
- PFN-style controls remain necessary even if the long-term target becomes more
  row-first.
- Sequence-order-centric mechanisms remain low priority unless a tabular
  justification appears.

# TF-RD-021A: Repeated-Input Sandwich Candidate And NanoTabPFN Screen

This is the long-form evidence note for the repeated-input
`tabfoundry_sandwich` architecture candidate.
The lane now lives under the broader
[TF-RD-016](../../docs/development/roadmap.md#tf-rd-016-architecture-surface-adequacy-and-selective-expansion)
architecture-adequacy workstream rather than as a runtime-sidecar under
TF-RD-022.

- Status: `partial`
- Milestone: `Next`
- Dependency position: architecture sub-lane under TF-RD-016; TF-RD-022 remains
  a dependency surface for later runtime-aware hard-surface reads, but it does
  not own sandwich planning

## External Evidence

- Relevant references already called out in `reference/papers.md`:
  - SAINT for explicit row or column alternation
  - Perceiver for latent-bottleneck attention
  - Set Transformer or ISAB for induced-memory bottlenecks
  - TabPFN or TabICL family references for PFN-style table encoding and
    row-first ICL framing
- Dedicated long-form external synthesis is still missing.
- Sources to curate next:
  - precise row or column alternating tabular transformer precedents
  - latent-bottleneck scaling references that quantify memory or throughput
    behavior when the latent count is fixed

## Repo-Local Evidence

- implementation issue [#174](https://github.com/bensonlee5/tab-foundry/issues/174)
  now records the fixed-latent implementation landing
- umbrella issue [#178](https://github.com/bensonlee5/tab-foundry/issues/178)
  now owns long-running sandwich stabilization and future child tickets
- immediate nanoTabPFN screen issue
  [#179](https://github.com/bensonlee5/tab-foundry/issues/179) now owns only:
  - sweep 1 analogue: latent-count screen with width and depth fixed
  - sweep 2 analogue: width plus latent-count follow-up around sweep-1 winners
- draft sweep package
  `reference/system_delta_sweeps/tf_rd_021a_sandwich_nanotabpfn_screen_v1/`
  now exists with:
  - row `01` sandwich replay on the locked batch64-sqrt prior surface
  - rows `02` and `03` as the ready latent-only screen
  - rows `04` and `05` as blocked width follow-up rows that stay dormant until
    the latent screen lands
- `tabfoundry_sandwich` now exists as a separate experimental `model.arch`
  family with:
  - one fixed learned latent bank
  - an `R + C` repeated row/column summary stream
  - train-label or test-query conditioning fused into the row-summary tokens
  - repeated Perceiver-style cross-attention reads plus latent self-attention
    stages
  - shared Fourier row or column positions plus schema-aware feature-type
    encoding
  - test-row readout from the final latent state
  - `forward_batched()` support for prior-dump training
  - shared inspection, export, and training-surface wiring
- local CPU forward-check and local MPS forward-check both succeed on the new
  arch in the implementation branch
- tiny smoke coverage now exists for:
  - ordinary training with the real sandwich model
  - exact prior-dump training with the real sandwich model

## Current Interpretation

- the first useful question is still not whether sandwich wins on the hardest
  surfaces; it is whether the repeated-input latent-bottleneck design trains at all and
  scales predictably on the fastest feedback surface
- that is why the immediate sequence should start on nanoTabPFN prior-dump data
  rather than dagzoo:
  - prior-dump sweeps are smaller and faster
  - more historical sweep context already exists on that lane
  - they are the best place to settle the first latent-count and width reads
- width should not be mixed into the first executed pass; keep rows `01` through
  `03` as the latent-only screen, then unblock the width rows only if that first
  read justifies it
- `tabfoundry_sandwich` should now be treated as the primary long-term
  architecture candidate, while `tabfoundry_staged` remains the incumbent
  benchmark/reference line until sandwich earns promotion
- later dagzoo confirmation, runtime-aware hard-surface reads, and any eventual
  promotion decision should be broken into additional child issues under
  [#178](https://github.com/bensonlee5/tab-foundry/issues/178), not bundled
  into the immediate nanoTabPFN sweep issue
- MPS OOMs remain informative for local iteration but should not be counted as
  the quantitative hard-surface evidence lane

## Open Evidence Gaps

- no actual nanoTabPFN sandwich sweep results are recorded yet
- the checked-in sweep package is still draft-only; rows `04` and `05` are
  intentionally blocked until the latent screen resolves
- no longer-budget stability read exists yet for the new architecture
- no harder-surface confirmation read exists yet for the new architecture
- the repo still lacks the peak CUDA-memory telemetry from
  [#58](https://github.com/bensonlee5/tab-foundry/issues/58), so future hard
  surface reads still need the TF-RD-022 telemetry work to rank memory wins
  cleanly
- the quality or fit floor that would justify promoting or deferring this
  architecture candidate is not yet explicit

## Exit Signals

- sweep 1 identifies viable latent-count rows on nanoTabPFN prior-dump data
- sweep 2 identifies whether width plus latent-count changes materially improve
  the first viable rows
- the next sandwich follow-on issue can be scoped from those winners without
  carrying the old monolithic phased ladder forward
- later hard-surface reads can make one explicit keep/defer decision on whether
  sandwich should displace the staged reference line

# TF-RD-021A: Repeated-Input Sandwich Candidate And NanoTabPFN Screen

This is the long-form evidence note for the original repeated-input
`tabfoundry_sandwich` nanoTabPFN screen.
The lane lives under the broader
[TF-RD-016](../../docs/development/roadmap.md#tf-rd-016-architecture-surface-adequacy-and-selective-expansion)
architecture-adequacy workstream rather than as a runtime-sidecar under
TF-RD-022.

- Status: `completed`
- Milestone: `Completed`
- Dependency position: closed architecture sub-lane under TF-RD-016; successor
  replay work now moves to TF-RD-021B under umbrella issue `#178`

## External Evidence

- Relevant references already called out in `reference/papers.md`:
  - SAINT for explicit row or column alternation
  - Perceiver for latent-bottleneck attention
  - Set Transformer or ISAB for induced-memory bottlenecks
  - TabPFN or TabICL family references for PFN-style table encoding and
    row-first ICL framing
- Dedicated long-form external synthesis is still missing.
- Sources to curate next:
  - PerceiverIO-style readout references that separate latent reasoning from
    output-query fidelity
  - tabular or set-style references that preserve finer-grained feature tokens
    later into the model before pooling

## Repo-Local Evidence

- implementation issue [#174](https://github.com/bensonlee5/tab-foundry/issues/174)
  records the original fixed-latent implementation landing
- umbrella issue [#178](https://github.com/bensonlee5/tab-foundry/issues/178)
  still owns long-running sandwich stabilization and future child tickets
- immediate nanoTabPFN screen issue
  [#179](https://github.com/bensonlee5/tab-foundry/issues/179) now closes on
  row `01` as explicit negative evidence for the summary-bottleneck replay
- draft sweep package
  `reference/system_delta_sweeps/tf_rd_021a_sandwich_openml_screen_v1/`
  now closes with:
  - row `01` completed on the locked batch64-sqrt prior surface
  - rows `02` and `03` deferred to separate successor work instead of executed
  - rows `04` and `05` deferred as backlog width evidence, not active next
    steps
- completed row `01` recorded:
  - stable training on CPU with clipped-step fraction `0.0148`
  - final ROC AUC `0.6224`
  - locked staged anchor final ROC AUC `0.7634`
  - nanoTabPFN final ROC AUC `0.7515`

## Current Interpretation

- the first useful question is now answered: the repeated-input
  summary-bottleneck sandwich trains, but it is materially underpowered on the
  fast locked nanoTabPFN prior surface
- the result is negative evidence for the earlier architecture boundary, not
  for the broader sandwich family
- more latent-count or width tuning on the same topology is no longer the best
  next use of budget
- successor work should reopen the latent write path and final readout path
  before any fresh capacity sweep is authored

## Open Evidence Gaps

- no harder-surface confirmation read exists yet for the sandwich family
- no longer-budget stability read exists yet for the successor topology
- the repo still lacks the first locked-prior replay for the successor
  hybrid full-cell architecture tracked in TF-RD-021B
- the quality or fit floor that would justify promoting or deferring the whole
  sandwich family remains implicit until the successor replay lands

## Exit Signals

- TF-RD-021A is closed with row `01` as stable negative evidence for the
  summary-bottleneck replay
- the latent and width ladder is preserved as deferred backlog evidence rather
  than active queue work
- successor work is handed off to TF-RD-021B instead of extending the old
  monolithic ladder

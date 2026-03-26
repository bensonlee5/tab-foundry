# TF-RD-021A: Latent-Bank Sandwich Prototype And Phased Screen

This is the bounded sidecar evidence note for the experimental
`tabfoundry_sandwich` lane that now runs alongside
[TF-RD-021](../../docs/development/roadmap.md#tf-rd-021-training-runtime-and-vram-efficiency-on-the-promoted-anchor).

- Status: `partial`
- Milestone: `Next`
- Dependency position: bounded execution sidecar under TF-RD-021; it should
  not displace the canonical staged target, and if it survives the phased
  screen it should hand evidence back into later TF-RD-016 architecture
  decisions and the harder-surface CUDA reads

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
  now defines the fixed v0 architecture scope
- phased sweep issue [#175](https://github.com/bensonlee5/tab-foundry/issues/175)
  now defines the required data ladder:
  - local CPU or MPS smoke
  - nanoTabPFN prior-dump latent-size screen
  - nanoTabPFN prior-dump width plus latent-size screen
  - smaller dagzoo confirmation
  - hard dagzoo decision read on the TF-RD-020 anchor
- `tabfoundry_sandwich` now exists as a separate experimental `model.arch`
  family with:
  - learned row and column latent banks
  - pre-norm residual cross-attention blocks
  - train-write or test-read semantics
  - `forward_batched()` support for prior-dump training
  - shared inspection, export, and training-surface wiring
- local CPU forward-check and local MPS forward-check both succeed on the new
  arch in the implementation branch
- tiny smoke coverage now exists for:
  - ordinary training with the real sandwich model
  - exact prior-dump training with the real sandwich model

## Current Interpretation

- the first useful question is not whether the prototype wins on the hardest
  dagzoo fronts; it is whether the simpler latent-bank design trains at all and
  scales predictably on the fastest feedback surface
- that is why the sequence should start on nanoTabPFN prior-dump data rather
  than dagzoo:
  - prior-dump sweeps are smaller and faster
  - more historical sweep context already exists on that lane
  - they are the best place to settle latent-bank sizes and the first width read
- width should not be mixed into the first latent-size sweep; keep sweep 1 as a
  latent-only screen, then use sweep 2 to jointly vary width and latent sizes
- dagzoo remains necessary for the real decision because the actual fit and OOM
  pressure that motivated this work live on the harder TF-RD-020 surfaces, not
  on the prior dump
- MPS OOMs are informative for local iteration but should not be counted as the
  quantitative hard-surface evidence lane

## Open Evidence Gaps

- no actual nanoTabPFN sandwich sweep results are recorded yet
- no dagzoo confirmation read exists yet for the new architecture
- the repo still lacks the peak CUDA-memory telemetry from
  [#58](https://github.com/bensonlee5/tab-foundry/issues/58), so future hard
  dagzoo reads still need the TF-RD-021 telemetry work to rank memory wins
  cleanly
- the quality or fit floor that would justify keeping or stopping this sidecar
  is not yet explicit

## Exit Signals

- sweep 1 identifies viable latent-bank rows on nanoTabPFN prior-dump data
- sweep 2 identifies whether width plus latent-size changes materially improve
  the first viable rows
- dagzoo confirmation shows the prototype transfers off the prior dump without
  obvious collapse
- the hard dagzoo read can make one explicit keep or stop decision for the
  experimental sidecar

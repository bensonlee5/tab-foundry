# TF-RD-012: Inference Handoff And Later Modalities

This is the canonical long-form evidence note for
[TF-RD-012](../../docs/development/roadmap.md#tf-rd-012-inference-handoff-and-later-modalities).

- Status: `research`
- Milestone: `Later`
- Dependency position: follows
  [TF-RD-016](tf_rd_016_architecture_surface_adequacy.md) and the earlier
  post-008 training-surface gates, and stays behind the classification-first
  roadmap

## External Evidence

- Shared bibliography: [reference/papers.md](../papers.md)
- Current curated later-modality context already exists in the shared paper
  index through `Sentence-BERT`, `TaBERT`, `TURL`, and `TabDPT`
- the external literature is present, but it is explicitly later-lane evidence
  rather than a reason to move these modalities onto the current critical path
- External evidence to curate next: runtime-handoff case studies and any later
  modality references that become decision-relevant after the classification
  base is stable

## Repo-Local Evidence

- classification remains the only active supported prediction mode
- runtime handoff and later modalities remain deferred
- the roadmap requires TF-RD-013, TF-RD-018, at least one harder post-008
  ladder, and TF-RD-016 before runtime feedback should become a clean
  architecture constraint

## Current Interpretation

- advance separate-runtime handoff only after classification and export
  contracts settle
- keep time series, text-conditioned inputs, and other later modalities off the
  critical path while the promoted classification base is still stabilizing
- treat runtime feedback as a later constraint on a settled classification base,
  not as an early driver of architecture shape

## Open Evidence Gaps

- the repo does not yet have a canonical separate-runtime handoff contract
- there is no modality-specific benchmark or adequacy ladder for later inputs
- the later-modality bibliography is present, but no modality has a bounded
  execution plan yet

## Exit Signals

- inference handoff and later modalities build on the promoted staged base
  rather than running ahead of it
- runtime or modality expansion decisions are made only after the classification
  base is stable enough to interpret cost tradeoffs cleanly

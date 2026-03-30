# TF-RD-016: Architecture Surface Adequacy, Sandwich Simplification, And Selective Expansion

This is the canonical long-form evidence note for
[TF-RD-016](../../docs/development/roadmap.md#tf-rd-016-architecture-surface-adequacy-sandwich-simplification-and-selective-expansion).

- Status: `completed`
- Milestone: `Completed`
- Dependency position: historical closeout gate that now hands active work to
  [TF-RD-010](tf_rd_010_many_class_promotion.md), followed by
  [TF-RD-021](tf_rd_021_steering_derived_dagzoo_corpus_fronts.md),
  [TF-RD-022](tf_rd_022_training_runtime_vram_efficiency.md),
  [TF-RD-017](tf_rd_017_class_imbalance_robustness.md), and
  [TF-RD-009](tf_rd_009_scaling_law_measurement.md)

## External Evidence

- Shared bibliography: [reference/papers.md](../papers.md)
- Current curated context is broad rather than knob-specific: `TabICLv2`,
  `FT-Transformer`, `Deep Sets`, `Set Transformer`, `PerceiverIO`, and compact
  transformer recipe references from `nanochat`
- Dedicated micro-architecture adequacy literature remains secondary because the
  repo now has enough local simplification evidence to move on to benchmark
  definition

## Repo-Local Evidence

- `tabfoundry_sandwich` remains the primary classification architecture family
- TF-RD-021A and TF-RD-021B closed the immediate sandwich replay and bounded
  simplification passes
- issue [#184](https://github.com/bensonlee5/tab-foundry/issues/184) kept the
  compact hybrid control, but TF-RD-016 does not interpret that as a reason to
  freeze the full benchmark contract around the historical four-token surface
- issue [#178](https://github.com/bensonlee5/tab-foundry/issues/178) now closes
  on the decision that the next useful evidence is benchmark definition plus a
  bounded head/output evolution
- the follow-on benchmark config now fixes:
  - `feature_type_conditioning=film`
  - `sandwich_summary_tokens_per_axis=3`
  - `many_class_base=10`
  - direct multiclass head
- the follow-on benchmark program is explicitly repo-linked:
  - `dagzoo` owns synthetic training corpora
  - `tab-realdata-hub` owns real-data validation bundles and manifest
    materialization
  - `tab-foundry` owns the evolved sandwich model and sweep contracts

## Current Interpretation

- The sandwich backbone is coherent enough that more simplification-only replay
  is no longer the highest-value next step
- The next decision surface should be a benchmark program, not another local
  ablation ladder
- The selected architecture evolution is intentionally modest:
  - keep the backbone, tokenizer family, and Perceiver stack intact
  - replace the “small-class only” framing with a direct multiclass head
  - use FiLM and `3` summary tokens per axis as the new default benchmark
    contract
- Class imbalance remains important, but TF-RD-016 treats it as a benchmark
  coverage/reporting requirement for TF-RD-010 rather than a blocker that must
  open its own ladder before closeout

## Open Evidence Gaps

- TF-RD-016 itself is closed, but the benchmark program it hands off still
  depends on:
  - `tab-realdata-hub` medium and large classification bundle ownership
  - frozen legacy TF-RD-010 control baselines
  - first executed medium and large benchmark runs

## Exit Signals

- satisfied: the repo records one explicit decision that the sandwich family is
  ready for benchmark-defined multiclass evaluation
- satisfied: the active follow-on lane is now TF-RD-010 rather than another
  simplification-first pass
- satisfied: the `dagzoo -> tab-realdata-hub -> tab-foundry` linkage is the
  authoritative handoff model for the next classification phase

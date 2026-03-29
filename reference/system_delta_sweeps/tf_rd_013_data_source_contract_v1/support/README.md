# TF-RD-013 Support Bundle

Use this support bundle when you need the committed assumptions and comparison
surfaces behind TF-RD-013 issues `#120` and `#122`.

It is a historical reference-only support surface for that contract, not the
main roadmap or sweep execution guide. Reconstructing these local artifacts is
no longer a supported workflow.

Environment assumptions:

- `TAB_FOUNDRY_ROOT` is this repo root.
- `DAGZOO_ROOT` defaults to the sibling checkout `../dagzoo`.
- The dagzoo config ref is the sibling repo's default config, `../dagzoo/configs/default.yaml`.
- The curated comparator baseline is pinned to `src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json`.

Historical support surface:

- materializes one large unfiltered dagzoo `generate` output under `outputs/staged_ladder_support/tf_rd_013/generated_source/`
- keeps the initial promoted-anchor support surface pinned to dagzoo's `../dagzoo/configs/default.yaml` with a single `--num-datasets 8192` CPU generate call
- exports one OpenML-only curated comparator corpus under `outputs/staged_ladder_support/tf_rd_013/curated_realdata/openml_baseline/`
- reuses the pinned benchmark bundle task list and writes one packed shard per task with a deterministic `80/20` holdout split
- builds a local manifest for that generated source
- builds a local manifest for the curated OpenML comparator surface
- writes tracked JSON summaries for both runnable TF-RD-013 comparison surfaces

Committed files:

- `materialization_summary.json`: generated-source and curated-baseline artifact details, handoff metadata, sanitized lineage, and issue links.
- `manifest_characteristics_summary.json`: the available `anchor vs generated-source`, `anchor vs curated-baseline`, and `generated-source vs curated-baseline` comparisons for the initial TF-RD-013 sweep.

Local-only files:

- Generated dagzoo shards, curated OpenML packed shards, and the local manifests
  lived under `outputs/staged_ladder_support/tf_rd_013/`.
- `outputs/` stays ignored because the dagzoo artifacts are too large to commit as repo-tracked fixtures.

Curated comparator policy:

- Issue `#122` keeps the first curated comparator lane OpenML-only.
- Approved external manifest-backed augmentations remain optional and must come from the review ledger before they enter this support bundle.
- The curated comparator is evidence-only for the first promoted-anchor TF-RD-013 read; it is not a replacement handoff surface by default.

Dagzoo follow-up:

- The initial TF-RD-013 sweep intentionally starts with the unfiltered generated-source corpus.
- This support bundle still represents a single default-config generate invocation, not the longer-term multi-invocation shape program.
- Issue `#122` completed a neutral first promoted-anchor read: the anchor, the single-invocation dagzoo surface, and the OpenML-only comparator all landed on the same recorded large-bundle metrics.
- Issue `#127` now tracks the immediate TF-RD-013 follow-up on multi-invocation, shape-aware dagzoo coverage.
- Issue `#124` remains the later filtering-policy question rather than the immediate blocker from that first read.

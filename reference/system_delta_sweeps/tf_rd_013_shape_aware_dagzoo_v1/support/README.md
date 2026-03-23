# TF-RD-013 Shape-Aware Support Bundle

Use this support bundle when you need the committed assumptions, regeneration
flow, and comparison surfaces behind TF-RD-013 issue `#127`.

It is a reference-only support surface for that shape-aware follow-up, not the
main roadmap or sweep execution guide.

The canonical local regeneration flow is:

```bash
.venv/bin/python scripts/materialize_tf_rd_013_support.py --variant shape-aware --force
```

Environment assumptions:

- `TAB_FOUNDRY_ROOT` is this repo root.
- `DAGZOO_ROOT` defaults to the sibling checkout `../dagzoo`.
- The broader dagzoo follow-up uses three config-backed invocations:
  - `../dagzoo/configs/benchmark_cpu.yaml` with `2048` datasets
  - `../dagzoo/configs/default.yaml` with `4096` datasets
  - `../dagzoo/configs/benchmark_cuda_h100_large_shape.yaml` with `128` datasets
- The curated comparator baseline remains pinned to `src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json`.

Current support surface:

- materializes three explicit dagzoo generate runs under `outputs/staged_ladder_support/tf_rd_013_shape_aware_dagzoo_v1/`
- keeps each invocation's handoff manifest and identity separate for provenance review
- assembles one merged dagzoo manifest with `build_manifest(data_roots=[...])` and no single-handoff manifest metadata
- exports one OpenML-only curated comparator corpus under the same local support root
- writes tracked JSON summaries for the shape-aware dagzoo surface and the curated comparator surface

Committed files:

- `materialization_summary.json`: shape-program notes, per-invocation handoff summaries, merged-manifest assembly details, and curated comparator provenance.
- `manifest_characteristics_summary.json`: `anchor vs shape-aware dagzoo`, `anchor vs curated`, and `shape-aware dagzoo vs curated` manifest comparisons.

Local-only files:

- Generated dagzoo shards, curated OpenML packed shards, and local manifests live under `outputs/staged_ladder_support/tf_rd_013_shape_aware_dagzoo_v1/`.
- `outputs/` stays ignored because the dagzoo artifacts are too large to commit as repo-tracked fixtures.

Policy notes:

- This bundle is the issue `#127` follow-up to the neutral first promoted-anchor TF-RD-013 read from issue `#122`.
- The broader dagzoo read is still about synthetic-data coverage, not filtering policy; issue `#124` remains a later question only if the shape-aware result exposes a specific predictability problem.
- The curated comparator remains OpenML-first and evidence-only unless later approved augmentations are explicitly justified.

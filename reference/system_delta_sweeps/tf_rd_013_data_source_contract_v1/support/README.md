# TF-RD-013 Support Bundle

This directory is the committed reference-only support bundle for issue `#120`.

The canonical local regeneration flow is:

```bash
.venv/bin/python scripts/materialize_tf_rd_013_support.py --force
```

Environment assumptions:

- `TAB_FOUNDRY_ROOT` is this repo root.
- `DAGZOO_ROOT` defaults to the sibling checkout `../dagzoo`.
- The dagzoo config ref is the sibling repo's default config, `../dagzoo/configs/default.yaml`.

What the script does today:

- materializes one large unfiltered dagzoo `generate` output under `outputs/staged_ladder_support/tf_rd_013/generated_source/`
- keeps the initial promoted-anchor support surface pinned to dagzoo's `../dagzoo/configs/default.yaml` with a single `--num-datasets 8192` CPU generate call
- builds a local manifest for that generated source
- writes tracked JSON summaries for the initial unfiltered TF-RD-013 comparison surface

Committed files:

- `materialization_summary.json`: generated-source artifact details, handoff metadata, sanitized lineage, and issue links.
- `manifest_characteristics_summary.json`: the available `anchor vs generated-source` comparison for the initial unfiltered sweep.

Local-only files:

- Generated dagzoo shards and the local generated-source manifest live under `outputs/staged_ladder_support/tf_rd_013/`.
- `outputs/` stays ignored because the dagzoo artifacts are too large to commit as repo-tracked fixtures.

Filtering follow-up:

- The initial TF-RD-013 sweep intentionally starts with the unfiltered generated-source corpus.
- This support bundle still represents a single default-config generate invocation, not the longer-term multi-invocation shape program.
- Future TF-RD-013 dagzoo work should move toward an explicit multi-invocation, shape-aware support contract once the first promoted-anchor read is complete.
- Issue `#124` tracks the later decision about whether any filtered dagzoo variants should be introduced after that first read.

# TF-RD-013 Dagzoo Size Ladder Support Bundle

This directory is the committed reference-only support bundle for issue `#132`
under reopened epic `#96`.

The canonical local regeneration flow is:

```bash
.venv/bin/tab-foundry data dagzoo generate-manifest \
  --dagzoo-root ../dagzoo \
  --dagzoo-config configs/default.yaml \
  --handoff-root outputs/current_corpus/default_generated_source \
  --out-manifest data/manifests/default.parquet \
  --num-datasets 8192 \
  --seed 1 \
  --device cpu \
  --hardware-policy none

.venv/bin/python scripts/materialize_tf_rd_013_support.py --variant size-ladder --force
```

Environment assumptions:

- `TAB_FOUNDRY_ROOT` is this repo root.
- `DAGZOO_ROOT` defaults to the sibling checkout `../dagzoo`.
- `data/manifests/default.parquet` is a local/generated artifact for this flow, not a portable repo fixture.
- The fresh current-corpus bootstrap above is required on new machines before support materialization; stale absolute-path local snapshots are unsupported.
- Dagzoo generation stays fixed at `--device cpu --hardware-policy none` for every rung so the ladder isolates corpus content rather than generator hardware.
- The ladder reuses the same three config-backed regimes as the earlier shape-aware follow-up:
  - `../dagzoo/configs/benchmark_cpu.yaml`
  - `../dagzoo/configs/default.yaml`
  - `../dagzoo/configs/benchmark_cuda_h100_large_shape.yaml`

What the script does for this variant:

- treats `data/manifests/default.parquet` as the fresh current-corpus control for row 1 rather than regenerating it internally
- materializes three explicit dagzoo ladders under `outputs/staged_ladder_support/tf_rd_013_dagzoo_size_ladder_v1/`
- keeps each invocation handoff and identity record separate for provenance review
- assembles one merged manifest per rung with `build_manifest(data_roots=[...])`
- writes tracked JSON summaries for the three runnable dagzoo surfaces plus the anchor manifest comparison

Ladder definitions:

- `small`: `benchmark_cpu=128`, `default_medium=256`, `large_shape=8`
- `medium`: `benchmark_cpu=384`, `default_medium=768`, `large_shape=24`
- `large`: `benchmark_cpu=768`, `default_medium=1536`, `large_shape=48`

Tracked summary outputs after materialization:

- one materialization summary JSON with per-rung artifact details, per-invocation handoff summaries, merged-manifest assembly details, and issue links
- one manifest-characteristics summary JSON with `anchor vs small`, `anchor vs medium`, `anchor vs large`, and rung-to-rung manifest comparisons

Remote execution note:

- These JSON summaries are intentionally refreshed by the remote materialization run rather than committed from local ad hoc generation.

Local-only files:

- Generated dagzoo shards and local manifests live under `outputs/staged_ladder_support/tf_rd_013_dagzoo_size_ladder_v1/`.
- `outputs/` stays ignored because the generated artifacts are too large to commit as fixtures.

Policy notes:

- This bundle is binary-only and intentionally omits the curated real-data comparator and multiclass augmentation work.
- The ladder exists to establish a canonical fresh current-corpus control and then test whether shrunken shape-aware dagzoo corpora change the TF-RD-013 keep/defer read before TF-RD-018.
- Issue `#124` remains later filtering-policy work only if the size ladder leaves dagzoo plausible but exposes a narrower predictability or quality-policy question.

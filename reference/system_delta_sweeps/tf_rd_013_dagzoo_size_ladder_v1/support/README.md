# TF-RD-013 Dagzoo Size Ladder Support Bundle

This directory is the committed reference-only support bundle for issue `#132`
under reopened epic `#96`.

The canonical local corpus regeneration flow is:

```bash
.venv/bin/tab-foundry data corpus materialize \
  --recipe tf_rd_013_current_corpus_default_v1 \
  --dagzoo-root ../dagzoo \
  --force

.venv/bin/tab-foundry data corpus materialize \
  --recipe tf_rd_013_dagzoo_shape_aware_size_small_v1 \
  --dagzoo-root ../dagzoo \
  --force

.venv/bin/tab-foundry data corpus materialize \
  --recipe tf_rd_013_dagzoo_shape_aware_size_medium_v1 \
  --dagzoo-root ../dagzoo \
  --force

.venv/bin/tab-foundry data corpus materialize \
  --recipe tf_rd_013_dagzoo_shape_aware_size_large_v1 \
  --dagzoo-root ../dagzoo \
  --force
```

The TF-RD-013 support-summary wrapper can then be used to refresh the tracked
support JSONs from those first-class corpus records:

```bash
.venv/bin/python scripts/materialize_tf_rd_013_support.py --variant size-ladder --force
```

Environment assumptions:

- `TAB_FOUNDRY_ROOT` is this repo root.
- `DAGZOO_ROOT` defaults to the sibling checkout `../dagzoo`.
- The canonical local artifacts for this flow now live under `outputs/corpora/<recipe_id>/<corpus_id>/`.
- The fresh current-corpus control now resolves through recipe `tf_rd_013_current_corpus_default_v1`; stale absolute-path local snapshots are unsupported.
- Dagzoo generation stays fixed at `--device cpu --hardware-policy none` for every rung so the ladder isolates corpus content rather than generator hardware.
- The ladder reuses the same three config-backed regimes as the earlier shape-aware follow-up:
  - `../dagzoo/configs/benchmark_cpu.yaml`
  - `../dagzoo/configs/default.yaml`
  - `../dagzoo/configs/benchmark_cuda_h100_large_shape.yaml`

What the script does for this variant:

- materializes the four TF-RD-013 corpus recipes through the shared `tab-foundry data corpus materialize` pathway
- reuses the resulting local `corpus_record.json` artifacts instead of owning dagzoo generation logic directly
- writes tracked JSON summaries for the fresh current-corpus control plus the three runnable dagzoo size-ladder surfaces

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

- Generated dagzoo shards, manifests, and corpus records live under `outputs/corpora/`.
- `outputs/` stays ignored because the generated artifacts are too large to commit as fixtures.

Policy notes:

- This bundle is binary-only and intentionally omits the curated real-data comparator and multiclass augmentation work.
- The ladder exists to establish a canonical fresh current-corpus control and then test whether shrunken shape-aware dagzoo corpora change the TF-RD-013 keep/defer read before TF-RD-018.
- Issue `#124` remains later filtering-policy work only if the size ladder leaves dagzoo plausible but exposes a narrower predictability or quality-policy question.

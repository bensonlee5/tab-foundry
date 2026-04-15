# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_muon_width_depth_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_muon_width_depth_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_muon_width_depth_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_muon_width_screen_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_muon_width_depth_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `7fab975ee65c3e5f1a14150773c8b745cde9ad9e61d8db357bc423fcc0199bfb`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Surface role: `classification_scaling_law`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1383`, final BPF `2.1383`, final log loss `0.3951`, final Brier score `0.2585`, best ROC AUC `0.7563`, final ROC AUC `0.7583`, final training time `1825.3s`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Detailed Rows

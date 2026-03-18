# Input Norm Follow-Up Comparison Matrix

This file documents the draft `input_norm_followup` queue anchored on the metadata-only bridge baseline `dpnb_row_cls_cls2_linear_warmup_decay`.

Canonical sources:

- `reference/system_delta_sweeps/input_norm_followup/sweep.yaml`
- `reference/system_delta_sweeps/input_norm_followup/queue.yaml`
- `reference/system_delta_catalog.yaml`

## Intent

- Keep the promoted prenorm plus row-cls bridge architecture fixed.
- Re-answer the open preprocessing question: should `train_zscore_clip` remain the default, or should the baseline move toward unclipped or smooth-tail normalization?
- The primary smooth-tail row is `train_zscore_tanh`, with `train_robust_tanh` as the robust comparator.
- Add bounded batch-size sidecars at 16 and 64 with explicit sqrt LR scaling around the reference batch size of 32.

## Anchor

- Baseline surface: `dpnb_row_cls_cls2_linear_warmup_decay`
- Status: metadata-only on this machine because the promoted artifacts live on another computer
- Required replay row: `dpnb_input_norm_anchor_replay`

## Queue

| Order | Delta Ref | Family | Focus | Notes |
| --- | --- | --- | --- | --- |
| 1 | `dpnb_input_norm_anchor_replay` | `baseline_replay` | Local baseline replay | Required because the promoted bridge artifacts are not synced locally. |
| 2 | `dpnb_input_norm_zscore` | `input_normalization` | Remove hard clipping | Direct answer to whether clipping is still needed. |
| 3 | `dpnb_input_norm_winsorize_zscore` | `input_normalization` | Percentile-bounded z-score | Keeps outlier defense without a fixed post-z-score clip. |
| 4 | `dpnb_input_norm_zscore_tanh` | `input_normalization` | Smooth-tail z-score | Primary smooth-tail candidate. |
| 5 | `dpnb_input_norm_robust_tanh` | `input_normalization` | Robust smooth-tail | Robust comparator to row 4. |
| 6 | `dpnb_input_norm_anchor_replay_batch16_sqrt` | `batch_size` | Batch 16 anchor replay | `prior_dump_batch_size=16`, `sqrt` LR scaling. |
| 7 | `dpnb_input_norm_anchor_replay_batch64_sqrt` | `batch_size` | Batch 64 anchor replay | `prior_dump_batch_size=64`, `sqrt` LR scaling. |
| 8 | `dpnb_input_norm_zscore_tanh_batch16_sqrt` | `batch_size` | Smooth-tail plus batch 16 | Interaction probe after row 4. |
| 9 | `dpnb_input_norm_zscore_tanh_batch64_sqrt` | `batch_size` | Smooth-tail plus batch 64 | Interaction probe after row 4. |

## Interpretation Guardrails

- Treat rows 2-5 as the main scientific question and rows 6-9 as sidecar interaction probes.
- The batch-size rows keep the same base schedule shape and use sqrt LR scaling rather than re-tuning the whole optimizer surface.
- RMSNorm is intentionally excluded from this sweep so the result stays attributable to input normalization and batch size.

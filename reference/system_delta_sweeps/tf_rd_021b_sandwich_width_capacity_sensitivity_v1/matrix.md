# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_021b_sandwich_width_capacity_sensitivity_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_021b_sandwich_width_capacity_sensitivity_v1`
- Sweep status: `blocked_on_knob_sensitivity`
- Parent sweep id: `tf_rd_021b_sandwich_knob_sensitivity_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `tf_rd_021b_hybrid_full_cell_compact_prior_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_hybrid_prior`
- Training config profile: `cls_benchmark_sandwich_hybrid_prior`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.4672`, final Brier score `0.3072`, best ROC AUC `0.7370`, final ROC AUC `0.7370`, final training time `470.5s`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| trunk width | Width is the primary channel-capacity axis once topology is stable enough to scale. | Compact hybrid control uses `d_icl=60` with `sandwich_heads=4`. | These rows read whether width is the first live post-topology scaling axis. |
| readout-head width | Prediction-head capacity can matter independently of trunk width on compact models. | Compact hybrid control uses `head_hidden_dim=96`. | These rows test whether head width matters independently before the later power-curve phase. |
| attention partitioning | Not changed here. | Keep `sandwich_heads=4` fixed. | Width rows should not confound total width with head-factor movement. |
| benchmark contract | Not applicable. | Locked medium binary bundle with no external comparator. | Keep the benchmark bundle fixed so this follow-up measures width or head sensitivity only. |
| training surface | Not specified by PerceiverIO. | Training surface label `prior_cosine_warmup`. | Budget and optimizer stay frozen so this follow-up is still an architecture-capacity read. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_021b_sandwich_dicl48_v1` | width_capacity | yes | blocked_on_knob_sensitivity | none | Reduce hybrid sandwich width from 60 to 48 while keeping head count fixed at 4 and the rest of the compact control unchanged. | Wait for the TF-RD-021B knob-sensitivity screen before executing this width-capacity follow-up. |
| 2 | `delta_tf_rd_021b_sandwich_dicl96_v1` | width_capacity | yes | blocked_on_knob_sensitivity | none | Increase hybrid sandwich width from 60 to 96 while keeping head count fixed at 4 and the rest of the compact control unchanged. | Wait for the TF-RD-021B knob-sensitivity screen before executing this width-capacity follow-up. |
| 3 | `delta_tf_rd_021b_sandwich_headhidden64_v1` | width_capacity | yes | blocked_on_knob_sensitivity | none | Reduce the prediction head hidden size from 96 to 64 while keeping the hybrid trunk fixed at the compact control. | Wait for the TF-RD-021B knob-sensitivity screen before executing this width-capacity follow-up. |
| 4 | `delta_tf_rd_021b_sandwich_headhidden128_v1` | width_capacity | yes | blocked_on_knob_sensitivity | none | Increase the prediction head hidden size from 96 to 128 while keeping the hybrid trunk fixed at the compact control. | Wait for the TF-RD-021B knob-sensitivity screen before executing this width-capacity follow-up. |

## Detailed Rows

### 1. `delta_tf_rd_021b_sandwich_dicl48_v1`

- Dimension family: `model`
- Status: `blocked_on_knob_sensitivity`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Reduce hybrid sandwich width from 60 to 48 while keeping head count fixed at 4 and the rest of the compact control unchanged.
- Rationale: Use the smaller width bracket first to test whether the compact hybrid control is already wider than needed.
- Hypothesis: If the compact control is width-overprovisioned, reducing `d_icl` from `60` to `48` should only weakly harm the final benchmark metrics.
- Upstream delta: Width is the first direct channel-capacity probe once sandwich-specific topology sensitivity is known.
- Anchor delta: Keep the compact hybrid control fixed and change only `d_icl` from `60` to `48` while holding `sandwich_heads=4` fixed.
- Expected effect: If the compact control is width-overprovisioned, dropping to `d_icl=48` should preserve most of the benchmark fit.
- Effective labels: model=`cls_benchmark_sandwich_hybrid_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 48, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Execute only after the first sandwich-specific sensitivity screen is interpreted.
  - Compare directly against the locked compact control before any longer-budget power-curve work opens.
- Adequacy knobs to dimension explicitly:
  - Keep `sandwich_heads=4` fixed so this row isolates channel width rather than attention partitioning.
  - Execute only after the stage-1 knob screen is interpreted.
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_width_capacity_sensitivity_v1/delta_tf_rd_021b_sandwich_dicl48_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_021b_sandwich_dicl96_v1`

- Dimension family: `model`
- Status: `blocked_on_knob_sensitivity`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Increase hybrid sandwich width from 60 to 96 while keeping head count fixed at 4 and the rest of the compact control unchanged.
- Rationale: Use the larger width bracket to test whether the compact hybrid control is still width-limited.
- Hypothesis: If trunk width is the first live post-topology capacity axis, increasing `d_icl` from `60` to `96` should improve benchmark fit enough to justify later power-curve work.
- Upstream delta: Width is the primary candidate scaling axis if sandwich-specific topology knobs prove low-sensitivity.
- Anchor delta: Keep the compact hybrid control fixed and change only `d_icl` from `60` to `96` while holding `sandwich_heads=4` fixed.
- Expected effect: If the compact control is width-limited, increasing `d_icl` to `96` should improve fit enough to justify later power-curve work.
- Effective labels: model=`cls_benchmark_sandwich_hybrid_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Execute only after the first sandwich-specific sensitivity screen is interpreted.
  - Compare directly against the locked compact control before any multi-rung scaling program is authored.
- Adequacy knobs to dimension explicitly:
  - Keep `sandwich_heads=4` fixed so the width read is not confounded by head-factor changes.
  - Execute only after the stage-1 knob screen is interpreted.
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_width_capacity_sensitivity_v1/delta_tf_rd_021b_sandwich_dicl96_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_021b_sandwich_headhidden64_v1`

- Dimension family: `model`
- Status: `blocked_on_knob_sensitivity`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Reduce the prediction head hidden size from 96 to 64 while keeping the hybrid trunk fixed at the compact control.
- Rationale: Use the smaller head-hidden bracket to test whether the compact control readout is overprovisioned relative to the trunk.
- Hypothesis: If the trunk is the real bottleneck, reducing `head_hidden_dim` from `96` to `64` should only weakly harm the benchmark read.
- Upstream delta: Head MLP size is a narrower capacity axis than trunk width and may be unnecessary if the trunk is the real bottleneck.
- Anchor delta: Keep the compact hybrid control fixed and change only `head_hidden_dim` from `96` to `64`.
- Expected effect: If the compact control is trunk-limited rather than head-limited, shrinking the head hidden size should have little effect.
- Effective labels: model=`cls_benchmark_sandwich_hybrid_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 64, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Execute only after the first sandwich-specific sensitivity screen is interpreted.
  - Compare directly against the locked compact control before treating readout-head width as a scaling axis.
- Adequacy knobs to dimension explicitly:
  - Keep `d_icl=60` and all sandwich-specific knobs fixed so this row isolates readout-head capacity.
  - Execute only after the stage-1 knob screen is interpreted.
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_width_capacity_sensitivity_v1/delta_tf_rd_021b_sandwich_headhidden64_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_021b_sandwich_headhidden128_v1`

- Dimension family: `model`
- Status: `blocked_on_knob_sensitivity`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Increase the prediction head hidden size from 96 to 128 while keeping the hybrid trunk fixed at the compact control.
- Rationale: Use the larger head-hidden bracket to test whether the compact control is readout-limited even if the trunk stays fixed.
- Hypothesis: If the hybrid control is readout-limited, increasing `head_hidden_dim` from `96` to `128` should improve the final benchmark metrics without changing trunk width.
- Upstream delta: Head hidden size is the bounded readout-capacity follow-up once sandwich-specific topology sensitivity is understood.
- Anchor delta: Keep the compact hybrid control fixed and change only `head_hidden_dim` from `96` to `128`.
- Expected effect: If the compact control is head-limited at readout time, increasing the head hidden size may improve fit even when trunk width stays fixed.
- Effective labels: model=`cls_benchmark_sandwich_hybrid_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 128, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Execute only after the first sandwich-specific sensitivity screen is interpreted.
  - Compare directly against the locked compact control before the later sandwich-local power-curve phase opens.
- Adequacy knobs to dimension explicitly:
  - Keep `d_icl=60` and all sandwich-specific knobs fixed so this row isolates head capacity.
  - Execute only after the stage-1 knob screen is interpreted.
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_width_capacity_sensitivity_v1/delta_tf_rd_021b_sandwich_headhidden128_v1/result_card.md`
- Benchmark metrics: pending

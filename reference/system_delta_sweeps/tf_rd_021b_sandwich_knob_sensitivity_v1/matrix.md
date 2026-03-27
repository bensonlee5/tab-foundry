# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_021b_sandwich_knob_sensitivity_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_021b_sandwich_knob_sensitivity_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_021a_sandwich_nanotabpfn_screen_v1`
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
| latent bank size | Perceiver-style models use latent count as an explicit memory and fidelity knob. | Compact hybrid control uses `sandwich_latents=24`. | This screen asks whether the current local control is already overprovisioned or still materially sensitive on latent-token count. |
| repeated cross-attend depth | Perceiver stacks can trade repeated latent reads against compute. | Compact hybrid control uses `sandwich_layers=2`. | If one repeated stage is enough here, later scaling should not keep repeated depth as a first-class free knob. |
| attention partitioning | Transformer head count changes factorization without changing total width. | Compact hybrid control uses `sandwich_heads=4`. | A weak ablation would argue for freezing head count rather than carrying it into a later compound scaling recipe. |
| feed-forward width | FF expansion changes per-token capacity independently of attention routing. | Compact hybrid control uses `sandwich_ff_expansion=2`. | This read separates trunk MLP capacity from the architectural gains of the hybrid full-cell path. |
| summary-stream bandwidth | The successor widened row and column summaries from one token per axis position to `K=4`. | Compact hybrid control uses `sandwich_summary_tokens_per_axis=4`. | If this ablation is weak, summary-token multiplicity is a good candidate to freeze lower before later scaling work. |
| latent self-refinement depth | Perceiver-style models can vary how much latent self-attention happens between cross-attention reads. | Compact hybrid control uses `sandwich_self_attention_per_cross=4`. | This read tests whether most of the gain comes from repeated latent refinement or from the stage-`0` and readout structural bypasses. |
| pre-Perceiver row mixer | The successor adds per-row feature self-attention before the first latent read. | Compact hybrid control uses `sandwich_pre_row_attention_layers=1`. | This isolates whether row-wise feature mixing is materially useful before the latent bottleneck. |
| pre-Perceiver column mixer | The successor also adds per-column row self-attention before the first latent read. | Compact hybrid control uses `sandwich_pre_column_attention_layers=1`. | This isolates whether the second axial pre-mixer is carrying signal or just extra compute. |
| benchmark contract | Not applicable. | Locked medium binary bundle with no external comparator. | Keep the bundle and benchmark artifact path fixed so the screen is about architecture sensitivity, not about benchmark-surface movement. |
| training surface | No repo-local prior recipe is fixed by PerceiverIO. | Training surface label `prior_cosine_warmup` on the legacy prior backend. | Keep the training recipe fixed so this screen reads sandwich-specific knobs rather than optimizer or budget adequacy. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_021b_sandwich_latents12_v1` | architecture_sensitivity | yes | completed | none | Halve the hybrid sandwich latent bank from 24 to 12 while keeping the compact control otherwise fixed. | Run as one factor in the first TF-RD-021B sandwich knob-sensitivity screen. |
| 2 | `delta_tf_rd_021b_sandwich_layers1_v1` | architecture_sensitivity | yes | completed | none | Reduce the hybrid sandwich repeated cross-attend stack from 2 stages to 1 while keeping the compact control otherwise fixed. | Run as one factor in the first TF-RD-021B sandwich knob-sensitivity screen. |
| 3 | `delta_tf_rd_021b_sandwich_heads2_v1` | architecture_sensitivity | yes | completed | none | Reduce hybrid sandwich attention heads from 4 to 2 while keeping width and all other compact-control settings fixed. | Run as one factor in the first TF-RD-021B sandwich knob-sensitivity screen. |
| 4 | `delta_tf_rd_021b_sandwich_ffexp1_v1` | architecture_sensitivity | yes | ready | none | Reduce the hybrid sandwich feed-forward expansion from 2x to 1x while keeping the compact control otherwise fixed. | Run as one factor in the first TF-RD-021B sandwich knob-sensitivity screen. |
| 5 | `delta_tf_rd_021b_sandwich_summarytokens1_v1` | architecture_sensitivity | yes | ready | none | Reduce row and column summary token multiplicity from 4 tokens per axis position to 1 while keeping the compact control otherwise fixed. | Run as one factor in the first TF-RD-021B sandwich knob-sensitivity screen. |
| 6 | `delta_tf_rd_021b_sandwich_selfattn1_v1` | architecture_sensitivity | yes | ready | none | Reduce latent self-attention refinement between cross-attention reads from 4 blocks to 1 while keeping the compact control otherwise fixed. | Run as one factor in the first TF-RD-021B sandwich knob-sensitivity screen. |
| 7 | `delta_tf_rd_021b_sandwich_prerow0_v1` | architecture_sensitivity | yes | ready | none | Remove the pre-Perceiver per-row feature self-attention mixer while keeping the compact control otherwise fixed. | Run as one factor in the first TF-RD-021B sandwich knob-sensitivity screen. |
| 8 | `delta_tf_rd_021b_sandwich_precol0_v1` | architecture_sensitivity | yes | ready | none | Remove the pre-Perceiver per-column row self-attention mixer while keeping the compact control otherwise fixed. | Run as one factor in the first TF-RD-021B sandwich knob-sensitivity screen. |

## Detailed Rows

### 1. `delta_tf_rd_021b_sandwich_latents12_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Halve the hybrid sandwich latent bank from 24 to 12 while keeping the compact control otherwise fixed.
- Rationale: Start the screen with the lower latent-bank bracket to read whether the compact hybrid control is already overprovisioned on latent tokens.
- Hypothesis: If the hybrid sandwich fit is not very sensitive to latent-bank size on this surface, halving latents from `24` to `12` should only weakly degrade the final benchmark metrics.
- Upstream delta: Perceiver-style latent banks make latent count the first memory-vs-fidelity ablation on the hybrid sandwich line.
- Anchor delta: Keep the registered compact hybrid control fixed and change only `sandwich_latents` from `24` to `12`.
- Expected effect: If the compact hybrid control is overprovisioned on latent tokens, quality should move only slightly while parameter count and compute fall.
- Effective labels: model=`cls_benchmark_sandwich_hybrid_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 12, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Keep the locked medium bundle, prior-dump surface, and `2500`-step cosine budget fixed.
  - Compare directly against the registered compact hybrid control before any width or budget change is opened.
- Adequacy knobs to dimension explicitly:
  - Keep the locked medium bundle, prior-dump surface, and 2500-step cosine budget fixed.
  - Interpret directly against the registered compact hybrid control before width or head-MLP changes.
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_021b_sandwich_knob_sensitivity_v1_01_delta_tf_rd_021b_sandwich_latents12_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_knob_sensitivity_v1/delta_tf_rd_021b_sandwich_latents12_v1/result_card.md`
- Registered run: `sd_tf_rd_021b_sandwich_knob_sensitivity_v1_01_delta_tf_rd_021b_sandwich_latents12_v1_v1` with final log loss `0.4760`, delta final log loss `+0.0088`, final Brier score `0.3133`, delta final Brier score `+0.0061`, best ROC AUC `0.7292`, final ROC AUC `0.7292`, final-minus-best `+0.0000`, delta final ROC AUC `-0.0078`, delta drift `+0.0000`, delta final training time `-393.7s`

### 2. `delta_tf_rd_021b_sandwich_layers1_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Reduce the hybrid sandwich repeated cross-attend stack from 2 stages to 1 while keeping the compact control otherwise fixed.
- Rationale: Test whether the compact control actually needs two repeated latent-read stages or whether one stage already carries most of the gain.
- Hypothesis: If repeated cross-attend depth is low-sensitivity on this surface, reducing `sandwich_layers` from `2` to `1` should only weakly hurt benchmark fit.
- Upstream delta: Perceiver-style models expose repeated latent-read depth as a distinct capacity knob after the first full-input read.
- Anchor delta: Keep the registered compact hybrid control fixed and change only `sandwich_layers` from `2` to `1`.
- Expected effect: If one repeated stage is already enough on this surface, the compact control may be deeper than necessary.
- Effective labels: model=`cls_benchmark_sandwich_hybrid_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Keep stage-`0` full-cell access and the final cell-stream readout fixed.
  - Compare directly against the compact hybrid control before later scale work treats repeated depth as a live axis.
- Adequacy knobs to dimension explicitly:
  - Keep the stage-0 full-cell read, summary-token count, and train budget fixed.
  - Read this as repeated-stage sensitivity, not as a general model-size change.
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_021b_sandwich_knob_sensitivity_v1_02_delta_tf_rd_021b_sandwich_layers1_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_knob_sensitivity_v1/delta_tf_rd_021b_sandwich_layers1_v1/result_card.md`
- Registered run: `sd_tf_rd_021b_sandwich_knob_sensitivity_v1_02_delta_tf_rd_021b_sandwich_layers1_v1_v1` with final log loss `0.4721`, delta final log loss `+0.0049`, final Brier score `0.3104`, delta final Brier score `+0.0031`, best ROC AUC `0.7275`, final ROC AUC `0.7265`, final-minus-best `-0.0010`, delta final ROC AUC `-0.0105`, delta drift `-0.0010`, delta final training time `-405.8s`

### 3. `delta_tf_rd_021b_sandwich_heads2_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Reduce hybrid sandwich attention heads from 4 to 2 while keeping width and all other compact-control settings fixed.
- Rationale: Read whether head factorization matters independently of the compact control width.
- Hypothesis: If the current compact control mainly needs total channel width rather than four-way head partitioning, reducing `sandwich_heads` from `4` to `2` should only weakly hurt.
- Upstream delta: Attention head factorization is a standard transformer capacity axis that may be overprovisioned on compact tabular models.
- Anchor delta: Keep the registered compact hybrid control fixed and change only `sandwich_heads` from `4` to `2`.
- Expected effect: If the hybrid control mostly needs width rather than head factorization, dropping to two heads should be only mildly harmful.
- Effective labels: model=`cls_benchmark_sandwich_hybrid_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 2, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Keep `d_icl=60` fixed so this is a pure head-factorization ablation.
  - Compare directly against the compact control before any width ladder opens.
- Adequacy knobs to dimension explicitly:
  - Keep `d_icl=60` fixed so this row isolates attention partitioning rather than total width.
  - Compare directly against the compact control before opening any width ladder.
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Canonical rerun registered as `sd_tf_rd_021b_sandwich_knob_sensitivity_v1_03_delta_tf_rd_021b_sandwich_heads2_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_knob_sensitivity_v1/delta_tf_rd_021b_sandwich_heads2_v1/result_card.md`
- Registered run: `sd_tf_rd_021b_sandwich_knob_sensitivity_v1_03_delta_tf_rd_021b_sandwich_heads2_v1_v1` with final log loss `0.4641`, delta final log loss `-0.0031`, final Brier score `0.3053`, delta final Brier score `-0.0020`, best ROC AUC `0.7369`, final ROC AUC `0.7366`, final-minus-best `-0.0003`, delta final ROC AUC `-0.0005`, delta drift `-0.0003`, delta final training time `-402.5s`

### 4. `delta_tf_rd_021b_sandwich_ffexp1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Reduce the hybrid sandwich feed-forward expansion from 2x to 1x while keeping the compact control otherwise fixed.
- Rationale: Determine whether the compact control gain is coming from architectural routing or from extra FF capacity in the latent trunk.
- Hypothesis: If trunk MLP width is not the dominant factor here, reducing `sandwich_ff_expansion` from `2` to `1` should only weakly degrade the benchmark read.
- Upstream delta: FF expansion is one of the simplest capacity axes to simplify before introducing broader scaling laws.
- Anchor delta: Keep the registered compact hybrid control fixed and change only `sandwich_ff_expansion` from `2` to `1`.
- Expected effect: If most of the current gain comes from attention structure rather than trunk MLP capacity, shrinking FF expansion should only weakly degrade performance.
- Effective labels: model=`cls_benchmark_sandwich_hybrid_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 1, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Keep latent count, stage count, and summary-token multiplicity fixed.
  - Compare directly against the compact control before carrying FF expansion into a later compound scaling recipe.
- Adequacy knobs to dimension explicitly:
  - Keep latent count, stage count, and summary-token count fixed so this row isolates FF capacity.
  - Compare directly against the compact control before treating FF expansion as a live scaling axis.
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_knob_sensitivity_v1/delta_tf_rd_021b_sandwich_ffexp1_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_021b_sandwich_summarytokens1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Reduce row and column summary token multiplicity from 4 tokens per axis position to 1 while keeping the compact control otherwise fixed.
- Rationale: Test whether the widened `K=4` row or column summary stream actually matters once stage `0` and the readout can already see the full cell stream.
- Hypothesis: If most of the recovered signal comes from the raw-cell bypass, reducing `sandwich_summary_tokens_per_axis` from `4` to `1` should only weakly hurt.
- Upstream delta: The hybrid successor deliberately widened the summary stream with `K=4`; this row tests whether that extra summary bandwidth actually matters.
- Anchor delta: Keep the registered compact hybrid control fixed and change only `sandwich_summary_tokens_per_axis` from `4` to `1`.
- Expected effect: If the gain comes mostly from stage-0 raw cells and the final cell readout, collapsing summary multiplicity back to 1 should only weakly hurt.
- Effective labels: model=`cls_benchmark_sandwich_hybrid_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 1, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Keep the stage-`0` full-cell read and final cell-stream readout fixed.
  - Compare directly against the compact control before any larger summary-token ladder is considered.
- Adequacy knobs to dimension explicitly:
  - Keep the stage-0 full-cell path and final full-cell readout fixed.
  - Read this as summary-stream bandwidth sensitivity, not as a generic width ablation.
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_knob_sensitivity_v1/delta_tf_rd_021b_sandwich_summarytokens1_v1/result_card.md`
- Benchmark metrics: pending

### 6. `delta_tf_rd_021b_sandwich_selfattn1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Reduce latent self-attention refinement between cross-attention reads from 4 blocks to 1 while keeping the compact control otherwise fixed.
- Rationale: Determine whether the compact control really needs four latent self-attention blocks between cross-attention segments.
- Hypothesis: If the hybrid gain comes mostly from the structural bypasses rather than repeated latent refinement, reducing `sandwich_self_attention_per_cross` from `4` to `1` should only weakly hurt.
- Upstream delta: The hybrid successor increased self-refinement depth between cross-attention segments; this row tests whether that extra latent processing is carrying quality.
- Anchor delta: Keep the registered compact hybrid control fixed and change only `sandwich_self_attention_per_cross` from `4` to `1`.
- Expected effect: If repeated latent self-processing is not the bottleneck, dropping to one self-attention block should preserve most of the fit.
- Effective labels: model=`cls_benchmark_sandwich_hybrid_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 1, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Keep stage count and latent count fixed so this row isolates latent self-refinement depth.
  - Compare directly against the compact control before treating self-attention depth as a scaling axis.
- Adequacy knobs to dimension explicitly:
  - Keep stage count and latent count fixed so this row isolates self-refinement depth.
  - Compare directly against the compact control before turning self-attention depth into a future scaling axis.
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_knob_sensitivity_v1/delta_tf_rd_021b_sandwich_selfattn1_v1/result_card.md`
- Benchmark metrics: pending

### 7. `delta_tf_rd_021b_sandwich_prerow0_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Remove the pre-Perceiver per-row feature self-attention mixer while keeping the compact control otherwise fixed.
- Rationale: Test whether the row-wise feature pre-mixer is carrying the gain or whether raw-cell access alone is doing most of the work.
- Hypothesis: If stage `0` full-cell access is sufficient without extra row-wise feature mixing, removing `sandwich_pre_row_attention_layers` should only weakly degrade the control.
- Upstream delta: The hybrid successor added a row-wise feature mixer before the first latent read; this row asks whether that axial pre-mixer is materially useful.
- Anchor delta: Keep the registered compact hybrid control fixed and change only `sandwich_pre_row_attention_layers` from `1` to `0`.
- Expected effect: If most of the recovery comes from raw-cell access alone, removing the row-wise feature mixer may be only weakly harmful.
- Effective labels: model=`cls_benchmark_sandwich_hybrid_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 0, 'sandwich_pre_column_attention_layers': 1}`
- Parameter adequacy plan:
  - Keep the column-wise pre-mixer on so the ablation isolates row-wise feature mixing.
  - Compare directly against the compact control before adding deeper axial pre-mixers.
- Adequacy knobs to dimension explicitly:
  - Keep the column-wise pre-mixer on so the ablation isolates row-wise feature mixing.
  - Compare directly against the compact control before adding deeper axial pre-mixers.
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_knob_sensitivity_v1/delta_tf_rd_021b_sandwich_prerow0_v1/result_card.md`
- Benchmark metrics: pending

### 8. `delta_tf_rd_021b_sandwich_precol0_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Remove the pre-Perceiver per-column row self-attention mixer while keeping the compact control otherwise fixed.
- Rationale: Test whether the column-wise row pre-mixer is necessary after the row-wise feature mixer and the stage-`0` full-cell read are already present.
- Hypothesis: If the second axial pre-mixer is not materially useful, removing `sandwich_pre_column_attention_layers` should only weakly degrade the control.
- Upstream delta: The hybrid successor added a column-wise row mixer before the first latent read; this row tests whether that second axial pass is necessary.
- Anchor delta: Keep the registered compact hybrid control fixed and change only `sandwich_pre_column_attention_layers` from `1` to `0`.
- Expected effect: If the row-wise feature mixer already captures most of the useful pre-latent structure, removing the column-wise row mixer may be low-impact.
- Effective labels: model=`cls_benchmark_sandwich_hybrid_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 96, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 4, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 0}`
- Parameter adequacy plan:
  - Keep the row-wise feature mixer on so the ablation isolates column-wise row mixing.
  - Compare directly against the compact control before deeper axial-mixer changes are considered.
- Adequacy knobs to dimension explicitly:
  - Keep the row-wise feature mixer on so the ablation isolates column-wise row mixing.
  - Compare directly against the compact control before deeper axial-mixer changes are considered.
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021b_sandwich_knob_sensitivity_v1/delta_tf_rd_021b_sandwich_precol0_v1/result_card.md`
- Benchmark metrics: pending

# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_024_classification_knob_sweep_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_024_classification_knob_sweep_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_024_classification_knob_sweep_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_010_classification_evolution_medium_v4`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_024_classification_knob_sweep_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `d4e93d45dd0630b3354b85a3b73462f55adc6c76ec8642452ed1abfc4e8226ff`

## Locked Surface

- Anchor run id: `sd_tf_rd_010_classification_evolution_medium_v4_01_delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v8`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1`
- Surface role: `classification_architecture_followup`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1136`, final BPF `2.1136`, final log loss `0.6812`, final Brier score `0.4229`, best ROC AUC `0.6094`, final ROC AUC `0.6094`, final training time `7449.8s`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| latent bank size | Perceiver-style models use latent count as an explicit memory and fidelity knob. | TF-RD-010 medium control uses `sandwich_latents=24`. | Read latent count only as a bounded post-performance architecture knob on the closed multiclass benchmark contract. |
| attention partitioning | Transformer head count changes factorization without changing total width. | TF-RD-010 medium control uses `sandwich_heads=4`. | If the lower head bracket is weak here, head count should stay frozen outside later scale-law work. |
| feed-forward width | FF expansion changes per-token capacity independently of attention routing. | TF-RD-010 medium control uses `sandwich_ff_expansion=2`. | This reads trunk MLP capacity only after runtime policy is fixed. |
| summary-stream bandwidth | Sandwich summary tokens trade compact contextual bandwidth against compute. | TF-RD-010 medium control uses `sandwich_summary_tokens_per_axis=3`. | If the current benchmark contract mostly benefits from the broader hybrid structure, summary multiplicity should be a bounded follow-on knob rather than a fresh architecture branch. |
| latent self-refinement depth | Perceiver-style models can vary how much latent self-attention happens between cross-attention reads. | TF-RD-010 medium control uses `sandwich_self_attention_per_cross=4`. | This asks whether repeated latent refinement is still carrying signal once the direct multiclass benchmark surface is fixed. |
| readout capacity | Head-hidden width changes prediction-head capacity without reopening trunk width or depth. | TF-RD-010 medium control uses `head_hidden_dim=96`. | Treat readout width as a bounded post-performance follow-up, not as a proxy for `d_icl` or broader scaling work. |
| benchmark contract | Not applicable. | TF-RD-010 medium contract screens rows first; any keep must then validate on the closed TF-RD-010 large contract. | Medium is the fast screening rung, but promotion requires consistency on the harder large rung before the knob stays live. |
| training and runtime surface | No external paper fixes this exact prior-dump or runtime policy. | Training experiment `cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1` with the inherited TF-RD-022 runtime policy. | Hold optimizer, schedule, batching, and runtime policy fixed so this sweep isolates architecture knobs only. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_024_cls_sandwich_latents12_v1` | architecture_followup | no | blocked_on_runtime_policy | none | Reuse the lower-bracket latent-bank family by reducing `sandwich_latents` from `24` to `12` on the TF-RD-010 multiclass benchmark contract after TF-RD-022 freezes the runtime policy. | Wait for the TF-RD-022 large-validation gate, then screen this row on the TF-RD-010 medium rung and validate any keep signal on the closed large rung. |
| 2 | `delta_tf_rd_024_cls_sandwich_heads2_v1` | architecture_followup | no | blocked_on_runtime_policy | none | Reuse the historical head-partition family by reducing `sandwich_heads` from `4` to `2` on the TF-RD-010 multiclass benchmark contract. | Wait for the TF-RD-022 large-validation gate, then screen this row on the TF-RD-010 medium rung and validate any keep signal on the closed large rung. |
| 3 | `delta_tf_rd_024_cls_sandwich_ffexp1_v1` | architecture_followup | no | blocked_on_runtime_policy | none | Reuse the lower-MLP bracket by reducing `sandwich_ff_expansion` from `2` to `1` on the inherited multiclass benchmark surface. | Wait for the TF-RD-022 large-validation gate, then screen this row on the TF-RD-010 medium rung and validate any keep signal on the closed large rung. |
| 4 | `delta_tf_rd_024_cls_sandwich_summarytokens1_v1` | architecture_followup | no | blocked_on_runtime_policy | none | Reuse the lower summary-stream bracket by reducing `sandwich_summary_tokens_per_axis` from `3` to `1` on the inherited multiclass benchmark surface. | Wait for the TF-RD-022 large-validation gate, then screen this row on the TF-RD-010 medium rung and validate any keep signal on the closed large rung. |
| 5 | `delta_tf_rd_024_cls_sandwich_selfattn1_v1` | architecture_followup | no | blocked_on_runtime_policy | none | Reuse the lower latent-refinement bracket by reducing `sandwich_self_attention_per_cross` from `4` to `1` on the inherited multiclass benchmark surface. | Wait for the TF-RD-022 large-validation gate, then screen this row on the TF-RD-010 medium rung and validate any keep signal on the closed large rung. |
| 6 | `delta_tf_rd_024_cls_sandwich_headhidden64_v1` | architecture_followup | no | blocked_on_runtime_policy | none | Reuse the lower readout-capacity bracket by reducing `head_hidden_dim` from `96` to `64` on the inherited multiclass benchmark surface. | Wait for the TF-RD-022 large-validation gate, then screen this row on the TF-RD-010 medium rung and validate any keep signal on the closed large rung. |
| 7 | `delta_tf_rd_024_cls_sandwich_headhidden128_v1` | architecture_followup | no | blocked_on_runtime_policy | none | Reuse the upper readout-capacity bracket by increasing `head_hidden_dim` from `96` to `128` on the inherited multiclass benchmark surface. | Wait for the TF-RD-022 large-validation gate, then screen this row on the TF-RD-010 medium rung and validate any keep signal on the closed large rung. |

## Detailed Rows

### 1. `delta_tf_rd_024_cls_sandwich_latents12_v1`

- Dimension family: `model`
- Status: `blocked_on_runtime_policy`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Reuse the lower-bracket latent-bank family by reducing `sandwich_latents` from `24` to `12` on the TF-RD-010 multiclass benchmark contract after TF-RD-022 freezes the runtime policy.
- Rationale: Reuse the TF-RD-021B latent-count family on the closed TF-RD-010 medium contract to see whether the retained sandwich remains overprovisioned on latent-bank count once TF-RD-022 freezes the runtime policy.
- Hypothesis: Halving the latent bank from `24` to `12` should preserve most of the medium-rung quality while lowering runtime and VRAM cost if the current multiclass sandwich is still overprovisioned after the runtime-policy cleanup.
- Upstream delta: Reuses the TF-RD-021B latent-count family on the current classification benchmark surface instead of opening a new architecture path.
- Anchor delta: Changes `model.sandwich_latents` from `24` to `12` while holding `d_icl=60`, `sandwich_layers=2`, the TF-RD-010 benchmark contract, and the inherited TF-RD-022 runtime policy fixed.
- Expected effect: If the classification sandwich remains overprovisioned on latent tokens after TF-RD-022, the lower latent bracket should preserve most benchmark quality at lower runtime cost.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `127312872348e6a94c827cd1e78862b6918d5fcb320b6cc14ef98924d7967d97`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 12, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Wait for the TF-RD-022 large-validation gate, then execute this row on the inherited policy surface without changing optimizer or schedule settings.
  - Validate any kept medium-screen signal on the closed large-rung TF-RD-010 contract before treating latent count as a persistent follow-on knob.
- Adequacy knobs to dimension explicitly:
  - closed TF-RD-010 medium contract with curated `tf_rd_010_dagzoo_medium_control_curated_v5`
  - inherited TF-RD-022 runtime policy surface
  - latent-bank count only; width, layers, batch, LR, and clipping remain frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `blocked`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_024_classification_knob_sweep_v1/delta_tf_rd_024_cls_sandwich_latents12_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_024_cls_sandwich_heads2_v1`

- Dimension family: `model`
- Status: `blocked_on_runtime_policy`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Reuse the historical head-partition family by reducing `sandwich_heads` from `4` to `2` on the TF-RD-010 multiclass benchmark contract.
- Rationale: Reuse the TF-RD-021B attention-head family on the closed TF-RD-010 contract to test whether head factorization remains a live post-performance knob once runtime policy is explicit.
- Hypothesis: Reducing `sandwich_heads` from `4` to `2` should stay close on the medium benchmark if the retained sandwich is more trunk-limited than head-partition-limited under the inherited TF-RD-022 policy.
- Upstream delta: Reuses the TF-RD-021B attention-head family on the current classification benchmark surface.
- Anchor delta: Changes `model.sandwich_heads` from `4` to `2` while holding `d_icl=60`, `sandwich_layers=2`, `sandwich_latents=24`, the TF-RD-010 benchmark contract, and the inherited TF-RD-022 runtime policy fixed.
- Expected effect: If the retained sandwich is width-limited rather than head-factorization-limited after TF-RD-022, this lower head count should be close on the inherited benchmark contract.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5deffebfed77ed1947dd846b679cf361aa2aad2ec82b35cb1dddd9ad1cf5fbec`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 2, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute only after the TF-RD-022 runtime policy is explicit and the anchor is rerun on that surface.
  - Keep head count as a live follow-on dimension only if the row stays close on medium and earns large-rung validation.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 medium benchmark contract
  - inherited TF-RD-022 runtime policy surface
  - head factorization only; `d_icl`, `sandwich_layers`, optimizer, and batch remain frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `blocked`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_024_classification_knob_sweep_v1/delta_tf_rd_024_cls_sandwich_heads2_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_024_cls_sandwich_ffexp1_v1`

- Dimension family: `model`
- Status: `blocked_on_runtime_policy`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Reuse the lower-MLP bracket by reducing `sandwich_ff_expansion` from `2` to `1` on the inherited multiclass benchmark surface.
- Rationale: Reuse the TF-RD-021B feed-forward expansion family on the closed TF-RD-010 contract to check whether the retained sandwich still needs the larger trunk MLP bracket after TF-RD-022.
- Hypothesis: Reducing `sandwich_ff_expansion` from `2` to `1` should remain competitive on medium if the current direct-multiclass sandwich gains are mostly structural rather than dependent on the larger FF bracket.
- Upstream delta: Reuses the TF-RD-021B feed-forward expansion family on the current classification contract.
- Anchor delta: Changes `model.sandwich_ff_expansion` from `2` to `1` while holding `d_icl=60`, `sandwich_layers=2`, the TF-RD-010 benchmark contract, and the inherited TF-RD-022 runtime policy fixed.
- Expected effect: If the retained sandwich gains are still mostly structural after TF-RD-022, the lower FF expansion should remain competitive on the benchmark contract.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `f2c1ecd3fcaebc755c7958da6b351cdd82d5418800e5aa864471da1c6413ac69`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 1, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after TF-RD-022 closes and keep the schedule, optimizer, and data front identical to the anchor.
  - Promote the lower FF bracket only if the medium delta is negligible and large-rung validation does not reverse the read.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 medium benchmark contract
  - inherited TF-RD-022 runtime policy surface
  - feed-forward expansion only; no training-dynamics or width/depth reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `blocked`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_024_classification_knob_sweep_v1/delta_tf_rd_024_cls_sandwich_ffexp1_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_024_cls_sandwich_summarytokens1_v1`

- Dimension family: `model`
- Status: `blocked_on_runtime_policy`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Reuse the lower summary-stream bracket by reducing `sandwich_summary_tokens_per_axis` from `3` to `1` on the inherited multiclass benchmark surface.
- Rationale: Reuse the TF-RD-021B summary-token family on the closed TF-RD-010 contract to measure whether the retained sandwich still needs extra summary-stream bandwidth after runtime policy is fixed.
- Hypothesis: Because the direct-multiclass sandwich already keeps the raw-cell bypass, reducing `sandwich_summary_tokens_per_axis` from `3` to `1` may preserve most of the signal if extra summary bandwidth is no longer the binding factor.
- Upstream delta: Reuses the TF-RD-021B summary-token family while respecting the current direct-multiclass benchmark contract.
- Anchor delta: Changes `model.sandwich_summary_tokens_per_axis` from `3` to `1` while holding `d_icl=60`, `sandwich_layers=2`, the TF-RD-010 benchmark contract, and the inherited TF-RD-022 runtime policy fixed.
- Expected effect: If the retained sandwich mostly benefits from the broader hybrid structure rather than extra summary bandwidth, the lower summary-token bracket should stay competitive after TF-RD-022.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5cd409a3bba5d00e6cb71f462d575496282eed909bbdf3768b7c4f2c82efb7e6`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 1, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute only after the TF-RD-022 runtime policy closes and the anchor is rerun on that surface.
  - Carry this as a follow-on knob only if medium and large validation both show it is not materially harmful.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 medium benchmark contract
  - inherited TF-RD-022 runtime policy surface
  - summary-stream bandwidth only; width, layers, optimizer, and batch stay frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `blocked`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_024_classification_knob_sweep_v1/delta_tf_rd_024_cls_sandwich_summarytokens1_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_024_cls_sandwich_selfattn1_v1`

- Dimension family: `model`
- Status: `blocked_on_runtime_policy`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Reuse the lower latent-refinement bracket by reducing `sandwich_self_attention_per_cross` from `4` to `1` on the inherited multiclass benchmark surface.
- Rationale: Reuse the TF-RD-021B latent self-refinement family on the closed TF-RD-010 contract to test whether repeated latent self-attention is still a meaningful contributor once TF-RD-022 stabilizes the runtime surface.
- Hypothesis: Reducing `sandwich_self_attention_per_cross` from `4` to `1` should stay near-anchor if most of the retained gain now comes from the broader hybrid structure rather than repeated latent self-refinement.
- Upstream delta: Reuses the TF-RD-021B latent self-refinement family on the current classification benchmark contract.
- Anchor delta: Changes `model.sandwich_self_attention_per_cross` from `4` to `1` while holding `d_icl=60`, `sandwich_layers=2`, the TF-RD-010 benchmark contract, and the inherited TF-RD-022 runtime policy fixed.
- Expected effect: If most of the retained gain comes from the hybrid full-cell bypasses rather than repeated latent self-refinement, the lower self-attention bracket should stay near-anchor after TF-RD-022.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d4d55f8269b6f191d36d4a7922e284cb9ca1c0f9ce0f34fd12ccbf9e4f858287`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 1, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after the TF-RD-022 runtime policy closes, then compare directly against the inherited benchmark anchor before any broader follow-up.
  - Carry this as a live axis only if both medium and large validation say the lower bracket is competitive.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 medium benchmark contract
  - inherited TF-RD-022 runtime policy surface
  - latent self-attention depth only; width, layers, batch, LR, and clipping stay frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `blocked`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_024_classification_knob_sweep_v1/delta_tf_rd_024_cls_sandwich_selfattn1_v1/result_card.md`
- Benchmark metrics: pending

### 6. `delta_tf_rd_024_cls_sandwich_headhidden64_v1`

- Dimension family: `model`
- Status: `blocked_on_runtime_policy`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Reuse the lower readout-capacity bracket by reducing `head_hidden_dim` from `96` to `64` on the inherited multiclass benchmark surface.
- Rationale: Reuse the TF-RD-021B lower head-hidden bracket on the closed TF-RD-010 contract to see whether the current classifier head is overprovisioned after TF-RD-022.
- Hypothesis: Reducing `head_hidden_dim` from `96` to `64` should stay close to anchor quality if the retained sandwich is still trunk-limited rather than readout-limited under the inherited runtime policy.
- Upstream delta: Reuses the TF-RD-021B head-hidden family on the current classification benchmark contract.
- Anchor delta: Changes `model.head_hidden_dim` from `96` to `64` while holding `d_icl=60`, `sandwich_layers=2`, the TF-RD-010 benchmark contract, and the inherited TF-RD-022 runtime policy fixed.
- Expected effect: If the retained sandwich is still trunk-limited rather than head-limited after TF-RD-022, the smaller readout bracket should stay close to anchor quality.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ab5388692849916ebd8edf8534837a938e98983c7b68d1e18b5c8b5600768cc1`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 64, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after the TF-RD-022 runtime policy closes and compare directly against the inherited benchmark anchor.
  - Keep the lower head bracket only if medium and large validation both show negligible loss.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 medium benchmark contract
  - inherited TF-RD-022 runtime policy surface
  - readout capacity only; `d_icl`, `sandwich_layers`, optimizer, and batch remain frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `blocked`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_024_classification_knob_sweep_v1/delta_tf_rd_024_cls_sandwich_headhidden64_v1/result_card.md`
- Benchmark metrics: pending

### 7. `delta_tf_rd_024_cls_sandwich_headhidden128_v1`

- Dimension family: `model`
- Status: `blocked_on_runtime_policy`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Reuse the upper readout-capacity bracket by increasing `head_hidden_dim` from `96` to `128` on the inherited multiclass benchmark surface.
- Rationale: Reuse the TF-RD-021B upper head-hidden bracket on the closed TF-RD-010 contract to test whether the retained multiclass sandwich is actually readout-limited once TF-RD-022 fixes the runtime surface.
- Hypothesis: Increasing `head_hidden_dim` from `96` to `128` should help only if the current classifier head is the binding readout bottleneck; otherwise it should add cost without a durable medium-rung gain.
- Upstream delta: Reuses the TF-RD-021B head-hidden family on the current classification benchmark contract.
- Anchor delta: Changes `model.head_hidden_dim` from `96` to `128` while holding `d_icl=60`, `sandwich_layers=2`, the TF-RD-010 benchmark contract, and the inherited TF-RD-022 runtime policy fixed.
- Expected effect: If the retained sandwich is readout-limited after TF-RD-022, the larger head-hidden bracket should improve the inherited benchmark fit enough to justify a later large-rung validation.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `197875f59ee5ecd0352178a3320390c4e194e0a9b7388fd007cd202db4f028f2`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 0, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'trace_activations': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 60, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 128, 'sandwich_latents': 24, 'sandwich_layers': 2, 'sandwich_heads': 4, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Execute after the TF-RD-022 runtime policy closes and compare directly against the inherited benchmark anchor.
  - Keep the higher head bracket as a live follow-on only if the medium gain survives large-rung validation.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 medium benchmark contract
  - inherited TF-RD-022 runtime policy surface
  - readout capacity only; `d_icl`, `sandwich_layers`, optimizer, and batch remain frozen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `blocked`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_024_classification_knob_sweep_v1/delta_tf_rd_024_cls_sandwich_headhidden128_v1/result_card.md`
- Benchmark metrics: pending

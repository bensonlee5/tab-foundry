# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_009_muon_width_depth_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_009_muon_width_depth_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_009_muon_width_depth_medium_v1`
- Sweep status: `draft`
- Parent sweep id: `tf_rd_009_muon_width_screen_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_009_muon_width_depth_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `9b8c712509a5dfde46e6e70c24698acf8c43ddf5b2de5a25874243c841111bef`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Surface role: `classification_scaling_law`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.1383`, final BPF `2.1383`, final log loss `0.3951`, final Brier score `0.2585`, best ROC AUC `0.7563`, final ROC AUC `0.7583`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1` | classification_scaling_law | no | ready | none | Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy. | Benchmark `72x1` as the lower fresh-Muon diagonal seed against the carried `128x2` width-screen baseline; historical schedulefree rows remain context only. |
| 2 | `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1` | classification_scaling_law | no | ready | none | Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge. | Benchmark `112x3` after `72x1`; if it lands cleanly, keep it as the carried upper seed for the fresh Muon diagonal. |
| 3 | `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1` | classification_scaling_law | no | ready | none | Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Benchmark `144x4` only after the lower and upper fresh-Muon seeds are in place, then carry it into Muon Phase 2 if the matched-budget evidence is usable. |
| 4 | `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1` | classification_scaling_law | no | ready | none | Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe. | Benchmark `192x5` as the higher interior fresh-Muon row before any upper-extension selector rerun; do not treat it as the reopened `#269` branch. |
| 5 | `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1` | classification_scaling_law | no | ready | none | Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family. | Benchmark `264x6` as the largest retained fresh-Muon Phase-1 row; if it lands, use it as ceiling evidence rather than silently extending into the upper-family branch. |

## Detailed Rows

### 1. `delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the lower diagonal TF-RD-009 joint width-depth row at `d_icl=72`, `sandwich_layers=1`, derived by matching the formal `60x2` anchor against the empirical depth-aware sandwich parameter bridge rather than the older `L * d^2` proxy.
- Rationale: Materialize the lower fresh-Muon diagonal seed at `72x1`, solving the formal `60x2` parameter target on the frozen RTX 8000 bridge after the measured width screen chose `128x2` as the in-family baseline.
- Hypothesis: none
- Upstream delta: TF-RD-009 uses the literature-backed rule that width and depth must be co-designed once depth moves, then picks the concrete lower row from the empirical sandwich bridge `P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2` on the frozen medium surface. That bridge chooses the queue row only; the reported law fit is deferred to measured benchmark-registry `model_size.total_params`.
- Anchor delta: Fresh Muon diagonal row `72x1`, benchmarked relative to the carried `128x2` width-screen baseline while keeping `60x2` as the formal external anchor.
- Expected effect: If the sandwich target can trade one layer for more width while staying genuinely close to the formal `60x2` anchor in parameter scale, `72x1` should show whether the lower diagonal stays competitive against the active carried in-family baseline at matched regime budget.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `11a9899045fd44740fde0f7f7265eb9a975e0e66524421326da56910864cc285`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 72, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 1, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Treat the historical schedulefree `72x1 -> 176x6` diagonal as context only; this Muon family is rederived from the landed `48x2/60x2/96x2/128x2` width screen plus the frozen RTX 8000 planning formulas.
  - Use the formal external Muon anchor `60x2` only to set the lower parameter target, then solve that target at `L=1` and round to width rung `8`, which yields `72x1`.
  - Compare this row against the carried `128x2` Muon width-screen baseline at matched regime budget before promoting any later Phase-2 or upper-family work.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Rederived from the fresh Muon width screen with target `646970` params and solved width `70.65` before rung rounding.
  - Predicted local parameter count `671184` and reserved VRAM `9.32 GB` come from the frozen RTX 8000 planning formulas, not from the historical schedulefree diagonal.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the upper diagonal TF-RD-009 joint width-depth row at `d_icl=112`, `sandwich_layers=3`, derived by matching the width-only upper evidence row `128x2` against the empirical depth-aware sandwich parameter bridge.
- Rationale: Materialize the upper fresh-Muon diagonal seed at `112x3`, solving the carried `128x2` parameter target at `L=3` on the frozen RTX 8000 bridge instead of inheriting the historical schedulefree diagonal.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as the paper-constrained upper seed for the broadened medium-rung family, after which the ladder extends toward the retained `rtx8000_44gb` ceiling using the same row-construction bridge instead of a grid search.
- Anchor delta: Fresh Muon diagonal row `112x3`, benchmarked relative to the carried `128x2` width-screen baseline after solving its parameter target at `L=3`.
- Expected effect: If the fixed-budget law continues above the carried in-family baseline, `112x3` should improve the matched-regime-budget objective while staying within a cleaner stability envelope than the already-warned width-only `128x2` row.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `85814ddf829eacaa8e9c2669efea325c087993e60bdbfd609e9f40154d65a615`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 112, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_layers': 3, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture'}`
- Parameter adequacy plan:
  - Treat `128x2` as the carried in-family Muon baseline from the landed width screen and solve its observed parameter target at `L=3`, rounded to width rung `8`, to produce `112x3`.
  - Keep the post-#271 packed Muon runtime surface and the closed v6 one-epoch contract fixed; this row changes width and depth only.
  - Use this row as the upper seed for the fresh Muon diagonal before any interpolated `L=4` / `L=5` rows or upper-family selector rerun.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - inherited TF-RD-022 compile-eager-dynamic runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Rederived from the fresh Muon width screen with target `2849422` params and solved width `112.99` before rung rounding.
  - Predicted local parameter count `2800205` and reserved VRAM `11.29 GB` come from the frozen RTX 8000 planning formulas, not from the historical schedulefree diagonal.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the first fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=144`, `sandwich_layers=4`, using the rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Materialize the first interpolated fresh-Muon diagonal row at `144x4`, taking the log-space midpoint between the `112x3` upper seed and the retained `32.5 GB` Phase-1 ceiling probe.
- Hypothesis: none
- Upstream delta: TF-RD-009 rederives the Muon Phase-1 diagonal from the landed width screen plus the frozen RTX 8000 planning formulas, then uses log-space parameter interpolation instead of inheriting the historical schedulefree ladder.
- Anchor delta: Fresh Muon diagonal row `144x4`, benchmarked relative to the carried `128x2` width-screen baseline as the first interior interpolated Phase-1 point.
- Expected effect: If the fresh Muon fixed-budget law stays smooth beyond the upper seed, `144x4` should provide the first interior Phase-1 measurement between the carried `128x2` baseline and the retained ceiling probe.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e30f9244fd24579ca5ba343e6f70aee431b39702c986760815536c21f2d96d12`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 144, 'sandwich_layers': 4}`
- Parameter adequacy plan:
  - Interpolate fresh Muon parameter targets in log space between the derived `112x3` upper seed and the retained `32.5 GB` reserved-VRAM Phase-1 ceiling, then solve the `L=4` target on the frozen RTX 8000 bridge.
  - Keep historical schedulefree interpolation rows out of the active family; this row exists only because the fresh Muon diagonal was rederived from the landed width screen.
  - Use `144x4` as the first interior Phase-1 Muon row before any one-epoch NS or batch-critical expansion under `#274`.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Rederived from the fresh Muon width screen with target `5847218` params and solved width `147.01` before rung rounding.
  - Predicted local parameter count `5610697` and reserved VRAM `13.89 GB` come from the frozen RTX 8000 planning formulas, not from the historical schedulefree diagonal.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the second fresh-Muon interpolated TF-RD-009 width-depth row at `d_icl=192`, `sandwich_layers=5`, using the higher rederived log-space interior target between the `112x3` seed and the retained `264x6` Phase-1 ceiling probe.
- Rationale: Materialize the second interpolated fresh-Muon diagonal row at `192x5`, taking the higher log-space interior target between `112x3` and the retained `32.5 GB` Phase-1 ceiling probe.
- Hypothesis: none
- Upstream delta: TF-RD-009 keeps the fresh Muon Phase-1 family below the reopened upper-extension branch by interpolating interior rows on the frozen RTX 8000 bridge rather than by reviving the historical schedulefree ladder or running a grid search.
- Anchor delta: Fresh Muon diagonal row `192x5`, benchmarked relative to the carried `128x2` width-screen baseline as the higher interior interpolated Phase-1 point.
- Expected effect: If the fresh Muon fixed-budget law remains smooth in the pre-ceiling region, `192x5` should extend the interior Phase-1 evidence without collapsing directly into the later upper-family reopen.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `a677177127d418cf5bcc004ada3e268d95d8047b69f20227e79277ae7d283488`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 192, 'sandwich_layers': 5}`
- Parameter adequacy plan:
  - Use the same fresh-Muon log-space interpolation policy as `144x4`, but solve the `L=5` interior target so the Phase-1 family spans the pre-ceiling region cleanly.
  - Interpret this row inside the fresh Muon family `{72x1, 112x3, 128x2, 144x4, 192x5, 264x6}` ordered by measured parameter count, not by the historical schedulefree queue.
  - Keep `#269` as a later upper-family selector rerun after Muon Phase 2; this row remains part of the base fresh-Muon Phase-1 diagonal only.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Rederived from the fresh Muon width screen with target `12209806` params and solved width `195.92` before rung rounding.
  - Predicted local parameter count `11727112` and reserved VRAM `19.57 GB` come from the frozen RTX 8000 planning formulas, not from the historical schedulefree diagonal.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Execute the retained fresh-Muon TF-RD-009 ceiling probe at `d_icl=264`, `sandwich_layers=6`, chosen to land near the carried `32.5 GB` RTX 8000 reserved-memory target while staying in the base Phase-1 family.
- Rationale: Materialize the retained fresh-Muon Phase-1 ceiling probe at `264x6`, solving the frozen RTX 8000 reserved-VRAM target of `32.5 GB` at `L=6` and rounding to width rung `8`.
- Hypothesis: none
- Upstream delta: TF-RD-009 treats this as an intentional Muon Phase-1 ceiling probe derived from the frozen RTX 8000 planning formulas and the landed width screen, rather than as a reopened upper-family row.
- Anchor delta: Fresh Muon diagonal row `264x6`, benchmarked relative to the carried `128x2` width-screen baseline as the retained Phase-1 ceiling probe.
- Expected effect: If the rederived Muon Phase-1 family can reach the retained local ceiling cleanly, `264x6` should either extend the matched-budget law into the near-saturation regime or provide explicit ceiling evidence without needing the later upper-family selector.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `65da367784e87491aa576cccb3457df140819cc49c8244e749915e549d9891b6`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_latents': 24, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_summary_tokens_per_axis': 3, 'sandwich_self_attention_per_cross': 4, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'feature_type_conditioning': 'film', 'floating_likelihood': 'single_gaussian', 'integer_likelihood': 'hybrid_mixture', 'd_icl': 264, 'sandwich_layers': 6}`
- Parameter adequacy plan:
  - Keep the fresh Muon Phase-1 family below the reopened upper-extension branch by retaining the historical RTX 8000 reserved-VRAM target of `32.5 GB` as the local ceiling probe.
  - Solve the ceiling target at `L=6` on the frozen RTX 8000 bridge and round to width rung `8`, yielding `264x6` with predicted reserved memory around `32.33 GB`.
  - Use this row as the largest retained Phase-1 Muon geometry; rerun `#269` only after Muon Phase 2 lands and the upper-family selector is recomputed on Muon-only evidence.
- Adequacy knobs to dimension explicitly:
  - fixed TF-RD-010 curated medium benchmark contract
  - carried post-#271 packed Muon runtime and optimizer surface
  - carried TF-RD-024 heads1 architecture with non-scaling sandwich knobs frozen
  - joint width-depth movement only; no optimizer retune, curriculum slice, or token-budget reopen
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Rederived from the fresh Muon width screen with target `25678017` params and solved width `264.94` before rung rounding.
  - Predicted local parameter count `25495777` and reserved VRAM `32.33 GB` come from the frozen RTX 8000 planning formulas, not from the historical schedulefree diagonal.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_009_muon_width_depth_medium_v1/delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1/result_card.md`
- Benchmark metrics: pending

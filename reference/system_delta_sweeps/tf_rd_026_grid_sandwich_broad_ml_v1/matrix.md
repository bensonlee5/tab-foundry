# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_026_grid_sandwich_broad_ml_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_026_grid_sandwich_broad_ml_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_026_grid_sandwich_broad_ml_v1`
- Sweep status: `ready`
- Parent sweep id: `tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_026_grid_sandwich_broad_ml_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `bd4a5e87d73dbd1c07973314e9a82ef84ac6d4e49481d4728ef8a032aa20979c`

## Locked Surface

- Anchor run id: `null`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_workstation_grid_sandwich`
- Training config profile: `cls_workstation_grid_sandwich`
- Surface role: `classification_grid_broad_ml_followup`
- Comparison policy: `anchor_only`
- Anchor metrics: `pending trusted rerun`

## Anchor Comparison

Upstream reference: `Broad-ML grid sandwich architecture follow-ons` from `Hyper-Connections, Differential Transformer, SwiGLU gated FFNs, and recurrent refinement`.

Pending trusted rerun: no anchor is registered yet, so this matrix records the locked benchmark surface and queue state before the first anchor promotion.

| Dimension | Upstream Broad-ML grid sandwich architecture follow-ons | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| residual topology | Hyper-Connection-style multi-stream residual routing can improve deep transformer optimization by giving mixers more stable state paths. | The grid sandwich control uses ordinary prenorm residuals in each row and column mixer block. | Read `hyper_connection_lite` as a topology test inside the grid row/column blocks, not as a return to routed evidence-bank behavior. |
| distractor suppression | Differential attention subtracts a learned fraction of a second attention map from the primary map to suppress common-mode distractors. | The grid sandwich control uses standard attention in row and column mixing. | Read this as an attention-form test at fixed width, depth, data, and optimizer budget. |
| gated token capacity | SwiGLU-style gated FFNs often improve transformer quality at matched or near-matched parameter budgets. | The grid sandwich control uses GELU FFNs. | Read this as a cell-token FFN capacity test without changing the surrounding grid topology. |
| recurrent refinement | Sharing a transformation across recurrent iterations can improve iterative reasoning while controlling parameter count. | The grid sandwich control instantiates four distinct grid mixer layers. | Read `grid_recurrence_steps=8` as a full-shared-layer efficiency baseline. Use checkpoint-time contiguous chunk ablation/repeat diagnostics to choose any apples-to-apples localized repeat candidate. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_026_grid_control_replay_v1` | grid_sandwich_broad_ml | no | ready | none | Replay the default grid sandwich control surface as the local executable anchor for the TF-RD-026 broad-ML architecture campaign. | Execute row `01` with `--promote-first-executed-row-to-anchor`. |
| 2 | `delta_tf_rd_026_grid_hyper_connection_lite_v1` | grid_sandwich_broad_ml | no | ready | none | Replace standard grid mixer residuals with the lightweight two-stream hyper-connection residual topology. | Execute against the promoted row `01` sweep anchor. |
| 3 | `delta_tf_rd_026_grid_differential_attention_v1` | grid_sandwich_broad_ml | no | ready | none | Switch grid row/column attention from standard attention to the differential attention variant. | Execute against the promoted row `01` sweep anchor. |
| 4 | `delta_tf_rd_026_grid_swiglu_ffn_v1` | grid_sandwich_broad_ml | no | ready | none | Replace grid mixer GELU FFNs with the parameter-matched SwiGLU FFN variant. | Execute against the promoted row `01` sweep anchor. |
| 5 | `delta_tf_rd_026_grid_recurrent_8_v1` | grid_sandwich_broad_ml | no | ready | none | Share one grid mixer layer across eight recurrent refinement steps as a full-shared-layer efficiency baseline. | Execute against the promoted row `01` sweep anchor. |
| 6 | `delta_tf_rd_026_grid_hc_swiglu_combo_v1` | grid_sandwich_broad_ml | no | blocked | none | Combine hyper-connection-lite residual topology with SwiGLU grid FFNs after both standalone mechanisms train cleanly. | Unblock only after rows `02` and `04` both train cleanly. |
| 7 | `delta_tf_rd_026_grid_repeat_candidate_1_v1` | grid_sandwich_broad_ml | no | blocked | none | Blocked placeholder for the top contiguous grid-core repeat candidate selected by the checkpoint perturbation diagnostic. | Run the grid-core perturbation diagnostic after row `01` promotion. |
| 8 | `delta_tf_rd_026_grid_repeat_candidate_2_v1` | grid_sandwich_broad_ml | no | blocked | none | Blocked placeholder for the second contiguous grid-core repeat candidate selected by the checkpoint perturbation diagnostic. | Keep blocked until the top candidate has been selected and reviewed. |
| 9 | `delta_tf_rd_026_grid_recurrent_8_swiglu_v1` | grid_sandwich_broad_ml | no | ready | none | Combine the shared eight-step recurrent grid core with SwiGLU grid FFNs. | Execute against the promoted row `01` sweep anchor after row `05` inference verification passes. |

## Detailed Rows

### 1. `delta_tf_rd_026_grid_control_replay_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Replay the default grid sandwich control surface as the local executable anchor for the TF-RD-026 broad-ML architecture campaign.
- Rationale: Establish a local executable control replay for the grid sandwich surface before evaluating broad-ML architecture changes.
- Hypothesis: Replaying the current `cls_workstation_grid_sandwich` surface should reproduce the carried medium anchor closely enough to serve as the sweep-local anchor for rows `02` through `05`.
- Upstream delta: Local control replay; no upstream architecture change.
- Anchor delta: Replays the default grid sandwich surface with `d_icl=144`, `sandwich_layers=4`, `sandwich_heads=1`, Muon, medium v6 corpus, and 5000 prior-dump steps.
- Expected effect: Should reproduce the carried grid sandwich medium behavior closely enough to anchor isolated broad-ML follow-up rows.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5f4c1c6b7bc1c4899914713f26309b886dd8462b96a3d30170e3569a3e82878a`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'grid_sandwich', 'grid_residual_mode': 'prenorm', 'grid_attention_mode': 'standard', 'grid_ffn_mode': 'gelu', 'd_icl': 144, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'feature_type_conditioning': 'film'}`
- Parameter adequacy plan:
  - Execute this row first with `--promote-first-executed-row-to-anchor`.
  - Treat this as the local anchor replay, then compare its final log loss against the carried `0.4221534937` reference before interpreting candidate gains.
- Adequacy knobs to dimension explicitly:
  - medium v6 corpus held fixed
  - Muon optimizer and 5000-step prior-dump budget held fixed
  - grid sandwich architecture held at `d_icl=144`, `sandwich_layers=4`, `sandwich_heads=1`
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - Carried checkpoint is not locally executable, so row `01` replaces it as the active sweep anchor only after successful promotion.
- Follow-up run ids: `[]`
- Result card path: `/workspace/tab-foundry/outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_control_replay_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_026_grid_hyper_connection_lite_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Replace standard grid mixer residuals with the lightweight two-stream hyper-connection residual topology.
- Rationale: Test whether deeper row/column grid mixing benefits from a lightweight two-stream residual topology.
- Hypothesis: Maintaining two residual streams per cell token should improve optimization or representation flow through the grid core without reopening routed evidence-bank behavior.
- Upstream delta: Inspired by Hyper-Connections-style residual state mixing, adapted to the grid sandwich row/column blocks without reintroducing routed evidence banks.
- Anchor delta: Changes only `model.grid_residual_mode` from `prenorm` to `hyper_connection_lite`.
- Expected effect: May improve optimization and information flow through deeper row/column grid mixing at fixed data, width, depth, and optimizer budget.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `ad2b151950aa00cb214d192f1ea5e4da9a191ce23fe8236f951cecdeb39c147d`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'grid_residual_mode': 'hyper_connection_lite', 'arch': 'grid_sandwich', 'd_icl': 144, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'feature_type_conditioning': 'film', 'grid_attention_mode': 'standard', 'grid_ffn_mode': 'gelu'}`
- Parameter adequacy plan:
  - Execute after row `01` is promoted to the active sweep anchor.
  - Inspect train stability and final benchmark metrics before combining this with any other architecture gate.
- Adequacy knobs to dimension explicitly:
  - isolated `grid_residual_mode` change only
  - medium v6 corpus, Muon, and 5000-step budget fixed
  - no routed evidence-bank behavior or batch/runtime retune
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `/workspace/tab-foundry/outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_hyper_connection_lite_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_026_grid_differential_attention_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Switch grid row/column attention from standard attention to the differential attention variant.
- Rationale: Test whether row and column grid attention benefit from subtractive distractor suppression at fixed width, depth, and optimizer budget.
- Hypothesis: Differential attention should help if tabular grid mixing currently over-attends to common-mode or noisy cross-feature distractors.
- Upstream delta: Inspired by Differential Transformer attention, using subtractive attention maps to suppress common-mode distractors.
- Anchor delta: Changes only `model.grid_attention_mode` from `standard` to `differential`.
- Expected effect: May improve grid mixing if the current attention blocks over-attend to noisy or redundant feature/task context.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `aea39453bbc77886018892631c1d8b92204c0cc275b1278aa7160f3c66fedf5f`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'grid_attention_mode': 'differential', 'arch': 'grid_sandwich', 'd_icl': 144, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'feature_type_conditioning': 'film', 'grid_residual_mode': 'prenorm', 'grid_ffn_mode': 'gelu'}`
- Parameter adequacy plan:
  - Execute after row `01` is promoted to the active sweep anchor.
  - Check shape/runtime stability before reading benchmark quality because this is the most attention-kernel-sensitive row.
- Adequacy knobs to dimension explicitly:
  - isolated `grid_attention_mode` change only
  - shape and attention-kernel behavior verified before quality interpretation
  - medium v6 corpus, Muon, and 5000-step budget fixed
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `/workspace/tab-foundry/outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_differential_attention_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_026_grid_swiglu_ffn_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Replace grid mixer GELU FFNs with the parameter-matched SwiGLU FFN variant.
- Rationale: Test whether gated FFN capacity improves cell-token transformations under the existing grid sandwich topology.
- Hypothesis: Parameter-matched SwiGLU should improve the grid core if the current GELU FFNs are limiting per-cell feature transformations rather than attention routing.
- Upstream delta: Inspired by broad transformer gains from gated FFN blocks such as SwiGLU.
- Anchor delta: Changes only `model.grid_ffn_mode` from `gelu` to `swiglu`.
- Expected effect: May improve per-cell token transformations if the grid core is FFN-capacity-limited rather than attention-limited.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8b6983675a1d2afdcc6ad7e05e7ac52c4d08c30c1eb78572b505a248cee794a3`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'grid_ffn_mode': 'swiglu', 'arch': 'grid_sandwich', 'd_icl': 144, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'feature_type_conditioning': 'film', 'grid_residual_mode': 'prenorm', 'grid_attention_mode': 'standard'}`
- Parameter adequacy plan:
  - Execute after row `01` is promoted to the active sweep anchor.
  - If this row is clean and row `02` is also clean, unblock the combined row `06`.
- Adequacy knobs to dimension explicitly:
  - isolated `grid_ffn_mode` change only
  - hidden size remains parameter-matched by the implementation rule
  - medium v6 corpus, Muon, and 5000-step budget fixed
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `/workspace/tab-foundry/outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_swiglu_ffn_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_026_grid_recurrent_8_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Share one grid mixer layer across eight recurrent refinement steps as a full-shared-layer efficiency baseline.
- Rationale: Test whether a shared recurrent grid core benefits from additional refinement iterations without adding distinct layer parameters.
- Hypothesis: Sharing one grid mixer for eight iterations should help if the grid sandwich needs iterative row/column refinement more than additional independent layer parameters.
- Upstream delta: Inspired by recurrent/refinement-style transformer computation, where repeated shared computation can improve iterative reasoning without adding distinct layer parameters.
- Anchor delta: Sets `model.grid_recurrence_steps=8`; `sandwich_layers=4` remains present in the config but is ignored for the recurrent core layer count.
- Expected effect: May improve grid reasoning if repeated row/column refinement is more useful than the control's four independent grid mixer layers, but interpret any quality change alongside the smaller parameter count.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `db670e4234758a26e92c44c4edad8e7165b1d6cff4f2293cbb43d6b0db9bdca2`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'grid_recurrence_steps': 8, 'arch': 'grid_sandwich', 'd_icl': 144, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'feature_type_conditioning': 'film', 'grid_residual_mode': 'prenorm', 'grid_attention_mode': 'standard', 'grid_ffn_mode': 'gelu'}`
- Parameter adequacy plan:
  - Execute after row `01` is promoted to the active sweep anchor.
  - Interpret quality alongside runtime because this row trades fewer parameters for more shared-core iterations.
- Adequacy knobs to dimension explicitly:
  - isolated `grid_recurrence_steps=8` change only
  - `sandwich_layers=4` remains configured but the recurrent core uses one shared layer
  - interpret quality together with runtime and parameter count
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `/workspace/tab-foundry/outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_recurrent_8_v1/result_card.md`
- Benchmark metrics: pending

### 6. `delta_tf_rd_026_grid_hc_swiglu_combo_v1`

- Dimension family: `model`
- Status: `blocked`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Combine hyper-connection-lite residual topology with SwiGLU grid FFNs after both standalone mechanisms train cleanly.
- Rationale: Combine the two least entangled first-wave mechanisms only after both standalone rows prove trainable.
- Hypothesis: Hyper-connection-lite residual topology and SwiGLU token FFNs may stack if one improves gradient/state routing and the other improves per-cell capacity.
- Upstream delta: Interaction row for the two least entangled first-wave broad-ML mechanisms.
- Anchor delta: Combines only `model.grid_residual_mode=hyper_connection_lite` and `model.grid_ffn_mode=swiglu`.
- Expected effect: May stack if residual topology improves state flow while SwiGLU improves per-cell FFN capacity.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `d20d2a45745a731f5696eb1f99b12f31703a804acbb1142b8624c255d5e5ca71`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'grid_residual_mode': 'hyper_connection_lite', 'grid_ffn_mode': 'swiglu', 'arch': 'grid_sandwich', 'd_icl': 144, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'feature_type_conditioning': 'film', 'grid_attention_mode': 'standard'}`
- Parameter adequacy plan:
  - Keep blocked until rows `02` and `04` both train cleanly.
  - Execute only if both standalone mechanisms are stable enough to make an interaction read meaningful.
- Adequacy knobs to dimension explicitly:
  - combination row only; no attention or recurrence change
  - blocked until the standalone hyper-connection-lite and SwiGLU rows are clean
  - medium v6 corpus, Muon, and 5000-step budget fixed
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - Interaction row is uninterpretable until the standalone hyper-connection-lite and SwiGLU rows are clean.
- Follow-up run ids: `[]`
- Result card path: `/workspace/tab-foundry/outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_hc_swiglu_combo_v1/result_card.md`
- Benchmark metrics: pending

### 7. `delta_tf_rd_026_grid_repeat_candidate_1_v1`

- Dimension family: `model`
- Status: `blocked`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Blocked placeholder for the top contiguous grid-core repeat candidate selected by the checkpoint perturbation diagnostic.
- Rationale: Train the top localized repeat chunk only after checkpoint-time ablate/repeat diagnostics identify a concrete contiguous grid-core region.
- Hypothesis: A diagnostic-selected repeat chunk may improve refinement while keeping the full four-layer grid-core parameter count, making it more apples-to-apples than the shared one-layer recurrent baseline.
- Upstream delta: Uses checkpoint-time ablation/repeat evidence to choose a localized recurrent grid-core chunk before training another architecture row.
- Anchor delta: Placeholder row; after diagnostics, set the selected chunk-repeat training gate while keeping `model.grid_recurrence_steps=null` and the full four distinct grid mixer layers.
- Expected effect: Should preserve the full four-layer grid-core parameter count while adding repeated compute only around the best diagnostic chunk.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5f4c1c6b7bc1c4899914713f26309b886dd8462b96a3d30170e3569a3e82878a`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'grid_sandwich', 'd_icl': 144, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'feature_type_conditioning': 'film', 'grid_residual_mode': 'prenorm', 'grid_attention_mode': 'standard', 'grid_ffn_mode': 'gelu'}`
- Parameter adequacy plan:
  - Keep blocked until row `01` has a local checkpoint and `research grid-core perturb-checkpoint --repeat-count 2 --repeat-count 4` writes JSON/Markdown diagnostic artifacts.
  - Fill in the selected contiguous repeat chunk before execution; do not use the current placeholder as an executable training row.
- Adequacy knobs to dimension explicitly:
  - blocked until `research grid-core perturb-checkpoint --repeat-count 2 --repeat-count 4` ranks contiguous repeat chunks against row `01`
  - instantiate the full four-layer grid core with `grid_recurrence_steps=null`
  - add only the selected chunk-repeat training gate after the diagnostic selects a concrete chunk
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - Placeholder has no concrete repeat chunk until diagnostic artifacts select one.
- Notes:
  - Candidate must preserve the full four-layer grid-core parameter count.
- Follow-up run ids: `[]`
- Result card path: `/workspace/tab-foundry/outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_repeat_candidate_1_v1/result_card.md`
- Benchmark metrics: pending

### 8. `delta_tf_rd_026_grid_repeat_candidate_2_v1`

- Dimension family: `model`
- Status: `blocked`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Blocked placeholder for the second contiguous grid-core repeat candidate selected by the checkpoint perturbation diagnostic.
- Rationale: Train a second localized repeat chunk only if checkpoint diagnostics show more than one credible contiguous repeat candidate.
- Hypothesis: A runner-up repeat chunk can test whether localized recurrence is robust or whether the diagnostic points to a single fragile region.
- Upstream delta: Uses checkpoint-time ablation/repeat evidence to choose a second localized recurrent grid-core chunk only if the diagnostic shows multiple plausible regions.
- Anchor delta: Placeholder row; after diagnostics, set the second selected chunk-repeat training gate while keeping `model.grid_recurrence_steps=null` and the full four distinct grid mixer layers.
- Expected effect: Should preserve the full four-layer grid-core parameter count while testing whether the runner-up localized repeat is robust.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `5f4c1c6b7bc1c4899914713f26309b886dd8462b96a3d30170e3569a3e82878a`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'arch': 'grid_sandwich', 'd_icl': 144, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'feature_type_conditioning': 'film', 'grid_residual_mode': 'prenorm', 'grid_attention_mode': 'standard', 'grid_ffn_mode': 'gelu'}`
- Parameter adequacy plan:
  - Keep blocked unless the multi-repeat diagnostic surfaces a second credible repeat chunk.
  - Fill in the selected contiguous repeat chunk before execution; do not use the current placeholder as an executable training row.
- Adequacy knobs to dimension explicitly:
  - blocked until `research grid-core perturb-checkpoint --repeat-count 2 --repeat-count 4` ranks contiguous repeat chunks against row `01`
  - instantiate the full four-layer grid core with `grid_recurrence_steps=null`
  - skip this row if the diagnostic yields only one credible repeat candidate
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - Placeholder has no concrete repeat chunk until diagnostic artifacts select one.
- Notes:
  - Candidate must preserve the full four-layer grid-core parameter count.
- Follow-up run ids: `[]`
- Result card path: `/workspace/tab-foundry/outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_repeat_candidate_2_v1/result_card.md`
- Benchmark metrics: pending

### 9. `delta_tf_rd_026_grid_recurrent_8_swiglu_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Combine the shared eight-step recurrent grid core with SwiGLU grid FFNs.
- Rationale: Test whether row `05`'s shared recurrent grid-core hypothesis benefits from the SwiGLU FFN capacity that was the strongest standalone TF-RD-026 signal.
- Hypothesis: A one-layer grid core applied eight times may need stronger gated per-cell transformations to turn its better prior-train loss into benchmark generalization.
- Upstream delta: Follow-up to the full-shared-layer recurrence baseline and the standalone SwiGLU gain in TF-RD-026.
- Anchor delta: Combines only `model.grid_recurrence_steps=8` and `model.grid_ffn_mode=swiglu`; residual topology and attention remain standard.
- Expected effect: May preserve row `05`'s better prior-train behavior while adding the gated FFN capacity that improved row `04` benchmark quality.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `7954582d881c1a2c46dbc7181768ab8e3dba552275df8cf7c75b0735ac6fad51`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': False, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 5000}`
- Model overrides: `{'grid_recurrence_steps': 8, 'grid_ffn_mode': 'swiglu', 'arch': 'grid_sandwich', 'd_icl': 144, 'input_normalization': 'train_zscore_clip', 'many_class_base': 10, 'head_hidden_dim': 96, 'sandwich_layers': 4, 'sandwich_heads': 1, 'sandwich_ff_expansion': 2, 'sandwich_pre_row_attention_layers': 1, 'sandwich_pre_column_attention_layers': 1, 'sandwich_pre_column_inducing_tokens': 16, 'sandwich_packed_attention': True, 'feature_type_conditioning': 'film', 'grid_residual_mode': 'prenorm', 'grid_attention_mode': 'standard'}`
- Parameter adequacy plan:
  - Execute only after row `05` inference is verified to apply the shared grid core eight times and its benchmark-best/final metrics reproduce from recorded step checkpoints.
  - Interpret quality alongside runtime and parameter count because this row keeps row `05`'s shared one-layer recurrent core while adding SwiGLU FFN capacity.
- Adequacy knobs to dimension explicitly:
  - combination row only; no hyper-connection or differential-attention change
  - one shared grid mixer layer applied for eight grid-core iterations
  - interpret quality alongside the smaller recurrent parameter count and extra repeated compute
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `all`
- Interpretation status: `pending`
- Decision: `None`
- Confounders:
  - This row combines two mechanisms, so compare against both row `05` recurrent and row `04` SwiGLU before treating it as additive.
  - Activation checkpointing is disabled for this row because the recurrent+SwiGLU combination trips a TorchDynamo activation-checkpoint tracing assertion before the first training step; compile remains enabled.
- Notes:
  - Expected construction is one shared grid mixer layer, eight grid-core applications, SwiGLU grid FFNs, and roughly 1.53M trainable parameters.
  - Runtime differs from rows `01`-`05` only by disabling activation checkpointing to avoid the TorchDynamo checkpoint tracing failure observed on the VM.
- Follow-up run ids: `[]`
- Result card path: `/workspace/tab-foundry/outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_recurrent_8_swiglu_v1/result_card.md`
- Benchmark metrics: pending

# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_026_grid_sandwich_broad_ml_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_026_grid_sandwich_broad_ml_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_026_grid_sandwich_broad_ml_v1`
- Sweep status: `ready`
- Parent sweep id: `tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_026_grid_sandwich_broad_ml_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `cd97def11f6edd957094fd8687cb1958f534fa8d3ceba01bd796f4ac430abc58`

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
| recurrent refinement | Sharing a transformation across recurrent iterations can improve iterative reasoning while controlling parameter count. | The grid sandwich control instantiates four distinct grid mixer layers. | Read `grid_recurrence_steps=8` as a shared-core refinement test rather than a depth scaling result. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_026_grid_control_replay_v1` | grid_sandwich_broad_ml | no | ready | none | Replay the default grid sandwich control surface as the local executable anchor for the TF-RD-026 broad-ML architecture campaign. | Execute row `01` with `--promote-first-executed-row-to-anchor`. |
| 2 | `delta_tf_rd_026_grid_hyper_connection_lite_v1` | grid_sandwich_broad_ml | no | ready | none | Replace standard grid mixer residuals with the lightweight two-stream hyper-connection residual topology. | Execute against the promoted row `01` sweep anchor. |
| 3 | `delta_tf_rd_026_grid_differential_attention_v1` | grid_sandwich_broad_ml | no | ready | none | Switch grid row/column attention from standard attention to the differential attention variant. | Execute against the promoted row `01` sweep anchor. |
| 4 | `delta_tf_rd_026_grid_swiglu_ffn_v1` | grid_sandwich_broad_ml | no | ready | none | Replace grid mixer GELU FFNs with the parameter-matched SwiGLU FFN variant. | Execute against the promoted row `01` sweep anchor. |
| 5 | `delta_tf_rd_026_grid_recurrent_8_v1` | grid_sandwich_broad_ml | no | ready | none | Share one grid mixer layer across eight recurrent refinement steps. | Execute against the promoted row `01` sweep anchor. |
| 6 | `delta_tf_rd_026_grid_hc_swiglu_combo_v1` | grid_sandwich_broad_ml | no | blocked | none | Combine hyper-connection-lite residual topology with SwiGLU grid FFNs after both standalone mechanisms train cleanly. | Unblock only after rows `02` and `04` both train cleanly. |

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
- Resolved surface fingerprint: `08a336c3461868262f721c97bb1626360ebf69b3ea03c15579b228da232ca355`
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
- Result card path: `outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_control_replay_v1/result_card.md`
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
- Resolved surface fingerprint: `b7e85d062d559fd990c9ed5bc1cdf77d46761e18e03f42a00ad1e4bc29832ea1`
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
- Result card path: `outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_hyper_connection_lite_v1/result_card.md`
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
- Resolved surface fingerprint: `715c162a2318a6ffd1c141ff5ba0d33cc93c47fd65bf7204d76a2a87086c768b`
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
- Result card path: `outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_differential_attention_v1/result_card.md`
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
- Resolved surface fingerprint: `be2e04e417182e340f3c72adcb57bbdb0252f78420367f70b4ab6cbf1b3d2348`
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
- Result card path: `outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_swiglu_ffn_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_026_grid_recurrent_8_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Share one grid mixer layer across eight recurrent refinement steps.
- Rationale: Test whether a shared recurrent grid core benefits from additional refinement iterations without adding distinct layer parameters.
- Hypothesis: Sharing one grid mixer for eight iterations should help if the grid sandwich needs iterative row/column refinement more than additional independent layer parameters.
- Upstream delta: Inspired by recurrent/refinement-style transformer computation, where repeated shared computation can improve iterative reasoning without adding distinct layer parameters.
- Anchor delta: Sets `model.grid_recurrence_steps=8`; `sandwich_layers=4` remains present in the config but is ignored for the recurrent core layer count.
- Expected effect: May improve grid reasoning if repeated row/column refinement is more useful than the control's four independent grid mixer layers.
- Effective labels: model=`grid_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `8a34bac2d6d12d62d315b11ebc949a075f850be515683e57efcb18a87a8dbe73`
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
- Result card path: `outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_recurrent_8_v1/result_card.md`
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
- Resolved surface fingerprint: `6c300a126d515ab4d523f372ddef41f2662f19b547f626f6df1e2eb9ba8db8ef`
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
- Result card path: `outputs/staged_ladder/research/tf_rd_026_grid_sandwich_broad_ml_v1/delta_tf_rd_026_grid_hc_swiglu_combo_v1/result_card.md`
- Benchmark metrics: pending

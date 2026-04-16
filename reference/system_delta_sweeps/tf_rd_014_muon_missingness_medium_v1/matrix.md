# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_014_muon_missingness_medium_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_014_muon_missingness_medium_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_014_muon_missingness_medium_v1`
- Sweep status: `ready`
- Parent sweep id: `tf_rd_009_muon_width_depth_medium_v1`
- Complexity level: `classification_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_014_muon_missingness_medium_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `d3d1594ed34ad7255c9a415f4ca83c8e64ccac3d4d898bdcdeb4f3d80aa2f358`

## Locked Surface

- Anchor run id: `sd_tf_rd_009_muon_width_depth_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`
- Benchmark manifest: local benchmark-manifest id `openml_classification_medium_v1`
- Control baseline id: `cls_benchmark_linear_multiclass_medium_v1`
- External benchmarks: `none`
- Training experiment: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Training config profile: `cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1`
- Surface role: `classification_scaling_law`
- Comparison policy: `anchor_only`
- Anchor metrics: final BPC `2.3114`, final BPF `2.3114`, final log loss `0.4009`, final Brier score `0.2635`, best ROC AUC `0.6468`, final ROC AUC `0.7607`, final training time `4030.1s`

## Anchor Comparison

Upstream reference: `PerceiverIO` from `https://openreview.net/forum?id=fILj7WpI-g`.

| Dimension | Upstream PerceiverIO | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| benchmark surface | Not applicable. | Hub-owned medium classification manifest materialized from `openml_classification_medium_v1.json`. | Keep the refreshed multiclass allow-missing validation rung fixed while TF-RD-014 changes only the synthetic training-front missingness regime. |
| model family | Not applicable. | Carried Muon sandwich `264x6` with `sandwich_heads=1`, `sandwich_latents=24`, `sandwich_summary_tokens_per_axis=3`, and `head_hidden_dim=96`. | TF-RD-014 measures robustness on the carried Muon family rather than reopening TF-RD-009 architecture search. |
| runtime policy | Not applicable. | Post-#271 Muon runtime and optimizer bundle with the fixed one-epoch `2500`-step medium contract. | Keep runtime, optimizer, compile, and checkpoint policy fixed so TF-RD-014 isolates only training-front missingness. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control` | provenance | no | ready | none | Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests. | Execute first and use row `01` as the exploratory TF-RD-014 control anchor for direct comparison against rows `02` through `04`; do not promote beyond TF-RD-014. |
| 2 | `delta_data_manifest_root_tf_rd_010_missingness_mcar` | missingness | no | ready | none | Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Execute only after row `01` is recorded, then compare directly against the clean Muon control row; keep the result exploratory and non-promotable. |
| 3 | `delta_data_manifest_root_tf_rd_010_missingness_mar` | missingness | no | ready | none | Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Execute only after rows `01` and `02` are recorded, then compare directly against the clean Muon control row; keep the result exploratory and non-promotable. |
| 4 | `delta_data_manifest_root_tf_rd_010_missingness_mnar` | missingness | no | ready | none | Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed. | Execute only after rows `01` through `03` are recorded, then compare directly against the clean Muon control row; keep the result exploratory and non-promotable. |

## Detailed Rows

### 1. `delta_data_manifest_root_tf_rd_010_dagzoo_medium_control`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 dagzoo classification control corpus (`n_classes_min=2`, `n_classes_max=10`) while the evolved sandwich benchmark contract is defined against hub-owned validation manifests.
- Rationale: Establish the clean TF-RD-014 Muon control row on the carried `264x6` surface before reading any missingness effect on the refreshed multiclass medium benchmark rung.
- Hypothesis: The carried Muon `264x6` control should provide the strongest exploratory baseline on the refreshed medium manifest, and it is the only valid direct comparison point for the missingness rows.
- Upstream delta: Not applicable; this is a repo-local synthetic training-front contract tied to the first benchmark-evolution lane.
- Anchor delta: Keep the carried Muon `264x6` model and runtime surface fixed and train on `tf_rd_010_dagzoo_medium_control_curated_v6` against the refreshed hub-owned medium classification manifest.
- Expected effect: Establish the TF-RD-010 classification control corpus that both the medium and large validation rungs will compare against.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_dagzoo_medium_control`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `65da367784e87491aa576cccb3457df140819cc49c8244e749915e549d9891b6`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_dagzoo_medium_control_curated_v6'}`
- Parameter adequacy plan:
  - Keep the exact Muon runtime and optimizer bundle from `sd_tf_rd_009_muon_width_depth_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1`; TF-RD-014 changes only the training corpus.
  - Use the refreshed hub-owned medium manifest and verify its multiclass allow-missing contract before execution.
  - If a compatible local `264x6` training artifact is restored into the workspace later, benchmark that artifact instead of retraining; otherwise retrain row `01` on the same surface and use it as the exploratory control anchor for rows `02` through `04`.
- Adequacy knobs to dimension explicitly:
  - explicit dagzoo provenance for the classification control corpus
  - medium and large real-data validation separation via `tab-realdata-hub` manifests
  - class-count coverage, feature-count coverage, missingness policy, and minority-class floor on the validation side
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - `tab-realdata-hub` owns the validation manifest; `dagzoo` owns the synthetic training fronts.
  - The refreshed medium manifest now comes from the hub-owned `openml_classification_medium_v1.json` bundle and carries multiclass allow-missing tasks.
  - This package is exploratory robustness evidence only and does not change the carried Muon baseline for TF-RD-009 Phase 2.
  - No compatible reusable `264x6` training artifact is pinned in this checkout today, so row `01` should retrain unless that artifact is restored before execution.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_014_muon_missingness_medium_v1/delta_data_manifest_root_tf_rd_010_dagzoo_medium_control/result_card.md`
- Benchmark metrics: pending

### 2. `delta_data_manifest_root_tf_rd_010_missingness_mcar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MCAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Read whether MCAR exposure changes the carried Muon `264x6` behavior on the refreshed multiclass medium benchmark before structured missingness is considered.
- Hypothesis: MCAR may modestly improve robustness relative to the clean control without the stronger structure of MAR or MNAR, but it should still be judged only against the clean Muon control row.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the carried Muon `264x6` model and runtime surface fixed and replace the clean control corpus with `tf_rd_010_missingness_mcar_v3`.
- Expected effect: Moderate MCAR should test whether the evolved sandwich target benefits from missingness exposure before any larger benchmark-front escalation.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mcar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `a8a3cefcafc9161c1d78a66c6d6985e15e01ffd73fb36ef13483e5c27f36ca10`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mcar_v3'}`
- Parameter adequacy plan:
  - Compare directly against row `01`; do not interpret this row against the older `128x2` screen or the TF-RD-010 sandwich anchor.
  - Keep the refreshed multiclass medium manifest fixed and use `best_and_final` checkpoints only.
  - Treat any quality change as exploratory only because the missingness front remains `include_all` `v3` while the control front is curated `accepted_only` `v6`, and because the legacy `v3` front is smaller than the strict no-repeat `160000`-task contract.
- Adequacy knobs to dimension explicitly:
  - explicit MCAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - natural-log CE/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - `tab-realdata-hub` owns the validation manifest; `dagzoo` owns the synthetic training fronts.
  - The medium validation rung now uses the refreshed hub-owned multiclass allow-missing manifest.
  - This missingness row remains exploratory and non-promotable until curated missingness fronts exist.
  - `tf_rd_010_missingness_mcar_v3` does not meet the strict no-repeat `160000`-task one-epoch contract at `2500 x 16 x 4`, so this row keeps the same runtime bundle and step budget but remains a repeated-task exploratory read.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_014_muon_missingness_medium_v1/delta_data_manifest_root_tf_rd_010_missingness_mcar/result_card.md`
- Benchmark metrics: pending

### 3. `delta_data_manifest_root_tf_rd_010_missingness_mar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Read whether MAR exposure changes the carried Muon `264x6` behavior on the refreshed multiclass medium benchmark after the clean control and MCAR rows are defined.
- Hypothesis: MAR may provide a harder but more interpretable missingness mechanism than MNAR, but it should still be read strictly as exploratory robustness evidence versus the clean Muon control row.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the carried Muon `264x6` model and runtime surface fixed and replace the clean control corpus with `tf_rd_010_missingness_mar_v3`.
- Expected effect: Structured MAR may provide a harder but still interpretable missingness front for the first TF-RD-010 classification benchmark program.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `b17933fbe39720cb3321e570968a957896e0f3adbb20d1e570247e0b9856bfba`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mar_v3'}`
- Parameter adequacy plan:
  - Compare directly against row `01` and secondarily against row `02`; do not mix this row into TF-RD-009 scaling-law fitting.
  - Keep the refreshed multiclass medium manifest fixed and use `best_and_final` checkpoints only.
  - Treat any quality change as exploratory only because the missingness front remains `include_all` `v3` while the control front is curated `accepted_only` `v6`, and because the legacy `v3` front is smaller than the strict no-repeat `160000`-task contract.
- Adequacy knobs to dimension explicitly:
  - explicit MAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - natural-log CE/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - `tab-realdata-hub` owns the validation manifest; `dagzoo` owns the synthetic training fronts.
  - The medium validation rung now uses the refreshed hub-owned multiclass allow-missing manifest.
  - This missingness row remains exploratory and non-promotable until curated missingness fronts exist.
  - `tf_rd_010_missingness_mar_v3` does not meet the strict no-repeat `160000`-task one-epoch contract at `2500 x 16 x 4`, so this row keeps the same runtime bundle and step budget but remains a repeated-task exploratory read.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_014_muon_missingness_medium_v1/delta_data_manifest_root_tf_rd_010_missingness_mar/result_card.md`
- Benchmark metrics: pending

### 4. `delta_data_manifest_root_tf_rd_010_missingness_mnar`

- Dimension family: `data`
- Status: `ready`
- Binary applicable: `False`
- Recipe alias: `none`
- Description: Point training at the TF-RD-010 MNAR classification corpus (`n_classes_min=2`, `n_classes_max=10`) while keeping the evolved sandwich architecture and hub-backed validation contract fixed.
- Rationale: Read whether MNAR exposure changes the carried Muon `264x6` behavior on the refreshed multiclass medium benchmark after the clean control, MCAR, and MAR rows are defined.
- Hypothesis: MNAR may be the hardest synthetic missingness front, but it should still be interpreted only as exploratory robustness evidence versus the clean Muon control row.
- Upstream delta: Not applicable; this is a repo-local synthetic missingness front for the benchmark-evolution lane.
- Anchor delta: Keep the carried Muon `264x6` model and runtime surface fixed and replace the clean control corpus with `tf_rd_010_missingness_mnar_v3`.
- Expected effect: Structured MNAR may be the strongest synthetic missingness perturbation, but it risks a less interpretable first benchmark-evolution read than MCAR or MAR.
- Effective labels: model=`tabfoundry_sandwich`, data=`tf_rd_010_missingness_mnar`, preprocessing=`runtime_default`, training=`prior_cosine_warmup`
- Resolved surface fingerprint: `e7700d1b471567f3d05ff51047049b3fd38ebe6cda7721ebad68308011bbb728`
- Resolved runtime surface: `{'seed': 1, 'mixed_precision': 'bf16', 'num_workers': 'auto', 'loader_pin_memory': True, 'loader_persistent_workers': False, 'loader_prefetch_factor': 'auto', 'loader_task_batch_cache': False, 'loader_task_batch_cache_mode': 'bounded_streaming', 'non_blocking_device_transfer': True, 'grad_clip': 0.0, 'grad_accum_steps': 4, 'compile_model': True, 'compile_dynamic': True, 'compile_backend': 'eager', 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_shape_dispatch_mode': 'signature_family', 'compile_shape_dispatch_max_families': 16, 'trace_activations': False, 'signature_family_run_length': 4, 'module_grad_norm_every': 1, 'profile_step_timing': False, 'activation_checkpointing': True, 'eval_every': 25, 'checkpoint_every': 25, 'val_batches': 0, 'max_steps': 2500}`
- Data overrides: `{'source': 'manifest', 'corpus_ref': 'tf_rd_010_missingness_mnar_v3'}`
- Parameter adequacy plan:
  - Compare directly against row `01` and secondarily against rows `02` and `03`; do not mix this row into TF-RD-009 scaling-law fitting.
  - Keep the refreshed multiclass medium manifest fixed and use `best_and_final` checkpoints only.
  - Treat any quality change as exploratory only because the missingness front remains `include_all` `v3` while the control front is curated `accepted_only` `v6`, and because the legacy `v3` front is smaller than the strict no-repeat `160000`-task contract.
- Adequacy knobs to dimension explicitly:
  - explicit MNAR provenance in the dagzoo training front
  - fixed medium and large hub-owned validation manifests
  - natural-log CE/log-loss ranking under the direct multiclass head contract
- Execution policy: `benchmark_full`
- Benchmark checkpoint selection: `best_and_final`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - `tab-realdata-hub` owns the validation manifest; `dagzoo` owns the synthetic training fronts.
  - The medium validation rung now uses the refreshed hub-owned multiclass allow-missing manifest.
  - This missingness row remains exploratory and non-promotable until curated missingness fronts exist.
  - `tf_rd_010_missingness_mnar_v3` does not meet the strict no-repeat `160000`-task one-epoch contract at `2500 x 16 x 4`, so this row keeps the same runtime bundle and step budget but remains a repeated-task exploratory read.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_014_muon_missingness_medium_v1/delta_data_manifest_root_tf_rd_010_missingness_mnar/result_card.md`
- Benchmark metrics: pending

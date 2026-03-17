# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/binary_md_v3/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `binary_md_v3`
- Sweep status: `draft`
- Parent sweep id: `binary_md_v2`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `01_shared_norm_post_ln_binary_medium_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_missing_v1.json`
- Control baseline id: `cls_benchmark_linear_v3`
- Comparison policy: `anchor_only`
- Anchor metrics: best ROC AUC `0.7670`, final ROC AUC `0.7643`, final training time `505.3s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Shared feature encoder path with benchmark-external normalization. | Feature encoder swaps change both the representation path and where normalization lives. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Same mean-padded linear target conditioner. | The draft missingness sweep preserves the promoted label-conditioning path while isolating missingness handling. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Same nano post-norm cell transformer block. | Transformer-block changes remain out of scope for this follow-on missingness workstream. |
| tokenizer | One scalar token per feature. | Same scalar-per-feature tokenizer. | Tokenization stays fixed so missingness-mode rows isolate only missing-value representation choices. |
| column encoder | None on the upstream direct path. | No column-set encoder on the anchor path. | Column-set modeling remains absent and should not explain any missingness findings. |
| row readout | Target-column readout from the final cell tensor. | Same target-column row pool. | Readout remains on the direct upstream-style path. |
| context encoder | None on the upstream direct path. | None on the anchor path. | Context encoding remains absent; this workstream is about missingness surfaces, not deeper table structure. |
| prediction head | Direct binary logits head. | Direct binary logits head. | The prediction head stays on the narrow upstream-style binary path. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Draft missingness bundle `nanotabpfn_openml_binary_medium_missing` with data surface label `missing_manifest_default`. | Missing-manifest alternatives are first-class queue rows and stay blocked until their artifacts exist. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`, with raw-missing rows switching explicitly to `runtime_no_impute`. | Missingness rows must carry their required runtime policy in the surfaced preprocessing labels instead of inheriting it silently. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_constant_lr`. | Training remains fixed while missingness model, data, and preprocessing surfaces are isolated. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Legacy stage alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_preproc_impute_missing_off` | missing_values | yes | blocked_on_artifacts | none | Disable runtime mean imputation while keeping the fitted label-remap path intact. | Freeze `cls_benchmark_linear_v3` on `nanotabpfn_openml_binary_medium_missing_v1`, register `01_shared_norm_post_ln_binary_medium_missing_v1`, materialize the `missing_manifest_default`, `missing_manifest_forbid_any`, and `missing_manifest_accepted_only` artifacts, and reserve track `system_delta_binary_medium_missing_v1` before moving this row to `ready`. |
| 2 | `delta_model_missingness_feature_mask` | missingness_encoding | yes | blocked_on_artifacts | none | Switch the model missingness surface from `none` to `feature_mask` while carrying the required raw-missing runtime policy with the row. | Freeze `cls_benchmark_linear_v3` on `nanotabpfn_openml_binary_medium_missing_v1`, register `01_shared_norm_post_ln_binary_medium_missing_v1`, materialize the `missing_manifest_default`, `missing_manifest_forbid_any`, and `missing_manifest_accepted_only` artifacts, and reserve track `system_delta_binary_medium_missing_v1` before moving this row to `ready`. |
| 3 | `delta_model_missingness_explicit_token` | missingness_encoding | yes | blocked_on_artifacts | none | Switch the model missingness surface from `none` to `explicit_token` while carrying the required raw-missing runtime policy with the row. | Freeze `cls_benchmark_linear_v3` on `nanotabpfn_openml_binary_medium_missing_v1`, register `01_shared_norm_post_ln_binary_medium_missing_v1`, materialize the `missing_manifest_default`, `missing_manifest_forbid_any`, and `missing_manifest_accepted_only` artifacts, and reserve track `system_delta_binary_medium_missing_v1` before moving this row to `ready`. |
| 4 | `delta_preproc_all_nan_fill_nonzero` | missing_values | yes | blocked_on_artifacts | none | Keep imputation on but change the all-NaN fallback fill value from 0.0 to 1.0. | Freeze `cls_benchmark_linear_v3` on `nanotabpfn_openml_binary_medium_missing_v1`, register `01_shared_norm_post_ln_binary_medium_missing_v1`, materialize the `missing_manifest_default`, `missing_manifest_forbid_any`, and `missing_manifest_accepted_only` artifacts, and reserve track `system_delta_binary_medium_missing_v1` before moving this row to `ready`. |
| 5 | `delta_data_missing_manifest_forbid_any` | missing_values | yes | blocked_on_artifacts | none | Rebuild the missingness manifest surface to exclude any dataset containing NaN or Inf while keeping model and preprocessing fixed. | Freeze `cls_benchmark_linear_v3` on `nanotabpfn_openml_binary_medium_missing_v1`, register `01_shared_norm_post_ln_binary_medium_missing_v1`, materialize the `missing_manifest_default`, `missing_manifest_forbid_any`, and `missing_manifest_accepted_only` artifacts, and reserve track `system_delta_binary_medium_missing_v1` before moving this row to `ready`. |
| 6 | `delta_data_missing_manifest_accepted_only` | missing_values | yes | blocked_on_artifacts | none | Rebuild the missingness manifest surface under accepted-only filtering while preserving missing-valued records that pass curation. | Freeze `cls_benchmark_linear_v3` on `nanotabpfn_openml_binary_medium_missing_v1`, register `01_shared_norm_post_ln_binary_medium_missing_v1`, materialize the `missing_manifest_default`, `missing_manifest_forbid_any`, and `missing_manifest_accepted_only` artifacts, and reserve track `system_delta_binary_medium_missing_v1` before moving this row to `ready`. |

## Detailed Rows

### 1. `delta_preproc_impute_missing_off`

- Dimension family: `preprocessing`
- Status: `blocked_on_artifacts`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Disable runtime mean imputation while keeping the fitted label-remap path intact.
- Rationale: Benchmark the promoted Shared+LayerNorm surface on the draft missingness bundle with runtime imputation disabled, but only after the missingness-specific anchor artifacts exist.
- Hypothesis: Raw-missing support tensors may improve attribution for genuinely missing-valued tasks once the missingness manifests and anchor are frozen.
- Upstream delta: Upstream notebook preprocessing imputes as part of its benchmark helper path.
- Anchor delta: Changes only preprocessing by switching the draft missingness surface from `runtime_default` to `runtime_no_impute`.
- Expected effect: Cleaner attribution around imputation usefulness, but possibly brittle feature tensors on some datasets.
- Effective labels: model=`anchor_model`, data=`missing_manifest_default`, preprocessing=`runtime_no_impute`, training=`prior_constant_lr`
- Preprocessing overrides: `{'impute_missing': False}`
- Parameter adequacy plan:
  - Confirm the finalized missingness bundle actually exercises support-side missing values before interpreting the row.
  - Compare against the frozen missingness anchor rather than the no-missing predecessor.
- Adequacy knobs to dimension explicitly:
  - Interpret against manifest missingness prevalence where available.
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - Required runtime raw-missing policy is encoded in the surfaced preprocessing label, not implied by the queue prose.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v3/delta_preproc_impute_missing_off/result_card.md`
- Benchmark metrics: pending

### 2. `delta_model_missingness_feature_mask`

- Dimension family: `model`
- Status: `blocked_on_artifacts`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Switch the model missingness surface from `none` to `feature_mask` while carrying the required raw-missing runtime policy with the row.
- Rationale: Test feature-mask missingness encoding on the promoted Shared+LayerNorm model path without introducing a separate preprocessing or data axis.
- Hypothesis: Feature-mask doubling may help the shared encoder exploit missingness structure on the missingness bundle once raw-missing execution is available.
- Upstream delta: Upstream nanoTabPFN binary runs do not benchmark a surfaced feature-mask missingness policy as a separate queue axis.
- Anchor delta: Changes only `model.missingness_mode` to `feature_mask`; the required raw-missing runtime policy travels with the row surface.
- Expected effect: Feature-mask doubling may expose missingness structure directly to the encoder, with risk of widening the effective input surface too aggressively.
- Effective labels: model=`delta_model_missingness_feature_mask`, data=`missing_manifest_default`, preprocessing=`runtime_no_impute`, training=`prior_constant_lr`
- Model overrides: `{'missingness_mode': 'feature_mask'}`
- Parameter adequacy plan:
  - Verify the frozen missingness anchor and this row share the same bundle, manifest, and training recipe.
  - Treat any movement as ambiguous until the missingness-specific anchor/control pair is registered on the new track.
- Adequacy knobs to dimension explicitly:
  - Compare only against a missingness-specific anchor on the same bundle and manifests.
  - Keep the raw-missing runtime policy attached to the row; it is required enablement, not a separate axis.
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - `data.allow_missing_values=true` and `preprocessing.impute_missing=false` are required enablement for non-`none` missingness modes and are intentionally part of this surfaced row.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v3/delta_model_missingness_feature_mask/result_card.md`
- Benchmark metrics: pending

### 3. `delta_model_missingness_explicit_token`

- Dimension family: `model`
- Status: `blocked_on_artifacts`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Switch the model missingness surface from `none` to `explicit_token` while carrying the required raw-missing runtime policy with the row.
- Rationale: Compare explicit-token missingness encoding against the same promoted Shared+LayerNorm backbone on the missingness bundle.
- Hypothesis: An explicit learned missing token may outperform feature-mask doubling if the missingness signal is sparse but semantically informative.
- Upstream delta: Upstream nanoTabPFN binary runs do not benchmark a surfaced explicit-token missingness policy as a separate queue axis.
- Anchor delta: Changes only `model.missingness_mode` to `explicit_token`; the required raw-missing runtime policy travels with the row surface.
- Expected effect: A learned missing token may preserve missingness semantics more compactly than feature-mask doubling, with risk that the token under-communicates sparse patterns.
- Effective labels: model=`delta_model_missingness_explicit_token`, data=`missing_manifest_default`, preprocessing=`runtime_no_impute`, training=`prior_constant_lr`
- Model overrides: `{'missingness_mode': 'explicit_token'}`
- Parameter adequacy plan:
  - Keep bundle, manifest, and optimizer fixed relative to the missingness anchor.
  - Interpret the row jointly with the feature-mask result before preferring one missingness encoding family.
- Adequacy knobs to dimension explicitly:
  - Compare only against a missingness-specific anchor on the same bundle and manifests.
  - Keep the raw-missing runtime policy attached to the row; it is required enablement, not a separate axis.
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - `data.allow_missing_values=true` and `preprocessing.impute_missing=false` are required enablement for non-`none` missingness modes and are intentionally part of this surfaced row.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v3/delta_model_missingness_explicit_token/result_card.md`
- Benchmark metrics: pending

### 4. `delta_preproc_all_nan_fill_nonzero`

- Dimension family: `preprocessing`
- Status: `blocked_on_artifacts`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Keep imputation on but change the all-NaN fallback fill value from 0.0 to 1.0.
- Rationale: Measure whether the all-NaN fallback fill value matters once the draft missingness workstream has a real raw-missing anchor and manifests.
- Hypothesis: The fill-value branch should matter only when the missingness manifests include fully missing columns or groups.
- Upstream delta: Upstream helper logic does not expose this repo-local fill-value contract as a first-class surface.
- Anchor delta: Keeps imputation enabled but changes the all-NaN fallback from `0.0` to `1.0` on the missingness bundle.
- Expected effect: Usually small, but important for understanding whether hidden fill defaults matter on this corpus.
- Effective labels: model=`anchor_model`, data=`missing_manifest_default`, preprocessing=`runtime_all_nan_fill_one`, training=`prior_constant_lr`
- Preprocessing overrides: `{'all_nan_fill': 1.0}`
- Parameter adequacy plan:
  - Report whether the materialized manifests actually contain any all-NaN columns before interpreting a flat result.
  - Compare only after the missingness anchor and control baseline are frozen on the missingness track.
- Adequacy knobs to dimension explicitly:
  - Report whether all-NaN columns are actually present in the compared manifests.
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - This row stays in the missingness workstream even though imputation remains enabled.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v3/delta_preproc_all_nan_fill_nonzero/result_card.md`
- Benchmark metrics: pending

### 5. `delta_data_missing_manifest_forbid_any`

- Dimension family: `data`
- Status: `blocked_on_artifacts`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Rebuild the missingness manifest surface to exclude any dataset containing NaN or Inf while keeping model and preprocessing fixed.
- Rationale: Rebuild the missingness bundle manifests to forbid any NaN/Inf rows while leaving model and preprocessing on the draft anchor surface.
- Hypothesis: Forbidding missing-valued records may simplify optimization but could erase exactly the structure this workstream is meant to measure.
- Upstream delta: Not applicable; this is a repo-local training-manifest policy axis.
- Anchor delta: Changes only the training data manifest policy from `missing_manifest_default` to `missing_manifest_forbid_any`.
- Expected effect: Potentially cleaner optimization, but at the cost of removing the missing-valued structure the workstream is meant to study.
- Effective labels: model=`anchor_model`, data=`missing_manifest_forbid_any`, preprocessing=`runtime_default`, training=`prior_constant_lr`
- Data overrides: `{'source': 'manifest', 'filter_policy': 'include_all', 'missing_value_policy': 'forbid_any', 'allow_missing_values': False}`
- Parameter adequacy plan:
  - Record dataset-count and missingness-distribution shifts versus `missing_manifest_default` before running.
  - Treat any performance gain as ambiguous unless the manifest contraction is explicitly characterized.
- Adequacy knobs to dimension explicitly:
  - Report dataset-count and missingness-distribution shifts versus `missing_manifest_default`.
  - Do not interpret gains without characterizing how much data was removed.
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - This row is the explicit data-policy ablation for removing missing-valued records.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v3/delta_data_missing_manifest_forbid_any/result_card.md`
- Benchmark metrics: pending

### 6. `delta_data_missing_manifest_accepted_only`

- Dimension family: `data`
- Status: `blocked_on_artifacts`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Rebuild the missingness manifest surface under accepted-only filtering while preserving missing-valued records that pass curation.
- Rationale: Rebuild the missingness manifests under accepted-only filtering while preserving missing-valued records that pass curation.
- Hypothesis: Accepted-only curation may improve data quality without erasing the missingness structure needed for the model/preprocessing comparisons.
- Upstream delta: Not applicable; this is a repo-local training-manifest policy axis.
- Anchor delta: Changes only the training data manifest policy from `missing_manifest_default` to `missing_manifest_accepted_only`.
- Expected effect: Cleaner curation may improve signal quality without erasing the missingness structure needed for the model and preprocessing rows.
- Effective labels: model=`anchor_model`, data=`missing_manifest_accepted_only`, preprocessing=`runtime_default`, training=`prior_constant_lr`
- Data overrides: `{'source': 'manifest', 'filter_policy': 'accepted_only', 'missing_value_policy': 'allow_any', 'allow_missing_values': True}`
- Parameter adequacy plan:
  - Record dataset-count, accepted-filter coverage, and missingness-distribution shifts versus `missing_manifest_default` before running.
  - Interpret only after the accepted-only manifest lineage is frozen alongside the missingness anchor.
- Adequacy knobs to dimension explicitly:
  - Report dataset-count, accepted-filter coverage, and missingness-distribution shifts versus `missing_manifest_default`.
  - Keep missing values enabled so the row stays a data-policy comparison, not a hidden preprocessing change.
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - This row keeps missing values but tightens curation so data-quality and missingness effects can be separated later.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v3/delta_data_missing_manifest_accepted_only/result_card.md`
- Benchmark metrics: pending

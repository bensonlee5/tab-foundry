# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/binary_md_v3/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `binary_md_v3`
- Sweep status: `active`
- Parent sweep id: `binary_md_v2`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `01_shared_norm_post_ln_binary_medium_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Comparison policy: `anchor_only`
- Anchor metrics: best ROC AUC `0.7670`, final ROC AUC `0.7643`, final training time `505.3s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Shared feature encoder path with benchmark-external normalization. | Feature encoder swaps change both the representation path and where normalization lives. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Same mean-padded linear target conditioner. | The anchor preserves the upstream label-conditioning mechanism. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Same nano post-norm cell transformer block. | This keeps the strongest structural tie to upstream nanoTabPFN. |
| tokenizer | One scalar token per feature. | Same scalar-per-feature tokenizer. | Tokenization remains aligned with upstream parity. |
| column encoder | None on the upstream direct path. | No column-set encoder on the anchor path. | Column-set modeling remains absent and should not explain anchor behavior. |
| row readout | Target-column readout from the final cell tensor. | Same target-column row pool. | Readout remains on the direct upstream-style path. |
| context encoder | None on the upstream direct path. | None on the anchor path. | Context encoding remains absent; later context rows will change both depth and label-flow semantics. |
| prediction head | Direct binary logits head. | Direct binary logits head. | The prediction head remains on the narrow upstream-style binary path. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_medium` (10 tasks) with data surface label `manifest_default`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `prior_constant_lr`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Legacy stage alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_preproc_impute_missing_off` | missing_values | yes | deferred_separate_workstream | none | Disable runtime mean imputation while keeping the fitted label-remap path intact. | Do not run on the main no-missing campaign; revisit only in a dedicated missingness workstream with explicit missing-valued training and evaluation inputs. |

## Detailed Rows

### 1. `delta_preproc_impute_missing_off`

- Dimension family: `preprocessing`
- Status: `deferred_separate_workstream`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Disable runtime mean imputation while keeping the fitted label-remap path intact.
- Rationale: Keep the missingness-only placeholder attached to the promoted shared+LayerNorm anchor while missingness work proceeds on a separate branch.
- Hypothesis: Missingness-specific conclusions should wait for a dedicated workstream with explicit missing-valued training and evaluation inputs.
- Upstream delta: Upstream notebook preprocessing imputes as part of its benchmark helper path.
- Anchor delta: Changes only preprocessing by disabling runtime mean imputation while keeping the shared+LayerNorm anchor model, data surface, and training recipe fixed.
- Expected effect: Cleaner attribution around imputation usefulness, but possibly brittle feature tensors on some datasets.
- Effective labels: model=`anchor_model`, data=`anchor_manifest_default`, preprocessing=`runtime_no_impute`, training=`prior_constant_lr`
- Preprocessing overrides: `{'impute_missing': False}`
- Parameter adequacy plan:
  - If the result is weak, note whether the benchmark tasks actually contain meaningful missingness on the support side.
- Adequacy knobs to dimension explicitly:
  - Interpret against manifest missingness prevalence where available.
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - Placeholder row only on this branch; missingness implementation lives on a separate branch.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v3/delta_preproc_impute_missing_off/result_card.md`
- Benchmark metrics: pending

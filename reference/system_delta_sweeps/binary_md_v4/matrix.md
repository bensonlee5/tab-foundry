# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/binary_md_v4/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `binary_md_v4`
- Sweep status: `draft`
- Parent sweep id: `binary_md_v3`
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
| 1 | `delta_preproc_norm_zscore` | input_normalization | yes | ready | none | Apply train-only z-score normalization (no clipping) to input features before encoding, changing model.input_normalization from none to train_zscore. | Run and compare against the anchor and other normalization rows. |
| 2 | `delta_preproc_norm_rankgauss` | input_normalization | yes | ready | none | Apply train-only rank-Gauss (quantile-to-normal via erfinv) normalization to input features before encoding, changing model.input_normalization from none to train_rankgauss. | Run and compare against the anchor and other normalization rows. |
| 3 | `delta_preproc_norm_robust` | input_normalization | yes | ready | none | Apply train-only robust (median / IQR) normalization to input features before encoding, changing model.input_normalization from none to train_robust. | Run and compare against the anchor and other normalization rows. |
| 4 | `delta_preproc_norm_winsorize_zscore` | input_normalization | yes | ready | none | Clip input features at the 1st/99th train percentiles, then apply z-score normalization, changing model.input_normalization from none to train_winsorize_zscore. | Run and compare against the anchor and other normalization rows. |
| 5 | `delta_preproc_norm_zscore` | input_normalization | yes | ready | none | Apply train-only z-score normalization (no clipping) to input features before encoding, changing model.input_normalization from none to train_zscore. | Run and compare against the anchor and other normalization rows. |
| 6 | `delta_preproc_norm_rankgauss` | input_normalization | yes | ready | none | Apply train-only rank-Gauss (quantile-to-normal via erfinv) normalization to input features before encoding, changing model.input_normalization from none to train_rankgauss. | Run and compare against the anchor and other normalization rows. |
| 7 | `delta_preproc_norm_robust` | input_normalization | yes | ready | none | Apply train-only robust (median / IQR) normalization to input features before encoding, changing model.input_normalization from none to train_robust. | Run and compare against the anchor and other normalization rows. |
| 8 | `delta_preproc_norm_winsorize_zscore` | input_normalization | yes | ready | none | Clip input features at the 1st/99th train percentiles, then apply z-score normalization, changing model.input_normalization from none to train_winsorize_zscore. | Run and compare against the anchor and other normalization rows. |

## Detailed Rows

### 1. `delta_preproc_norm_zscore`

- Dimension family: `preprocessing`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Apply train-only z-score normalization (no clipping) to input features before encoding, changing model.input_normalization from none to train_zscore.
- Rationale: Contextualize `delta_preproc_norm_zscore` against anchor `01_shared_norm_post_ln_binary_medium_v1` for sweep `binary_md_v4`.
- Hypothesis:
- Upstream delta: Upstream nanoTabPFN applies z-score+clip inside the feature encoder.
- Anchor delta: Changes `input_normalization` from `none` to `train_zscore` while keeping the anchor RMSNorm post-encoder norm.
- Expected effect: May reduce feature-scale sensitivity; the anchor sees raw features, so any improvement indicates the model benefits from pre-normalized inputs.
- Effective labels: model=`delta_preproc_norm_zscore`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Compare ROC AUC and loss curves against the anchor (no normalization).
  - Check whether normalization helps more on high-variance-feature tasks.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v4/delta_preproc_norm_zscore/result_card.md`
- Benchmark metrics: pending

### 2. `delta_preproc_norm_rankgauss`

- Dimension family: `preprocessing`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Apply train-only rank-Gauss (quantile-to-normal via erfinv) normalization to input features before encoding, changing model.input_normalization from none to train_rankgauss.
- Rationale: Contextualize `delta_preproc_norm_rankgauss` against anchor `01_shared_norm_post_ln_binary_medium_v1` for sweep `binary_md_v4`.
- Hypothesis:
- Upstream delta: Upstream nanoTabPFN applies z-score+clip inside the feature encoder.
- Anchor delta: Changes `input_normalization` from `none` to `train_rankgauss` while keeping the anchor RMSNorm post-encoder norm.
- Expected effect: Rank-Gauss is robust to outliers and heavy tails; may help on skewed-feature tasks where z-score under- or over-corrects.
- Effective labels: model=`delta_preproc_norm_rankgauss`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Compare ROC AUC and loss curves against the anchor (no normalization).
  - Look for per-task lift on skewed or heavy-tailed feature distributions.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v4/delta_preproc_norm_rankgauss/result_card.md`
- Benchmark metrics: pending

### 3. `delta_preproc_norm_robust`

- Dimension family: `preprocessing`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Apply train-only robust (median / IQR) normalization to input features before encoding, changing model.input_normalization from none to train_robust.
- Rationale: Contextualize `delta_preproc_norm_robust` against anchor `01_shared_norm_post_ln_binary_medium_v1` for sweep `binary_md_v4`.
- Hypothesis:
- Upstream delta: Upstream nanoTabPFN applies z-score+clip inside the feature encoder.
- Anchor delta: Changes `input_normalization` from `none` to `train_robust` while keeping the anchor RMSNorm post-encoder norm.
- Expected effect: Median/IQR scaling is less sensitive to outliers than z-score; may improve stability on tasks with extreme values without discarding ordinal information.
- Effective labels: model=`delta_preproc_norm_robust`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Compare ROC AUC and loss curves against the anchor (no normalization).
  - Check whether robust scaling helps on tasks with outlier-heavy columns.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v4/delta_preproc_norm_robust/result_card.md`
- Benchmark metrics: pending

### 4. `delta_preproc_norm_winsorize_zscore`

- Dimension family: `preprocessing`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Clip input features at the 1st/99th train percentiles, then apply z-score normalization, changing model.input_normalization from none to train_winsorize_zscore.
- Rationale: Contextualize `delta_preproc_norm_winsorize_zscore` against anchor `01_shared_norm_post_ln_binary_medium_v1` for sweep `binary_md_v4`.
- Hypothesis:
- Upstream delta: Upstream nanoTabPFN applies z-score+clip inside the feature encoder.
- Anchor delta: Changes `input_normalization` from `none` to `train_winsorize_zscore` while keeping the anchor RMSNorm post-encoder norm.
- Expected effect: Winsorization removes extreme outliers before z-scoring, giving a tighter effective range than plain z-score while preserving more ordinal detail than rank-Gauss.
- Effective labels: model=`delta_preproc_norm_winsorize_zscore`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Compare ROC AUC and loss curves against the anchor (no normalization).
  - Verify that winsorization bounds match expected 1st/99th percentile range.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v4/delta_preproc_norm_winsorize_zscore/result_card.md`
- Benchmark metrics: pending

### 5. `delta_preproc_norm_zscore`

- Dimension family: `preprocessing`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Apply train-only z-score normalization (no clipping) to input features before encoding, changing model.input_normalization from none to train_zscore.
- Rationale: Contextualize `delta_preproc_norm_zscore` with LayerNorm post-encoder against anchor `01_shared_norm_post_ln_binary_medium_v1` for sweep `binary_md_v4`.
- Hypothesis:
- Upstream delta: Upstream nanoTabPFN applies z-score+clip inside the feature encoder.
- Anchor delta: Changes input_normalization from train_zscore_clip to train_zscore while keeping the anchor LayerNorm post-encoder norm.
- Expected effect: May reduce feature-scale sensitivity; the anchor sees raw features, so any improvement indicates the model benefits from pre-normalized inputs.
- Effective labels: model=`delta_preproc_norm_zscore`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Compare ROC AUC and loss curves against the anchor (no normalization).
  - Check whether normalization helps more on high-variance-feature tasks.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v4/delta_preproc_norm_zscore/result_card.md`
- Benchmark metrics: pending

### 6. `delta_preproc_norm_rankgauss`

- Dimension family: `preprocessing`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Apply train-only rank-Gauss (quantile-to-normal via erfinv) normalization to input features before encoding, changing model.input_normalization from none to train_rankgauss.
- Rationale: Contextualize `delta_preproc_norm_rankgauss` with LayerNorm post-encoder against anchor `01_shared_norm_post_ln_binary_medium_v1` for sweep `binary_md_v4`.
- Hypothesis:
- Upstream delta: Upstream nanoTabPFN applies z-score+clip inside the feature encoder.
- Anchor delta: Changes input_normalization from train_zscore_clip to train_rankgauss while keeping the anchor LayerNorm post-encoder norm.
- Expected effect: Rank-Gauss is robust to outliers and heavy tails; may help on skewed-feature tasks where z-score under- or over-corrects.
- Effective labels: model=`delta_preproc_norm_rankgauss`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Compare ROC AUC and loss curves against the anchor (no normalization).
  - Look for per-task lift on skewed or heavy-tailed feature distributions.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v4/delta_preproc_norm_rankgauss/result_card.md`
- Benchmark metrics: pending

### 7. `delta_preproc_norm_robust`

- Dimension family: `preprocessing`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Apply train-only robust (median / IQR) normalization to input features before encoding, changing model.input_normalization from none to train_robust.
- Rationale: Contextualize `delta_preproc_norm_robust` with LayerNorm post-encoder against anchor `01_shared_norm_post_ln_binary_medium_v1` for sweep `binary_md_v4`.
- Hypothesis:
- Upstream delta: Upstream nanoTabPFN applies z-score+clip inside the feature encoder.
- Anchor delta: Changes input_normalization from train_zscore_clip to train_robust while keeping the anchor LayerNorm post-encoder norm.
- Expected effect: Median/IQR scaling is less sensitive to outliers than z-score; may improve stability on tasks with extreme values without discarding ordinal information.
- Effective labels: model=`delta_preproc_norm_robust`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Compare ROC AUC and loss curves against the anchor (no normalization).
  - Check whether robust scaling helps on tasks with outlier-heavy columns.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v4/delta_preproc_norm_robust/result_card.md`
- Benchmark metrics: pending

### 8. `delta_preproc_norm_winsorize_zscore`

- Dimension family: `preprocessing`
- Status: `ready`
- Binary applicable: `True`
- Legacy stage alias: `none`
- Description: Clip input features at the 1st/99th train percentiles, then apply z-score normalization, changing model.input_normalization from none to train_winsorize_zscore.
- Rationale: Contextualize `delta_preproc_norm_winsorize_zscore` with LayerNorm post-encoder against anchor `01_shared_norm_post_ln_binary_medium_v1` for sweep `binary_md_v4`.
- Hypothesis:
- Upstream delta: Upstream nanoTabPFN applies z-score+clip inside the feature encoder.
- Anchor delta: Changes input_normalization from train_zscore_clip to train_winsorize_zscore while keeping the anchor LayerNorm post-encoder norm.
- Expected effect: Winsorization removes extreme outliers before z-scoring, giving a tighter effective range than plain z-score while preserving more ordinal detail than rank-Gauss.
- Effective labels: model=`delta_preproc_norm_winsorize_zscore`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`prior_constant_lr`
- Preprocessing overrides: `{}`
- Parameter adequacy plan:
  - Compare ROC AUC and loss curves against the anchor (no normalization).
  - Verify that winsorization bounds match expected 1st/99th percentile range.
- Adequacy knobs to dimension explicitly:
  - model.input_normalization
- Interpretation status: `pending`
- Decision: `None`
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/binary_md_v4/delta_preproc_norm_winsorize_zscore/result_card.md`
- Benchmark metrics: pending

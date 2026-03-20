# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/shared_surface_bridge_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `shared_surface_bridge_v1`
- Sweep status: `completed`
- Parent sweep id: `binary_md_v1`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_shared_surface_bridge_v1_01_delta_architecture_screen_nano_exact_replay_v1`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- Training experiment: `cls_benchmark_staged`
- Training config profile: `cls_benchmark_staged`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.6951`, final Brier score `0.5020`, best ROC AUC `0.4339`, final ROC AUC `0.4339`, final training time `87.3s`

## Anchor Comparison

Upstream reference: `nanoTabPFN` from `https://github.com/automl/nanoTabPFN/blob/main/model.py`.

| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| feature encoder | Scalar feature linear encoder with internal train/test z-score+clip handling. | Same nano feature encoder path with internal benchmark normalization. | Feature encoding remains close to upstream parity; later deltas should be attributed elsewhere. |
| target conditioning | Mean-padded linear target encoder on the direct binary path. | Same mean-padded linear target conditioner. | The anchor preserves the upstream label-conditioning mechanism. |
| cell transformer block | Post-norm nanoTabPFN block with feature attention then row attention. | Same nano post-norm cell transformer block. | This keeps the strongest structural tie to upstream nanoTabPFN. |
| tokenizer | One scalar token per feature. | Same scalar-per-feature tokenizer. | Tokenization remains aligned with upstream parity. |
| column encoder | None on the upstream direct path. | No column-set encoder on the anchor path. | Column-set modeling remains absent and should not explain anchor behavior. |
| row readout | Target-column readout from the final cell tensor. | Same target-column row pool. | Readout remains on the direct upstream-style path. |
| context encoder | None on the upstream direct path. | None on the anchor path. | Context encoding remains absent; later context rows will change both depth and label-flow semantics. |
| prediction head | Direct binary logits head. | Direct binary logits head. | The prediction head remains on the narrow upstream-style binary path. |
| training data surface | OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract. | Benchmark bundle `nanotabpfn_openml_binary_medium` (10 tasks) with data surface label `anchor_manifest_default`. | Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose. |
| preprocessing | Notebook preprocessing inside the benchmark helper. | Benchmark preprocessing surface label `runtime_default`. | Preprocessing changes can alter the effective task definition and must be tracked explicitly. |
| training recipe | No repo-local prior-dump training-surface contract. | Training surface label `training_default`. | Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_architecture_screen_nano_exact_replay` | baseline_replay | yes | completed | nano_exact | Replay the public `nano_exact` stage on `cls_benchmark_staged` before comparing the shared-surface bridge rows. | Run first to establish the architecture-screen replay baseline for the shared-surface bridge. |
| 2 | `delta_architecture_screen_shared_norm` | feature_encoding | yes | completed | shared_norm | Evaluate the public `shared_norm` stage as the first stage-native step off the PFN-only encoder path on `cls_benchmark_staged`. | Compare directly against the architecture-screen `nano_exact` replay before moving deeper into the bridge. |
| 3 | `delta_architecture_screen_prenorm_block` | table_block | yes | completed | prenorm_block | Evaluate the public `prenorm_block` stage as the first stage-native backbone change on the shared architecture-screen surface. | Lock as the grouped-token handoff row and treat later bridge rows as optional follow-ons, not new default predecessors. |
| 4 | `delta_architecture_screen_small_class_head` | head | yes | completed | small_class_head | Evaluate the public `small_class_head` stage as the first shared-surface bridge row with the small-class head contract enabled. | Treat as a binary-lane no-op and keep `prenorm_block` as the grouped-token predecessor. |
| 5 | `delta_architecture_screen_test_self` | table_block | yes | completed | test_self | Evaluate the public `test_self` stage as the last shared-surface bridge row before grouped-token work. | Do not carry forward by default; only revalidate if grouped-token work makes test-self attention explicitly valuable. |

## Detailed Rows

### 1. `delta_architecture_screen_nano_exact_replay`

- Dimension family: `training`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `nano_exact`
- Description: Replay the public `nano_exact` stage on `cls_benchmark_staged` before comparing the shared-surface bridge rows.
- Rationale: Replay the public `nano_exact` stage on `cls_benchmark_staged` so the bridge sweep has a same-lane local comparator before any shared-surface row is judged.
- Hypothesis: The replay should recover the expected benchmark range while emitting the architecture-screen telemetry artifacts missing from the older prior-lane anchor run.
- Upstream delta: Not applicable; this is a repo-local architecture-screen replay of the public `nano_exact` stage.
- Anchor delta: Keep the public `nano_exact` stage fixed and only move execution onto the benchmark-facing architecture-screen lane.
- Expected effect: Establish a telemetry-complete local comparator for the stage-native bridge rows without changing the public model stage.
- Effective labels: model=`delta_architecture_screen_nano_exact_replay`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`training_default`
- Stage-local stability: column (grad `0.0000`, act delta `-0.0000`); row (grad `0.0000`, act delta `-0.0000`)
- Training overrides: `{}`
- Parameter adequacy plan:
  - Run this row first so the bridge sweep has a local `cls_benchmark_staged` comparator with `gradient_history.jsonl` and `telemetry.json`.
  - Compare later bridge rows against this replay rather than treating the older prior-lane anchor as artifact-complete.
- Adequacy knobs to dimension explicitly:
  - Keep data, preprocessing, and training surfaces fixed to the canonical architecture-screen contract.
  - Treat benchmark ROC as primary and use emitted telemetry only to confirm same-lane artifact parity.
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - Treat the older `01_nano_exact_md_prior_parity_fix_binary_medium_v1` registry run as historical anchor evidence only until this replay lands locally.
  - Canonical rerun registered as `sd_shared_surface_bridge_v1_01_delta_architecture_screen_nano_exact_replay_v1`.
  - Established the architecture-screen nano_exact replay anchor for TF-RD-003 shared-surface bridge comparisons.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/shared_surface_bridge_v1/delta_architecture_screen_nano_exact_replay/result_card.md`
- Registered run: `sd_shared_surface_bridge_v1_01_delta_architecture_screen_nano_exact_replay_v1` with final log loss `0.6951`, delta final log loss `+0.0000`, final Brier score `0.5020`, delta final Brier score `+0.0000`, best ROC AUC `0.4339`, final ROC AUC `0.4339`, final-minus-best `+0.0000`, delta final ROC AUC `+0.0000`, delta drift `+0.0000`, delta final training time `+0.0s`

### 2. `delta_architecture_screen_shared_norm`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `shared_norm`
- Description: Evaluate the public `shared_norm` stage as the first stage-native step off the PFN-only encoder path on `cls_benchmark_staged`.
- Rationale: Make `shared_norm` the first explicit step off the PFN-only encoder path on the benchmark-facing architecture-screen surface.
- Hypothesis: The public shared encoder should keep the bridge finite and interpretable enough to serve as the first mandatory shared-surface comparator.
- Upstream delta: Upstream nanoTabPFN does not expose the public shared-surface bridge row represented by `shared_norm`.
- Anchor delta: Starting from the architecture-screen `nano_exact` replay, switch only to the public `shared_norm` stage.
- Expected effect: Move feature handling onto the shared staged surface while keeping scalar-per-feature tokenization and the post-norm table block fixed.
- Effective labels: model=`delta_architecture_screen_shared_norm`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`training_default`
- Stage-local stability: column (grad `0.0000`, act delta `+0.0001`); row (grad `0.0000`, act delta `+0.0001`)
- Model overrides: `{'stage': 'shared_norm'}`
- Parameter adequacy plan:
  - Treat this as the first mandatory shared-surface comparator, not as a tokenizer or row-pool experiment.
  - If weak, keep the result as bridge evidence and do not skip straight to later shared-surface rows without recording the regression.
- Adequacy knobs to dimension explicitly:
  - Compare directly against the architecture-screen `nano_exact` replay before attributing later bridge changes.
  - Use activation and gradient telemetry to confirm whether the shared encoder remains stable enough to serve as the first bridge step.
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Historical hybrid-diagnostic `delta_shared_feature_norm` evidence remains context only; this row is the stage-native benchmark-facing shared-surface read.
  - Canonical rerun registered as `sd_shared_surface_bridge_v1_02_delta_architecture_screen_shared_norm_v1`.
  - Recorded the first stage-native shared-surface bridge read against the architecture-screen nano_exact replay anchor.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/shared_surface_bridge_v1/delta_architecture_screen_shared_norm/result_card.md`
- Registered run: `sd_shared_surface_bridge_v1_02_delta_architecture_screen_shared_norm_v1` with final log loss `0.6680`, delta final log loss `-0.0271`, final Brier score `0.4750`, delta final Brier score `-0.0270`, best ROC AUC `0.4928`, final ROC AUC `0.4928`, final-minus-best `+0.0000`, delta final ROC AUC `+0.0589`, delta drift `+0.0000`, delta final training time `+19.5s`

### 3. `delta_architecture_screen_prenorm_block`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `prenorm_block`
- Description: Evaluate the public `prenorm_block` stage as the first stage-native backbone change on the shared architecture-screen surface.
- Rationale: Test whether the public `prenorm_block` stage is the first viable shared-surface backbone change once the bridge is already off the PFN-only encoder.
- Hypothesis: The stage-native pre-norm block should be more informative than the earlier hybrid-diagnostic prenorm evidence because it runs on the public shared surface rather than on `nano_exact` with bounded overrides.
- Upstream delta: Upstream nanoTabPFN does not expose the public staged pre-norm bridge row represented by `prenorm_block`.
- Anchor delta: Starting from the stage-native `shared_norm` bridge row, switch only to the public `prenorm_block` stage.
- Expected effect: Keep the shared encoder fixed while switching the main table block to the staged pre-norm implementation.
- Effective labels: model=`delta_architecture_screen_prenorm_block`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`training_default`
- Stage-local stability: column (grad `0.0000`, act delta `-0.0066`); row (grad `0.0000`, act delta `-0.0434`)
- Model overrides: `{'stage': 'prenorm_block'}`
- Parameter adequacy plan:
  - Treat this row as the first backbone change after the shared encoder is active.
  - If mixed, preserve it as explicit bridge evidence rather than collapsing the result into the older hybrid-diagnostic prenorm evidence.
- Adequacy knobs to dimension explicitly:
  - Compare against both the architecture-screen `nano_exact` replay and the public `shared_norm` row.
  - Use final-vs-best benchmark behavior plus stage-local telemetry to separate outright regressions from bridge instability.
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `keep`
- Notes:
  - Do not collapse this row back into the older `delta_prenorm_block` conclusion; this is the first benchmark-facing stage-native prenorm read.
  - Canonical rerun registered as `sd_shared_surface_bridge_v1_03_delta_architecture_screen_prenorm_block_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Locked as the grouped-token handoff row because it delivered the first material bridge gain while preserving the strongest proper-score result among rows 3-5.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/shared_surface_bridge_v1/delta_architecture_screen_prenorm_block/result_card.md`
- Registered run: `sd_shared_surface_bridge_v1_03_delta_architecture_screen_prenorm_block_v1` with final log loss `0.6538`, delta final log loss `-0.0413`, final Brier score `0.4610`, delta final Brier score `-0.0410`, best ROC AUC `0.5479`, final ROC AUC `0.5479`, final-minus-best `+0.0000`, delta final ROC AUC `+0.1140`, delta drift `+0.0000`, delta final training time `-8.1s`

### 4. `delta_architecture_screen_small_class_head`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `small_class_head`
- Description: Evaluate the public `small_class_head` stage as the first shared-surface bridge row with the small-class head contract enabled.
- Rationale: Evaluate whether `small_class_head` should become the first explicit grouped-token predecessor row on the public shared-surface bridge.
- Hypothesis: The small-class head may improve bridge coherence enough that later grouped-token work should inherit it instead of stopping at the pre-norm block.
- Upstream delta: Upstream nanoTabPFN does not expose the public staged small-class bridge row represented by `small_class_head`.
- Anchor delta: Starting from the public `prenorm_block` bridge row, switch only to the public `small_class_head` stage.
- Expected effect: Keep the shared encoder and pre-norm block fixed while promoting the head to the small-class contract that later grouped-token work should inherit from.
- Effective labels: model=`delta_architecture_screen_small_class_head`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`training_default`
- Stage-local stability: column (grad `0.0000`, act delta `-0.0066`); row (grad `0.0000`, act delta `-0.0434`)
- Model overrides: `{'stage': 'small_class_head'}`
- Parameter adequacy plan:
  - Treat this as a bridge row, not as a many-class promotion or tokenization change.
  - Use it to decide whether later grouped-token work should inherit the small-class head rather than stopping at the pre-norm block.
- Adequacy knobs to dimension explicitly:
  - Compare against `prenorm_block` before attributing any effect to grouped-token readiness.
  - Watch whether head-contract changes are neutral, stabilizing, or clearly harmful on the fixed binary architecture-screen surface.
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row exists to make the grouped-token handoff explicit; do not interpret it as many-class promotion work.
  - Canonical rerun registered as `sd_shared_surface_bridge_v1_04_delta_architecture_screen_small_class_head_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Numerically matched `prenorm_block` on this binary architecture-screen lane and did not justify a distinct predecessor row.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/shared_surface_bridge_v1/delta_architecture_screen_small_class_head/result_card.md`
- Registered run: `sd_shared_surface_bridge_v1_04_delta_architecture_screen_small_class_head_v1` with final log loss `0.6538`, delta final log loss `-0.0413`, final Brier score `0.4610`, delta final Brier score `-0.0410`, best ROC AUC `0.5479`, final ROC AUC `0.5479`, final-minus-best `+0.0000`, delta final ROC AUC `+0.1140`, delta drift `+0.0000`, delta final training time `-16.3s`

### 5. `delta_architecture_screen_test_self`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `test_self`
- Description: Evaluate the public `test_self` stage as the last shared-surface bridge row before grouped-token work.
- Rationale: Evaluate whether `test_self` should displace `small_class_head` as the single grouped-token predecessor on the public shared-surface bridge.
- Hypothesis: Test-self attention may repair the late bridge rows enough to justify locking `test_self` as the grouped-token handoff row, but the effect may remain only locally beneficial.
- Upstream delta: Upstream nanoTabPFN does not expose the public staged test-self bridge row represented by `test_self`.
- Anchor delta: Starting from the public `small_class_head` bridge row, switch only to the public `test_self` stage.
- Expected effect: Keep the shared bridge surface fixed while enabling the public test-self attention setting that later grouped-token work may inherit.
- Effective labels: model=`delta_architecture_screen_test_self`, data=`anchor_manifest_default`, preprocessing=`runtime_default`, training=`training_default`
- Stage-local stability: column (grad `0.0000`, act delta `-0.0075`); row (grad `0.0000`, act delta `-0.0461`)
- Model overrides: `{'stage': 'test_self'}`
- Parameter adequacy plan:
  - Treat this as the final shared-surface bridge candidate before grouped tokens, not as tokenizer work.
  - Pick between `small_class_head` and `test_self` only after the full five-row bridge sweep is visible together.
- Adequacy knobs to dimension explicitly:
  - Compare against both `prenorm_block` and `small_class_head` before concluding that test-self attention should be the grouped-token predecessor.
  - Use stage-local telemetry to distinguish a local repair from a genuinely stronger bridge handoff row.
- Execution policy: `benchmark_full`
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - Earlier hybrid-diagnostic `delta_test_self_attention` evidence is historical context only; the keep/defer decision here must come from the stage-native bridge.
  - Canonical rerun registered as `sd_shared_surface_bridge_v1_05_delta_architecture_screen_test_self_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
  - Improved ROC AUC only marginally over `prenorm_block` while slightly worsening log loss and Brier score, so it was not selected as the default handoff row.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/shared_surface_bridge_v1/delta_architecture_screen_test_self/result_card.md`
- Registered run: `sd_shared_surface_bridge_v1_05_delta_architecture_screen_test_self_v1` with final log loss `0.6545`, delta final log loss `-0.0406`, final Brier score `0.4617`, delta final Brier score `-0.0403`, best ROC AUC `0.5493`, final ROC AUC `0.5493`, final-minus-best `+0.0000`, delta final ROC AUC `+0.1154`, delta drift `+0.0000`, delta final training time `-18.1s`

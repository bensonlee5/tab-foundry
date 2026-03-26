# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_021a_sandwich_nanotabpfn_screen_v1/queue.yaml` plus `reference/system_delta_catalog.yaml` and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_021a_sandwich_nanotabpfn_screen_v1`
- Sweep status: `draft`
- Parent sweep id: `None`
- Complexity level: `binary_md`

## Locked Surface

- Anchor run id: `sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
- Control baseline id: `cls_benchmark_linear_v2`
- External benchmarks: `nanotabpfn`
- Training experiment: `cls_benchmark_sandwich_prior`
- Training config profile: `cls_benchmark_sandwich_prior`
- Surface role: `architecture_screen`
- Comparison policy: `anchor_only`
- Anchor metrics: final log loss `0.3972`, final Brier score `0.2615`, best ROC AUC `0.7634`, final ROC AUC `0.7634`, final training time `257.5s`

## Anchor Comparison

Upstream reference: `Perceiver` from `https://proceedings.mlr.press/v139/jaegle21a.html`.

| Dimension | Upstream Perceiver | Locked anchor | Interpretation |
| --- | --- | --- | --- |
| repeated latent bottleneck | Perceiver uses one fixed learned latent array that repeatedly reads the input array through cross-attention and processes the latent state through latent transformer blocks. | The locked staged anchor has no separate fixed latent bank; it is a row-cls prior surface carried forward as the incumbent benchmark comparator. | This sweep asks first whether the sandwich latent bank is viable at all on the benchmark prior surface before harder-surface work opens. |
| input stream | Perceiver repeatedly reads one shared input array. | The staged anchor uses scalar-per-feature tokens with row-cls pooling rather than an `R + C` summary stream. | The sandwich `R + C` repeated-input stream is the structural architecture change under test relative to the incumbent staged anchor. |
| first capacity axis | Perceiver scaling papers make latent count an explicit memory and fidelity knob. | No latent-count axis exists on the staged anchor; the comparable local discriminator is the sandwich replay row. | Rows `02` and `03` isolate `sandwich_latents` first, with width held fixed. |
| second capacity axis | Perceiver scaling can also move latent-channel width once the basic latent budget is settled. | The staged anchor is not the width reference here; the sandwich replay and high-latent rows provide the local context. | Rows `04` and `05` stay blocked until the latent-only rows show whether widening from `d_icl=96` is justified. |
| training surface | No repo-local prior-dump benchmark recipe is defined by the Perceiver paper. | The locked benchmark recipe is `prior_linear_warmup_decay` with `prior_dump_batch_size=64`, sqrt LR scaling, and the nanoTabPFN medium bundle. | Keep the training surface frozen so the queue reads model capacity rather than optimizer or runtime movement. |
| scope guardrail | Not applicable. | The incumbent staged anchor remains the decision comparator while sandwich evidence is partial. | This sweep is a bounded nanoTabPFN screen only; it does not settle final promotion, dagzoo confirmation, or runtime policy. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_021a_sandwich_replay_v1` | architecture_capacity | yes | ready | none | Replay the current repeated-input `tabfoundry_sandwich` prior-benchmark baseline before reading latent-count movement. | Run first as the sandwich local replay, then interpret rows `02` and `03` relative to it before unblocking any width row. |
| 2 | `delta_tf_rd_021a_sandwich_latents24_v1` | architecture_capacity | yes | ready | none | Halve the sandwich latent count from 48 to 24 while keeping width and repeated-stage depth fixed. | Run second as the lower latent-count discriminator against row `01`. |
| 3 | `delta_tf_rd_021a_sandwich_latents96_v1` | architecture_capacity | yes | ready | none | Double the sandwich latent count from 48 to 96 while keeping width and repeated-stage depth fixed. | Run third as the high-latent discriminator, then decide whether row `05` should be unblocked. |
| 4 | `delta_tf_rd_021a_sandwich_width128_latents48_v1` | architecture_capacity | yes | blocked_on_latent_screen | none | Increase sandwich width from 96 to 128 while keeping the replay latent count fixed at 48. | Leave blocked until the latent-only screen says the replay latent count is still live and width-limited. |
| 5 | `delta_tf_rd_021a_sandwich_width128_latents96_v1` | architecture_capacity | yes | blocked_on_latent_screen | none | Increase sandwich width from 96 to 128 around the upper latent candidate at 96 latents. | Leave blocked until row `03` earns a width follow-up. |

## Detailed Rows

### 1. `delta_tf_rd_021a_sandwich_replay_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Replay the current repeated-input `tabfoundry_sandwich` prior-benchmark baseline before reading latent-count movement.
- Rationale: Reproduce the current repeated-input sandwich baseline on the locked batch64-sqrt prior surface so [#179](https://github.com/bensonlee5/tab-foundry/issues/179) has one explicit local comparator before the latent-count rows are interpreted.
- Hypothesis: If the current sandwich architecture is viable at all on the fast nanoTabPFN surface, this row should train cleanly and land within decision-relevant distance of the staged prior anchor.
- Upstream delta: Perceiver-style latent bottlenecks motivate treating latent count as an explicit capacity knob, but this replay row is repo-local.
- Anchor delta: Keep the nanoTabPFN medium bundle, `prior_linear_warmup_decay` batch64-sqrt recipe, and small-class classification contract fixed, then swap the staged row-cls prior anchor for the current repeated-input sandwich baseline.
- Expected effect: Establish one local sandwich benchmark reference on the locked batch64-sqrt prior surface before latent-only and width follow-up rows are interpreted.
- Effective labels: model=`cls_benchmark_sandwich_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 128, 'norm_type': 'layernorm', 'sandwich_latents': 48, 'sandwich_layers': 8, 'sandwich_heads': 8, 'sandwich_ff_expansion': 2}`
- Parameter adequacy plan:
  - Run this row first so the sandwich screen has one explicit local baseline on the benchmark prior surface.
  - Compare all later sandwich rows against both this replay and the staged prior anchor.
  - Keep the repeated-input architecture fixed while reading latent count and width.
- Adequacy knobs to dimension explicitly:
  - Keep the prior batch, LR scaling, and benchmark bundle fixed to the locked prior anchor surface.
  - Treat this row as the sandwich local reference for later rows, not as a promotion claim.
  - Rank rows by final log loss first, final Brier score second, final ROC AUC third, and runtime fourth.
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - This row is the local sandwich reference for the queue, not a promotion claim.
  - If this row is clearly non-viable, keep the result as explicit evidence instead of skipping straight to wider rows.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021a_sandwich_nanotabpfn_screen_v1/delta_tf_rd_021a_sandwich_replay_v1/result_card.md`
- Benchmark metrics: pending

### 2. `delta_tf_rd_021a_sandwich_latents24_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Halve the sandwich latent count from 48 to 24 while keeping width and repeated-stage depth fixed.
- Rationale: Read the lower latent bracket first so the initial screen can tell whether the sandwich replay is over-provisioned before width becomes a live axis.
- Hypothesis: If the current sandwich replay is carrying more latent memory than the prior benchmark needs, halving `sandwich_latents` to `24` may keep most of the quality while softening cost.
- Upstream delta: Perceiver-style models often trade latent count against compute and fidelity; this row is the lower-capacity bracket for the repo-local sandwich screen.
- Anchor delta: Keep row `01` fixed and change only `sandwich_latents` from `48` to `24`.
- Expected effect: Fewer latents may cut training cost if the repeated-input sandwich is overprovisioned, but could reduce fit on the prior benchmark.
- Effective labels: model=`cls_benchmark_sandwich_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 128, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 8, 'sandwich_heads': 8, 'sandwich_ff_expansion': 2}`
- Parameter adequacy plan:
  - Run after the sandwich replay as the lower latent-count bracket.
  - Use this row and the high-latent row to decide whether width follow-up should stay near the replay or move upward.
- Adequacy knobs to dimension explicitly:
  - Compare directly against the sandwich replay row before opening any width change.
  - Treat this as the lower latent bracket, not as a final small-model promotion claim.
  - Prefer this row only if the quality trade is clearly acceptable relative to the replay.
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Treat this as the lower latent bracket only; do not interpret it as a final compact-model claim.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021a_sandwich_nanotabpfn_screen_v1/delta_tf_rd_021a_sandwich_latents24_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_021a_sandwich_latents96_v1`

- Dimension family: `model`
- Status: `ready`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Double the sandwich latent count from 48 to 96 while keeping width and repeated-stage depth fixed.
- Rationale: Read the upper latent bracket on the same fixed-width sandwich surface before any width follow-up opens.
- Hypothesis: If the repeated-input sandwich is memory-bottlenecked rather than width-bottlenecked on this prior surface, doubling `sandwich_latents` to `96` should improve fit enough to justify the blocked width follow-up around the high-latent row.
- Upstream delta: Perceiver-style models often improve fidelity by enlarging the latent bank; this row is the upper latent bracket for the repo-local sandwich screen.
- Anchor delta: Keep row `01` fixed and change only `sandwich_latents` from `48` to `96`.
- Expected effect: More latents may improve fit if the current sandwich replay is memory-bottlenecked, though runtime and VRAM will increase.
- Effective labels: model=`cls_benchmark_sandwich_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 128, 'norm_type': 'layernorm', 'sandwich_latents': 96, 'sandwich_layers': 8, 'sandwich_heads': 8, 'sandwich_ff_expansion': 2}`
- Parameter adequacy plan:
  - Run after the replay and the low-latent row to bracket the first latent-count read.
  - If this row is clearly best, use it as the upper candidate for the blocked width follow-up.
- Adequacy knobs to dimension explicitly:
  - Compare directly against the sandwich replay before opening any width change.
  - Use runtime and memory only as guardrails; benchmark quality remains primary.
  - Treat this as the upper latent bracket for deciding the width follow-up starting point.
- Execution policy: `benchmark_full`
- Interpretation status: `pending`
- Decision: `None`
- Notes:
  - Use runtime and any local VRAM observation as guardrails only; this row exists to read quality first.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021a_sandwich_nanotabpfn_screen_v1/delta_tf_rd_021a_sandwich_latents96_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_021a_sandwich_width128_latents48_v1`

- Dimension family: `model`
- Status: `blocked_on_latent_screen`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Increase sandwich width from 96 to 128 while keeping the replay latent count fixed at 48.
- Rationale: Pre-author the baseline-width follow-up now, but keep it blocked until rows `01` through `03` say that widening the replay latent line is worth the extra budget.
- Hypothesis: If the replay latent count remains live after the first screen but still looks representation-limited, widening `d_icl` to `128` may recover quality without needing the larger latent bank.
- Upstream delta: Perceiver-style width changes alter latent-channel capacity and attention projection width; this is the lower width follow-up row once the latent screen is visible.
- Anchor delta: Keep row `01` fixed and change only `d_icl` from `96` to `128`.
- Expected effect: More width may improve fit if the replay is width-bottlenecked, though it increases training cost and should not be interpreted before the latent screen.
- Effective labels: model=`cls_benchmark_sandwich_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 128, 'norm_type': 'layernorm', 'sandwich_latents': 48, 'sandwich_layers': 8, 'sandwich_heads': 8, 'sandwich_ff_expansion': 2}`
- Parameter adequacy plan:
  - Unblock only after rows `01` through `03` establish whether the replay latent count remains a live candidate.
  - If unblocked, compare directly against the replay row before treating width as a needed axis.
- Adequacy knobs to dimension explicitly:
  - Keep this row blocked until the latent-only screen confirms that a width read is warranted.
  - Compare this row against the replay row and the high-latent row once unblocked.
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - Do not execute this row in the first pass.
  - If row `01` is clearly non-viable, leave this row blocked rather than escalating width as a rescue attempt.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021a_sandwich_nanotabpfn_screen_v1/delta_tf_rd_021a_sandwich_width128_latents48_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_021a_sandwich_width128_latents96_v1`

- Dimension family: `model`
- Status: `blocked_on_latent_screen`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Increase sandwich width from 96 to 128 around the upper latent candidate at 96 latents.
- Rationale: Pre-author the high-latent width follow-up now, but keep it blocked until row `03` shows that the upper latent bracket is worth widening.
- Hypothesis: If the `96`-latent row is viable but still width-limited, the joint `d_icl=128` plus `sandwich_latents=96` setting may become the first serious second-pass size probe.
- Upstream delta: Perceiver-style scaling often couples latent count and channel width; this row is the blocked width follow-up around the high-latent bracket.
- Anchor delta: Keep row `03` fixed and change only `d_icl` from `96` to `128`.
- Expected effect: Jointly larger width and latent count may improve fit if the sandwich screen is constrained on both bottlenecks, but it is intentionally deferred until the latent screen lands.
- Effective labels: model=`cls_benchmark_sandwich_prior`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 128, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 128, 'norm_type': 'layernorm', 'sandwich_latents': 96, 'sandwich_layers': 8, 'sandwich_heads': 8, 'sandwich_ff_expansion': 2}`
- Parameter adequacy plan:
  - Unblock only if the high-latent row is viable enough to justify a width follow-up.
  - Treat this as a bounded second-pass size probe rather than a new architecture fork.
- Adequacy knobs to dimension explicitly:
  - Keep this row blocked until the latent-only screen shows that the 96-latent candidate remains live.
  - Compare against both the 96-latent row and the replay row once unblocked.
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - This row is the only pre-authored width follow-up around the high-latent bracket.
  - Do not unblock unless row `03` is both viable and directionally stronger than row `01`.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021a_sandwich_nanotabpfn_screen_v1/delta_tf_rd_021a_sandwich_width128_latents96_v1/result_card.md`
- Benchmark metrics: pending

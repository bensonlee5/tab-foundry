# System Delta Matrix

This file is rendered from `reference/system_delta_sweeps/tf_rd_021a_sandwich_openml_screen_v1/resolved_queue.yaml` (derived from `reference/system_delta_sweeps/tf_rd_021a_sandwich_openml_screen_v1/queue.yaml` plus `reference/system_delta_catalog.yaml`) and the canonical benchmark registry.

## Sweep

- Sweep id: `tf_rd_021a_sandwich_openml_screen_v1`
- Sweep status: `completed`
- Parent sweep id: `None`
- Complexity level: `binary_md`
- Resolved queue path: `reference/system_delta_sweeps/tf_rd_021a_sandwich_openml_screen_v1/resolved_queue.yaml`
- Resolved queue inputs fingerprint: `093d974072eba8ad6c372c1781d4d853898f33b37b23a0cf16f4efe0c09f7e4d`

## Locked Surface

- Anchor run id: `sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2`
- Benchmark bundle: `src/tab_foundry/bench/openml_binary_medium_v1.json`
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
| first capacity axis | Perceiver scaling papers make latent count an explicit memory and fidelity knob. | No latent-count axis exists on the staged anchor; the comparable local discriminator is the sandwich replay row. | Rows `02` and `03` were pre-authored as latent-only follow-ups, but the sweep now closes before running them because row `01` made the architecture boundary look more important than latent count. |
| second capacity axis | Perceiver scaling can also move latent-channel width once the basic latent budget is settled. | The staged anchor is not the width reference here; the sandwich replay and high-latent rows provide the local context. | Rows `04` and `05` remain deferred backlog evidence rather than active width probes because the initial replay did not justify spending more budget on the same summary-bottleneck topology. |
| training surface | No repo-local prior-dump benchmark recipe is defined by the Perceiver paper. | The locked benchmark recipe is `prior_linear_warmup_decay` with `prior_dump_batch_size=64`, sqrt LR scaling, and the nanoTabPFN medium bundle. | Keep the training surface frozen so the queue reads model capacity rather than optimizer or runtime movement. |
| scope guardrail | Not applicable. | The incumbent staged anchor remains the decision comparator while sandwich evidence is partial. | This sweep is a bounded nanoTabPFN screen only; it does not settle final promotion, dagzoo confirmation, or runtime policy. |

## Queue Summary

| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `delta_tf_rd_021a_sandwich_replay_v1` | architecture_capacity | yes | completed | none | Replay the current repeated-input `tabfoundry_sandwich` prior-benchmark baseline before reading latent-count movement. | Close TF-RD-021A with this row as explicit negative evidence for the summary-bottleneck sandwich, then hand successor architecture work to TF-RD-021B under [#178](https://github.com/bensonlee5/tab-foundry/issues/178). |
| 2 | `delta_tf_rd_021a_sandwich_latents24_v1` | architecture_capacity | yes | deferred_separate_workstream | none | Halve the sandwich latent count from 48 to 24 while keeping width and repeated-stage depth fixed. | Do not run inside TF-RD-021A. Revisit latent-count movement only if the TF-RD-021B successor replay becomes viable enough to justify a fresh capacity ladder. |
| 3 | `delta_tf_rd_021a_sandwich_latents96_v1` | architecture_capacity | yes | deferred_separate_workstream | none | Double the sandwich latent count from 48 to 96 while keeping width and repeated-stage depth fixed. | Do not run inside TF-RD-021A. Revisit high-latent capacity only after TF-RD-021B establishes a stronger successor replay on the same locked prior surface. |
| 4 | `delta_tf_rd_021a_sandwich_width128_latents48_v1` | architecture_capacity | yes | deferred_separate_workstream | none | Increase sandwich width from 96 to 128 while keeping the replay latent count fixed at 48. | Do not execute in TF-RD-021A. Consider width movement only on the successor architecture if a fresh replay shows the topology is viable but still width-limited. |
| 5 | `delta_tf_rd_021a_sandwich_width128_latents96_v1` | architecture_capacity | yes | deferred_separate_workstream | none | Increase sandwich width from 96 to 128 around the upper latent candidate at 96 latents. | Do not execute in TF-RD-021A. Consider this only if the TF-RD-021B successor replay wins enough evidence to justify a new capacity screen. |

## Detailed Rows

### 1. `delta_tf_rd_021a_sandwich_replay_v1`

- Dimension family: `model`
- Status: `completed`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Replay the current repeated-input `tabfoundry_sandwich` prior-benchmark baseline before reading latent-count movement.
- Rationale: Reproduce the current repeated-input sandwich baseline on the locked batch64-sqrt prior surface so [#179](https://github.com/bensonlee5/tab-foundry/issues/179) has one explicit local comparator before the latent-count rows are interpreted.
- Hypothesis: If the current sandwich architecture is viable at all on the fast nanoTabPFN surface, this row should train cleanly and land within decision-relevant distance of the staged prior anchor.
- Upstream delta: Perceiver-style latent bottlenecks motivate treating latent count as an explicit capacity knob, but this replay row is repo-local.
- Anchor delta: Keep the nanoTabPFN medium bundle, `prior_linear_warmup_decay` batch64-sqrt recipe, and small-class classification contract fixed, then swap the staged row-cls prior anchor for the current repeated-input sandwich baseline.
- Expected effect: Establish one local sandwich benchmark reference on the locked batch64-sqrt prior surface before latent-only and width follow-up rows are interpreted.
- Effective labels: model=`tabfoundry_sandwich`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Resolved surface fingerprint: `00ecef9edd3468c699f82dedf6a9b4b4048625b226723313c548f88ca7fa28a9`
- Resolved runtime surface: `{'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': False}`
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
- Interpretation status: `completed`
- Decision: `defer`
- Notes:
  - This row is the local sandwich reference for the queue, not a promotion claim.
  - This row trained stably, but it underperformed the locked staged anchor badly enough that the latent and width ladder is no longer the preferred next discriminator.
  - Canonical rerun registered as `sd_tf_rd_021a_sandwich_nanotabpfn_screen_v1_01_delta_tf_rd_021a_sandwich_replay_v1_v1`.
  - Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021a_sandwich_openml_screen_v1/delta_tf_rd_021a_sandwich_replay_v1/result_card.md`
- Registered run: `sd_tf_rd_021a_sandwich_nanotabpfn_screen_v1_01_delta_tf_rd_021a_sandwich_replay_v1_v1` with final log loss `0.5549`, delta final log loss `+0.1577`, final Brier score `0.3774`, delta final Brier score `+0.1159`, best ROC AUC `0.6056`, final ROC AUC `0.6224`, final-minus-best `+0.0169`, delta final ROC AUC `-0.1410`, delta drift `+0.0169`, delta final training time `+353.5s`

### 2. `delta_tf_rd_021a_sandwich_latents24_v1`

- Dimension family: `model`
- Status: `deferred_separate_workstream`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Halve the sandwich latent count from 48 to 24 while keeping width and repeated-stage depth fixed.
- Rationale: Keep the lower-latent bracket as backlog evidence only because row `01` showed the summary-bottleneck architecture is the limiting factor, not an obviously under-tuned latent budget.
- Hypothesis: If the current sandwich replay is carrying more latent memory than the prior benchmark needs, halving `sandwich_latents` to `24` may keep most of the quality while softening cost.
- Upstream delta: Perceiver-style models often trade latent count against compute and fidelity; this row is the lower-capacity bracket for the repo-local sandwich screen.
- Anchor delta: Keep row `01` fixed and change only `sandwich_latents` from `48` to `24`.
- Expected effect: Fewer latents may cut training cost if the repeated-input sandwich is overprovisioned, but could reduce fit on the prior benchmark.
- Effective labels: model=`tabfoundry_sandwich`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Resolved surface fingerprint: `64571f218c789b64db3c30037a91fffafde965726a226f7b70bdcfe2bc940277`
- Resolved runtime surface: `{'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': False}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 128, 'norm_type': 'layernorm', 'sandwich_latents': 24, 'sandwich_layers': 8, 'sandwich_heads': 8, 'sandwich_ff_expansion': 2}`
- Parameter adequacy plan:
  - Run after the sandwich replay as the lower latent-count bracket.
  - Use this row and the high-latent row to decide whether width follow-up should stay near the replay or move upward.
- Adequacy knobs to dimension explicitly:
  - Compare directly against the sandwich replay row before opening any width change.
  - Treat this as the lower latent bracket, not as a final small-model promotion claim.
  - Prefer this row only if the quality trade is clearly acceptable relative to the replay.
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - Treat this as abandoned first-pass backlog evidence, not as a live compact-model discriminator.
  - The canonical row `01` replay closed the sweep before latent-count follow-up because the architecture boundary looked more limiting than latent count.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021a_sandwich_openml_screen_v1/delta_tf_rd_021a_sandwich_latents24_v1/result_card.md`
- Benchmark metrics: pending

### 3. `delta_tf_rd_021a_sandwich_latents96_v1`

- Dimension family: `model`
- Status: `deferred_separate_workstream`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Double the sandwich latent count from 48 to 96 while keeping width and repeated-stage depth fixed.
- Rationale: Keep the upper-latent bracket as backlog evidence only because row `01` did not justify spending more budget on the same summary-bottleneck topology.
- Hypothesis: If the repeated-input sandwich is memory-bottlenecked rather than width-bottlenecked on this prior surface, doubling `sandwich_latents` to `96` should improve fit enough to justify the blocked width follow-up around the high-latent row.
- Upstream delta: Perceiver-style models often improve fidelity by enlarging the latent bank; this row is the upper latent bracket for the repo-local sandwich screen.
- Anchor delta: Keep row `01` fixed and change only `sandwich_latents` from `48` to `96`.
- Expected effect: More latents may improve fit if the current sandwich replay is memory-bottlenecked, though runtime and VRAM will increase.
- Effective labels: model=`tabfoundry_sandwich`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Resolved surface fingerprint: `e9a9597dddcdc832e886a4b45f89036eaca8b7349c16d1eef5d549866c3089d4`
- Resolved runtime surface: `{'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': False}`
- Model overrides: `{'arch': 'tabfoundry_sandwich', 'd_icl': 96, 'input_normalization': 'train_zscore_clip', 'many_class_base': 2, 'head_hidden_dim': 128, 'norm_type': 'layernorm', 'sandwich_latents': 96, 'sandwich_layers': 8, 'sandwich_heads': 8, 'sandwich_ff_expansion': 2}`
- Parameter adequacy plan:
  - Run after the replay and the low-latent row to bracket the first latent-count read.
  - If this row is clearly best, use it as the upper candidate for the blocked width follow-up.
- Adequacy knobs to dimension explicitly:
  - Compare directly against the sandwich replay before opening any width change.
  - Use runtime and memory only as guardrails; benchmark quality remains primary.
  - Treat this as the upper latent bracket for deciding the width follow-up starting point.
- Execution policy: `benchmark_full`
- Interpretation status: `blocked`
- Decision: `None`
- Notes:
  - Treat this as abandoned first-pass backlog evidence, not as an active width-follow-up gate.
  - Use runtime and any later local VRAM observation only as guardrails if this row is ever revived under successor-architecture work.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021a_sandwich_openml_screen_v1/delta_tf_rd_021a_sandwich_latents96_v1/result_card.md`
- Benchmark metrics: pending

### 4. `delta_tf_rd_021a_sandwich_width128_latents48_v1`

- Dimension family: `model`
- Status: `deferred_separate_workstream`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Increase sandwich width from 96 to 128 while keeping the replay latent count fixed at 48.
- Rationale: Keep the baseline-width follow-up as backlog evidence only because the completed replay row did not justify widening the same summary-bottleneck architecture.
- Hypothesis: If the replay latent count remains live after the first screen but still looks representation-limited, widening `d_icl` to `128` may recover quality without needing the larger latent bank.
- Upstream delta: Perceiver-style width changes alter latent-channel capacity and attention projection width; this is the lower width follow-up row once the latent screen is visible.
- Anchor delta: Keep row `01` fixed and change only `d_icl` from `96` to `128`.
- Expected effect: More width may improve fit if the replay is width-bottlenecked, though it increases training cost and should not be interpreted before the latent screen.
- Effective labels: model=`tabfoundry_sandwich`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Resolved surface fingerprint: `7f0afef7ef10a52cb05baa1f6d6080539c83ea57287283cafa48c54049e29a2a`
- Resolved runtime surface: `{'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': False}`
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
  - This row is no longer blocked on the latent screen; it is deferred because the current topology underperformed before width became a meaningful next read.
  - Do not treat width as a rescue attempt for the closed summary-bottleneck replay.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021a_sandwich_openml_screen_v1/delta_tf_rd_021a_sandwich_width128_latents48_v1/result_card.md`
- Benchmark metrics: pending

### 5. `delta_tf_rd_021a_sandwich_width128_latents96_v1`

- Dimension family: `model`
- Status: `deferred_separate_workstream`
- Binary applicable: `True`
- Recipe alias: `none`
- Description: Increase sandwich width from 96 to 128 around the upper latent candidate at 96 latents.
- Rationale: Keep the high-latent width follow-up as backlog evidence only because the completed replay row did not justify widening the same summary-bottleneck architecture.
- Hypothesis: If the `96`-latent row is viable but still width-limited, the joint `d_icl=128` plus `sandwich_latents=96` setting may become the first serious second-pass size probe.
- Upstream delta: Perceiver-style scaling often couples latent count and channel width; this row is the blocked width follow-up around the high-latent bracket.
- Anchor delta: Keep row `03` fixed and change only `d_icl` from `96` to `128`.
- Expected effect: Jointly larger width and latent count may improve fit if the sandwich screen is constrained on both bottlenecks, but it is intentionally deferred until the latent screen lands.
- Effective labels: model=`tabfoundry_sandwich`, data=`prior_dump`, preprocessing=`runtime_default`, training=`prior_linear_warmup_decay`
- Resolved surface fingerprint: `eb3e1241064c60b3bc3fae9b7a88561be9f5d6bbf1c42af42da793674d29cb0f`
- Resolved runtime surface: `{'grad_clip': 1.0, 'max_steps': 2500, 'trace_activations': False}`
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
  - This row remains historical backlog evidence around the abandoned high-latent bracket.
  - Do not read this as an active next step for the closed TF-RD-021A sweep.
- Follow-up run ids: `[]`
- Result card path: `outputs/staged_ladder/research/tf_rd_021a_sandwich_openml_screen_v1/delta_tf_rd_021a_sandwich_width128_latents96_v1/result_card.md`
- Benchmark metrics: pending

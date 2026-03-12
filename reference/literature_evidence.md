# Literature Evidence Mapping

This document links roadmap items in `docs/development/roadmap.md` to primary sources and current repo gaps.

Related docs:

- Canonical roadmap: `docs/development/roadmap.md`
- Architecture strategy: `docs/ARCHITECTURE_STRATEGY.md`
- Paper index: `reference/PAPERS.md`

Conventions:

- Confidence: `high`, `medium`, or `low`.
- Lane: roadmap milestone lane supported by this evidence (`Now`, `Next`, `Later`).

## Evidence-to-Roadmap Mapping

| Source | Key Claim Used | Roadmap IDs | Lane | Confidence |
|--------|---------------|-------------|------|------------|
| Chinchilla (2203.15556) | Compute-optimal model sizing follows predictable power laws; matching data and parameters to compute budget is critical | BL-174, BL-164 | Now | high |
| Kaplan et al. (2001.08361) | Power-law relationships between compute, data, and parameters are robust across scales | BL-174 | Now | high |
| Power Lines (2505.13738) | Refined scaling law fitting methodology improves prediction accuracy at small scales | BL-174 | Now | medium-high |
| Broken Neural Scaling Laws (2210.14891) | Scaling curves have knees and plateaus that simple power laws miss; diagnostic methodology needed | BL-174 | Now | medium |
| Scaling Data-Constrained LMs (2305.16264) | Data repetition and mixing strategies matter when synthetic data budget is fixed | BL-174 | Next | medium |
| TabICLv2 (2602.11139) | QASS attention, feature tokenization, and staged training define the current architecture | BL-181, BL-172, BL-165 | Now | high |
| TabICL (2502.05564) | Curriculum and staged complexity training improves optimization behavior | BL-160 | Now | high |
| TabPFN v2 (Nature, 2024) | Core PFN architecture patterns; attention mechanisms for in-context tabular learning | BL-181, BL-172 | Now | high |
| Deep Sets (1703.06114) | Permutation invariance is the right default lens for row and column structure; sequence order should be justified, not assumed | BL-165, BL-167, BL-169 | Now | high |
| Set Transformer (1810.00825) | Attention over unordered sets with inducing points is a stronger architectural default for row/column modeling than generic LLM positional machinery | BL-165, BL-167, BL-169 | Now | high |
| FT-Transformer (2106.11959) | Per-feature linear embedding is the baseline tokenization approach for tabular transformers | BL-165 | Now | high |
| On Embeddings for Numerical Features in Tabular Deep Learning (2203.05556) | Numerical/value embeddings are a high-leverage tokenization choice for tabular models | BL-165 | Now | high |
| muP (2203.03466) | Width-independent hyperparameter transfer enables tuning at small scale and transferring to larger models | BL-164 | Now | high |
| SGDR (1608.03983) | Cosine annealing with warm restarts is the standard schedule for short-run training | BL-160 | Now | high |
| A Closer Look at TabPFN v2 (2502.17361) | Meta-feature sensitivity analysis reveals architecture robustness gaps | BL-165, BL-169 | Now | medium-high |
| TabPFN Unleashed (2502.02527) | Bias/variance control techniques for large-scale adaptation | BL-164, BL-169 | Next | medium |
| TuneTables (2402.11137) | Context optimization patterns for parameter-efficient adaptation | BL-165 | Next | medium |
| TabDPT (2410.18164) | Real-data pretraining shows alternative scaling paths beyond synthetic-only training | BL-174 | Next | medium |
| EquiTabPFN (2502.06684) | Target-permutation equivariance improves robustness in label-space handling | BL-166 | Next | medium |
| ColBERT (2004.12832) | Late interaction is a benchmark-first factorized interaction pattern for order-light structures, not a default backbone choice | BL-167 | Next | medium |
| Drift-Resilient TabPFN (2411.10634) | Architectural choices for temporal shift robustness inform backbone design | BL-167 | Later | medium |
| DeepSeek-V3 (2412.19437) | Multi-token prediction improves cross-feature dependency handling in compact transformers | BL-167 | Later | medium |
| nanoTabPFN (repo) | Training recipe and model sizing provide the direct baseline for tab-foundry benchmarking | BL-173, BL-155 | Now | high |
| nanochat (repo) | Compact-transformer recipe donor for optimizer partitioning, FFN/residual choices, model sizing, and training-loop structure when ideas do not depend on sequence order | BL-173, BL-160, BL-161, BL-169 | Now | high |

## Per-Epic Evidence Notes

### Epic 3: Modular Architecture Platform

#### BL-170 — Literature search and architecture references

- This document and `reference/PAPERS.md` are the primary deliverables.
- Confidence: high (this is the deliverable itself).

#### BL-171 — Neutral architecture naming and registry

- TabICLv2 (2602.11139) defines the current architecture family.
- Architecture strategy calls for treating `tabiclv2` as a compatibility identifier, not the long-term taxonomy.
- No single paper drives this; it is a repo-organization decision informed by the multi-family direction in the architecture strategy.

#### BL-181 — Split reusable model components

- TabICLv2 (2602.11139): defines QASS attention, feature tokenization, and target conditioning as separable components.
- TabPFN v2 (Nature): core attention patterns that should be reusable across families.
- Success signal: components can be composed independently without architecture-specific coupling.

#### BL-172 — Modular QASS and non-QASS backbone

- TabICLv2 (2602.11139): defines the QASS mechanism and its role in the architecture.
- Architecture strategy: QASS remains optional rather than structurally mandatory.
- Success signal: a non-QASS variant trains and evaluates through the same pipeline.

#### BL-173 — External baseline configs

- nanoTabPFN: training recipe, model sizing, optimizer choices.
- nanochat: compact-transformer FFN recipe, pre-norm residual structure, Muon integration, LR schedule, and model sizing patterns.
- Success signal: at least one external-inspired config runs through the benchmark pipeline.

#### BL-174 — Scaling-law measurement

- Chinchilla (2203.15556): methodology for fitting compute-optimal curves.
- Kaplan et al. (2001.08361): foundational power-law methodology.
- Power Lines (2505.13738): refined fitting at small scales.
- Broken Neural Scaling Laws (2210.14891): diagnostic methodology for knees and plateaus.
- Success signal: a fitted scaling curve across at least 3 model sizes with interpretable coefficients.

### Epic 4: Hyperparameter Tuning

#### BL-160 — Schedule sweep

- TabICL (2502.05564): staged complexity training informs schedule design.
- SGDR (1608.03983): cosine annealing with warm restarts as baseline schedule.
- nanochat: schedule patterns from compact transformer training.
- Success signal: at least 3 schedule variants compared on the same architecture and budget.

#### BL-161 — Optimizer sweep

- Muon optimizer (repo): modern optimizer for transformer weight convergence.
- nanochat: optimizer choices and integration patterns.
- Watchlist items such as `Polar Express` stay outside the curated evidence table until a primary source is available.
- Success signal: AdamW vs. Muon (and optionally others) compared on the same architecture and budget.

#### BL-162 — Batching and clipping sweep

- No single paper drives this directly; informed by general training recipe literature.
- Success signal: stability metrics (gradient norm variance, loss spikes) improve under best config.

#### BL-164 — Model size sweep

- Chinchilla (2203.15556): compute-optimal sizing methodology.
- muP (2203.03466): width-independent hyperparameter transfer.
- Success signal: at least 3 model sizes with consistent hyperparameters (via muP) producing a scaling curve.

### Epic 5: Architecture Ablations

#### BL-165 — Feature tokenization ablation

- FT-Transformer (2106.11959): per-feature linear embedding baseline.
- On Embeddings for Numerical Features in Tabular Deep Learning (2203.05556): value embedding and scalar encoding choices are likely high leverage.
- TabICLv2 (2602.11139): current tokenization approach.
- Deep Sets (1703.06114) and Set Transformer (1810.00825): tokenization changes should preserve the repo's set-structured view of rows and columns.
- A Closer Look at TabPFN v2 (2502.17361): meta-feature sensitivity.
- Success signal: tokenization variant produces measurable difference in scaling behavior.

#### BL-166 — Target conditioning ablation

- EquiTabPFN (2502.06684): target-permutation equivariance.
- TabICLv2 (2602.11139): current target conditioning approach.
- Success signal: conditioning variant produces measurable difference in label-space robustness.

#### BL-167 — Repeated asymmetric row attention ablation

- Deep Sets (1703.06114): row and column interactions should justify any departure from permutation-aware structure.
- Set Transformer (1810.00825): attention over unordered sets is the default comparison point.
- ColBERT (2004.12832): benchmark-first late interaction candidate if factorized row/column matching becomes the target hypothesis.
- Drift-Resilient TabPFN (2411.10634): attention patterns for robustness.
- DeepSeek-V3 (2412.19437): multi-token prediction as alternative dependency mechanism.
- Success signal: attention variant produces measurable difference in cross-feature dependency handling.

#### BL-169 — QASS and encoder simplification ablation

- TabICLv2 (2602.11139): defines the QASS mechanism.
- nanochat: `ReLU^2`, pre-norm residual simplicity, and compact-transformer recipe choices are strong simplification candidates when they do not rely on sequence order.
- Deep Sets (1703.06114) and Set Transformer (1810.00825): simplification should preserve set structure rather than importing language-sequence assumptions.
- A Closer Look at TabPFN v2 (2502.17361): analysis of architectural strengths/limitations.
- Success signal: QASS removal or simplification produces interpretable scaling trade-off.

## Evidence Limits and Assumptions

- This is a planning artifact, not a reproduction benchmark.
- Mappings are design-level and must be validated against repo benchmarks and tests.
- Lane assignments reflect current repo context and can be revised after implementation learning.
- Most scaling law papers target language models; their applicability to tabular foundation models is an empirical question this repo aims to answer.
- Default transfer rule: borrow compact-transformer recipe ideas when they do not depend on sequence order; prefer set/permutation-aware references for row/column structure.
- Sequence-order-centric mechanisms are intentionally de-prioritized until a tabular-specific justification appears.

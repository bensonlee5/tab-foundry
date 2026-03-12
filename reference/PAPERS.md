# Papers And References

This directory is the starting point for literature-first architecture work in `tab-foundry`.

It intentionally mirrors the role of `~/dev/dagzoo/reference`, but reads the shared literature through an **architecture, training recipe, and scaling predictability** lens rather than a data-generation lens.

Related docs:

- Architecture strategy: `docs/ARCHITECTURE_STRATEGY.md`
- Roadmap: `docs/development/roadmap.md`
- Evidence mapping: `reference/literature_evidence.md`
- External repos: `reference/external_repos.md`

## Adoption Tiers For This Repo

Use this directory to make explicit judgments, not just collect papers.

- `Likely adopt`
  - compact-transformer recipe ideas from `nanochat` when they do not depend on sequence order
  - set- and permutation-aware modeling for row and column structure
  - feature and value embedding work that improves scalar tabular tokenization
- `Benchmark first`
  - ColBERT-like late interaction or other factorized interaction patterns
  - expensive optimizers beyond current defaults, especially because the intended transformer family is small
  - retrieval-heavy or interaction-heavy mechanisms that materially change the inductive bias
- `Probably low relevance`
  - RoPE and similar positional schemes whose main value comes from token order in language
  - causal-LM-specific sequence machinery that does not cleanly transfer to unordered rows or columns

## Tabular Foundation Models

These are the core domain papers. Many overlap with dagzoo's collection but are read here through an architecture and training lens.

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| 2602.11139 | TabICLv2: A better, faster, scalable, and open tabular foundation model | Primary architecture reference — the model this repo implements. Defines QASS attention, feature tokenization, and the staged training recipe. | https://arxiv.org/abs/2602.11139 |
| 2502.05564 | TabICL: A Tabular Foundation Model for In-Context Learning on Large Data | Predecessor architecture; curriculum and staged complexity training details that inform the training loop design. | https://arxiv.org/abs/2502.05564 |
| — | Accurate predictions on small data with a tabular foundation model (TabPFN v2, Nature 2024) | Core PFN architecture and synthetic prior design; attention patterns and in-context learning mechanics that underpin the model family. | https://doi.org/10.1038/s41586-024-08328-6 |
| 2502.17361 | A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities | Strengths/limitations analysis; meta-feature sensitivity insights relevant to architecture robustness. | https://arxiv.org/abs/2502.17361 |
| 2410.18164 | TabDPT: Scaling Tabular Foundation Models on Real Data | Real-data pretraining as an alternative to purely synthetic training; informs architecture decisions around data source flexibility. | https://arxiv.org/abs/2410.18164 |
| 2502.02527 | TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems | Bias/variance control and large-scale adaptation techniques; relevant to scaling architecture decisions. | https://arxiv.org/abs/2502.02527 |
| 2311.10609 | Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks | Feature selection and compression for high-dimensional scalability; informs the feature tokenization pipeline. | https://arxiv.org/abs/2311.10609 |
| 2402.11137 | TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks | Parameter-efficient PFN adaptation; context optimization patterns relevant to architecture modularity. | https://arxiv.org/abs/2402.11137 |
| 2406.05207 | Retrieval & Fine-Tuning for In-Context Tabular Models (LoCalPFN) | Retrieval-conditioned adaptation; architecture patterns for dataset-aware inference. | https://arxiv.org/abs/2406.05207 |
| 2502.06684 | EquiTabPFN: A Target-Permutation Equivariant Prior Fitted Networks | Target-permutation equivariance; robust label-space handling relevant to readout head design. | https://arxiv.org/abs/2502.06684 |
| 2411.10634 | Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data | Architectural choices for temporal shift robustness; informs backbone design for distribution-shift tolerance. | https://arxiv.org/abs/2411.10634 |

## Set And Permutation-Aware Modeling

These references matter because rows and columns in tabular models are closer to sets than to language sequences. They should generally outrank generic LLM positional ideas when the two conflict.

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| 1703.06114 | Deep Sets | Canonical permutation-invariant baseline. Useful as the default mental model when deciding whether a row/column mechanism should care about order at all. | https://arxiv.org/abs/1703.06114 |
| 1810.00825 | Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks | Attention over unordered sets with inducing-point scaling. Especially relevant because the repo already uses ISAB-like components in the column path. | https://proceedings.mlr.press/v97/lee19d.html |

### Feature And Value Embeddings

These are likely higher-leverage than sequence positional tricks for a small tabular transformer.

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| 2106.11959 | Revisiting Deep Learning Models for Tabular Data (FT-Transformer) | Per-feature linear embedding baseline for the tokenization ablation. Keep as the simple reference point, not the final architecture identity. | https://arxiv.org/abs/2106.11959 |
| 2203.05556 | On Embeddings for Numerical Features in Tabular Deep Learning | Direct reference for scalar-to-vector value embeddings. High-priority source for feature/value tokenization ideas that fit unordered tabular structure. | https://arxiv.org/abs/2203.05556 |

### Late Interaction And Factorized Matching

These are benchmark-first references. They are interesting because tables are relatively order-light, but they should not become the default backbone assumption without evidence.

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| 2004.12832 | ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT | Factorized late interaction is a plausible row/column matching pattern for tabular models, but it is a benchmark-first hypothesis rather than a default adopt item. | https://arxiv.org/abs/2004.12832 |

## Compact Transformers And Training Recipes

Papers that directly serve the short-run training constraint and modular architecture goals. These are new to tab-foundry (not in dagzoo).

Default filter for this section:

- borrow compact-transformer recipe ideas when they do not depend on sequence order
- treat `nanochat` as the main practical repo reference for FFN choice, residual/pre-norm block hygiene, optimizer partitioning, and model sizing
- treat `ReLU^2` as a concrete candidate for the baseline FFN recipe
- spend optimization budget more freely than an LLM repo would, because the target transformer family is small
- keep RoPE and similar language-order mechanisms low priority unless a tabular-specific use case appears

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| 2203.15556 | Training Compute-Optimal Large Language Models (Chinchilla) | Core reference for the repo's primary goal of scaling predictability. Defines the methodology for fitting compute-optimal curves across model sizes. | https://arxiv.org/abs/2203.15556 |
| 2001.08361 | Scaling Laws for Neural Language Models | Foundational scaling law methodology; establishes the power-law relationships between compute, data, and model size that this repo aims to replicate for tabular models. | https://arxiv.org/abs/2001.08361 |
| 2203.03466 | Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP) | Width-independent hyperparameter transfer across model sizes. Directly relevant to model size sweeps (BL-164) — enables tuning at small scale and transferring to larger models. | https://arxiv.org/abs/2203.03466 |
| 1608.03983 | SGDR: Stochastic Gradient Descent with Warm Restarts | Cosine annealing with warm restarts; directly relevant to the schedule sweep work (BL-160) and short-run training budget optimization. | https://arxiv.org/abs/1608.03983 |
| 2412.19437 | DeepSeek-V3 Technical Report | Multi-token prediction and modern training recipe details for compact transformers. Informs architecture choices for cross-feature dependency modeling. | https://arxiv.org/abs/2412.19437 |

## Scaling Laws

Dedicated section since scaling predictability is the repo's primary goal.

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| 2203.15556 | Training Compute-Optimal Large Language Models (Chinchilla) | Direct methodology reference for fitting compute-optimal curves. The primary template for the scaling-law measurement work (BL-174). | https://arxiv.org/abs/2203.15556 |
| 2001.08361 | Scaling Laws for Neural Language Models | Foundational; establishes power-law relationships between compute budget, dataset size, and model parameters. | https://arxiv.org/abs/2001.08361 |
| 2505.13738 | Power Lines: Scaling Laws for Language Models | Scaling law methodology refinements; directly relevant to building the scaling predictability measurement infrastructure. | https://arxiv.org/abs/2505.13738 |
| 2305.16264 | Scaling Data-Constrained Language Models | Relevant when synthetic data budget is the bottleneck; analyzes how data repetition and mixing affects scaling behavior. | https://arxiv.org/abs/2305.16264 |
| 2210.14891 | Broken Neural Scaling Laws | Explains why scaling isn't always a simple power law; critical for diagnosing knees and plateaus in scaling curves. | https://arxiv.org/abs/2210.14891 |

## External Repo References

Notes on adjacent implementations. See `reference/external_repos.md` for detailed notes on what to borrow and why.

| Repo | Link | Why it matters |
|------|------|---------------|
| nanoTabPFN | https://github.com/automl/nanoTabPFN | Benchmark comparison target; training recipe choices and model sizing decisions provide direct baseline for tab-foundry. |
| nanochat | https://github.com/karpathy/nanochat | Optimizer, LR schedule, model sizing, and clean training loop patterns. Reference for compact transformer training infrastructure. |
| Muon optimizer | https://github.com/KellerJordan/Muon | Modern optimizer treating weights as orthogonal matrices. Relevant to optimizer sweep work (BL-161). |

## Usage Contract

- Major architecture tickets should cite the relevant references from this directory.
- Each reference note should say why it matters and what signal would count as success or failure.
- New papers should be added here before they inform architecture changes (literature-first construction).
- New entries should say whether they are `likely adopt`, `benchmark first`, or `probably low relevance`.
- Cross-reference with `reference/literature_evidence.md` for roadmap item mappings.

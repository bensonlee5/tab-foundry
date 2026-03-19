# Papers And References

This directory is the starting point for literature-first architecture work in `tab-foundry`.

It intentionally mirrors the role of `~/dev/dagzoo/reference`, but reads the shared literature through an **architecture, training recipe, and scaling predictability** lens rather than a data-generation lens.

Related docs:

- Design decisions and repo structure: `docs/development/design-decisions.md`
- Roadmap: `docs/development/roadmap.md`
- Workflow runbooks: `docs/workflows.md`
- Evidence mapping: `reference/evidence.md`

## Adoption Tiers For This Repo

Use this directory to make explicit judgments, not just collect papers.

- `Likely adopt`
  - numerical embeddings, including ordinal-as-numeric handling unless monotonicity is the central requirement
  - categorical entity embeddings as the default baseline before heavier contextualization
  - compact-transformer recipe ideas from `nanochat` when they do not depend on sequence order
  - typed column encoders that preserve set- and permutation-aware structure
- `Benchmark first`
  - TabTransformer contextual categorical embeddings
  - SAINT row/column attention and Perceiver-style latent bottlenecks
  - ColBERT-like late interaction or other factorized interaction patterns
  - table-aware text models
  - expensive optimizers beyond current defaults, especially because the intended transformer family is small
  - retrieval-heavy or interaction-heavy mechanisms that materially change the inductive bias
- `Later / TF-RD-012`
  - text-conditioned column handling via external text embeddings and table-aware text encoders
- `Probably low relevance`
  - RoPE and similar positional schemes whose main value comes from token order in language
  - causal-LM-specific sequence machinery that does not cleanly transfer to unordered rows or columns

## Tabular Foundation Models

These are the core domain papers. Many overlap with dagzoo's collection but are read here through an architecture and training lens.

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| 2602.11139 | TabICLv2: A better, faster, scalable, and open tabular foundation model | Primary external architecture reference for the row-first target lane. Defines QASS attention, feature tokenization, and the staged training recipe that informs the migration ladder. | https://arxiv.org/abs/2602.11139 |
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

### Typed Column Encoders

Treat tokenization here as typed column encoding, not just as a generic feature projection step. The question is how numerical, categorical, ordinal, and later text-conditioned columns should enter a small set-structured backbone.

Default typed-column policy:

- numerical and ordinal columns should default to numeric-style encoders
- categorical columns should default to learned embeddings before more complex contextualization
- text columns should default to external text embeddings first, not raw subword tokenization inside the main table backbone
- set/permutation-aware structure remains the governing constraint for how typed tokens enter the shared model

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| 2106.11959 | Revisiting Deep Learning Models for Tabular Data (FT-Transformer) | Baseline per-feature tokenization reference. Useful as the simple typed-column baseline before heavier encoder choices are justified. | https://arxiv.org/abs/2106.11959 |

### Numerical And Ordinal Columns

These are likely higher-leverage than sequence positional tricks for a small tabular transformer. Default repo stance: ordinals should usually be treated as numeric unless monotonicity is a central modeling requirement.

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| 2203.05556 | On Embeddings for Numerical Features in Tabular Deep Learning | Direct reference for scalar-to-vector value embeddings. High-priority source for feature/value tokenization ideas that fit unordered tabular structure. | https://arxiv.org/abs/2203.05556 |

### Categorical Columns

Categorical columns should default to learned embeddings before more expensive contextualization is introduced.

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| 1604.06737 | Entity Embeddings of Categorical Variables | Likely-adopt baseline for categorical columns, especially when cardinality is high enough that one-hot handling is a poor fit. | https://arxiv.org/abs/1604.06737 |
| 2012.06678 | TabTransformer: Tabular Data Modeling Using Contextual Embeddings | Benchmark-first contextualization reference for categorical columns. Useful when plain learned embeddings are not enough. | https://arxiv.org/abs/2012.06678 |

### Text-Conditioned Columns (Later / TF-RD-012)

These references belong to the later text-conditioned-input lane. The default direction is to use external text embeddings first, not raw subword tokenization inside the small tabular backbone.

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| D19-1410 | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Practical baseline for turning free-text cells or column values into fixed embeddings before they enter the tabular model. | https://aclanthology.org/D19-1410/ |
| 2020.acl-main.745 | TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data | Benchmark-first table-aware text model. Relevant if later work needs richer joint text-table reasoning rather than external text embeddings alone. | https://aclanthology.org/2020.acl-main.745/ |
| 2006.14806 | TURL: Table Understanding through Representation Learning | Later benchmark reference for structure-aware table/text representations. Useful when text-conditioned columns become a first-class roadmap item. | https://arxiv.org/abs/2006.14806 |

### Scaling Row/Column Token Counts

These are benchmark-first references for scaling row/column interaction costs. They are architecture references, not default tokenization choices.

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| 2106.01342 | SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training | Benchmark-first row/column attention reference. Useful if typed encoders become good enough that inter-row attention is the next bottleneck. | https://arxiv.org/abs/2106.01342 |
| 2103.03206 | Perceiver: General Perception with Iterative Attention | Latent-bottleneck reference for scaling to very large token counts. Relevant if row/column counts make full attention too expensive. | https://proceedings.mlr.press/v139/jaegle21a.html |

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
| 2203.03466 | Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP) | Width-independent hyperparameter transfer across model sizes. Directly relevant to scaling-law measurement on the promoted anchor (`TF-RD-009`). | https://arxiv.org/abs/2203.03466 |
| 1608.03983 | SGDR: Stochastic Gradient Descent with Warm Restarts | Cosine annealing with warm restarts; relevant to training-recipe work that supports the promoted anchor rather than replacing architecture migration (`TF-RD-009`). | https://arxiv.org/abs/1608.03983 |
| 2412.19437 | DeepSeek-V3 Technical Report | Multi-token prediction and modern training recipe details for compact transformers. Informs architecture choices for cross-feature dependency modeling. | https://arxiv.org/abs/2412.19437 |

## Scaling Laws

Dedicated section since scaling predictability is the repo's primary goal.

| arXiv ID | Title | Why it matters for tab-foundry | Source |
|----------|-------|-------------------------------|--------|
| 2203.15556 | Training Compute-Optimal Large Language Models (Chinchilla) | Direct methodology reference for fitting compute-optimal curves. The primary template for the scaling-law measurement work (`TF-RD-009`). | https://arxiv.org/abs/2203.15556 |
| 2001.08361 | Scaling Laws for Neural Language Models | Foundational; establishes power-law relationships between compute budget, dataset size, and model parameters. | https://arxiv.org/abs/2001.08361 |
| 2505.13738 | Power Lines: Scaling Laws for Language Models | Scaling law methodology refinements; directly relevant to building the scaling predictability measurement infrastructure. | https://arxiv.org/abs/2505.13738 |
| 2305.16264 | Scaling Data-Constrained Language Models | Relevant when synthetic data budget is the bottleneck; analyzes how data repetition and mixing affects scaling behavior. | https://arxiv.org/abs/2305.16264 |
| 2210.14891 | Broken Neural Scaling Laws | Explains why scaling isn't always a simple power law; critical for diagnosing knees and plateaus in scaling curves. | https://arxiv.org/abs/2210.14891 |

## External Repo References

Adjacent repo references that inform architecture and training decisions. Detailed borrowing rules live below.

| Repo | Link | Why it matters |
|------|------|---------------|
| nanoTabPFN | https://github.com/automl/nanoTabPFN | Benchmark comparison target; training recipe choices and model sizing decisions provide direct baseline for tab-foundry. |
| nanochat | https://github.com/karpathy/nanochat | Optimizer, LR schedule, model sizing, and clean training loop patterns. Reference for compact transformer training infrastructure. |
| Muon optimizer | https://github.com/KellerJordan/Muon | Modern optimizer treating weights as orthogonal matrices. Relevant to later training-recipe work on the promoted anchor (`TF-RD-009`). |

## External Baseline Borrowing Rules

Default transfer rule for repo references:

- borrow compact-transformer recipe ideas when they do not depend on sequence order
- prefer set- and permutation-aware references for row and column structure
- treat language-sequence positional machinery as low priority by default

### nanoTabPFN

- **URL:** https://github.com/automl/nanoTabPFN
- **Roadmap relevance:** TF-RD-001 (PFN control lane), TF-RD-008 (anchor promotion)

What it does well:

- clean, minimal TabPFN training implementation with good defaults
- establishes a concrete performance baseline for short-run tabular PFN training
- model sizing choices provide a known-good starting point

What to absorb:

- training recipe as a named baseline config
- depth/width/head sizing conventions as a reference point for model size sweeps
- evaluation protocol for direct comparison

Success signal:

- `tab-foundry` reaches parity with `nanoTabPFN` on the same benchmark suite and training budget
- at least one `tab-foundry` config is directly derived from `nanoTabPFN`'s recipe and runs through the benchmark pipeline

### nanochat

- **URL:** https://github.com/karpathy/nanochat
- **Roadmap relevance:** TF-RD-002 (measurement surfaces), TF-RD-007 (QASS attribution), TF-RD-009 (training and scaling work on the promoted anchor)

What it does well:

- minimalist, high-quality reference for modern transformer training pipelines
- clean integration of Muon with a standard training loop
- strong compact-transformer backbone hygiene: pre-norm residual layout, simple block structure, and small-model sizing discipline
- readable model definition and training loop that separates concerns well
- good LR schedule implementation with warmup and cosine decay

What to absorb:

- `ReLU^2` as a concrete compact-transformer FFN candidate instead of defaulting automatically to GELU
- pre-norm residual simplicity and readable block structure where it fits tabular modeling
- Muon plus AdamW parameter-group partitioning
- warmup and cosine schedule patterns
- depth/width/head scaling conventions
- value/token encoding ideas only where they translate cleanly to tabular value tokenization

What not to copy by default:

- causal-LM assumptions
- sequence-order positional encodings such as RoPE
- autoregressive masking or next-token machinery whose value depends on token order
- mechanisms whose main benefit is language-style sequence structure rather than unordered row/column interactions

Success signal:

- the `tab-foundry` training loop reaches comparable clarity and separation of concerns
- Muon integration works correctly for tabular parameter groups
- schedule patterns from `nanochat` are available as named options in sweep infrastructure
- a compact tabular baseline can resemble `nanochat`'s recipe choices while still using set-structured inductive biases

### Set / Permutation-Aware Priority Note

- **Type:** paper-backed priority rule rather than a single external repo
- **Roadmap relevance:** TF-RD-004 (tokenization migration), TF-RD-005 (row-embedding unlock), TF-RD-007 (QASS attribution)

Why it matters:

- rows and columns in tabular models are much closer to sets than to text sequences
- when repo references and paper references disagree, set- and permutation-aware work should usually win for row/column structure
- this is the main reason to elevate `Deep Sets` and `Set Transformer` ahead of generic LLM positional tricks

What to absorb:

- permutation invariance as the default test for row/column mechanisms
- set-style attention patterns before language-style positional patterns
- factorized or late interaction ideas only as benchmark-first hypotheses, not as default structure

Success signal:

- architecture ablations are framed in terms of set structure and interaction patterns rather than porting LLM sequence machinery
- new row/column blocks justify any positional assumptions explicitly instead of inheriting them by default

### Muon Optimizer

- **URL:** https://github.com/KellerJordan/Muon
- **Roadmap relevance:** TF-RD-009 (training work on the promoted anchor)

What it does well:

- treats weight matrices as elements of the orthogonal group
- has shown faster convergence and lower post-warmup variance than AdamW in compact transformer settings

What to absorb:

- apply Muon to weight matrices only, with embeddings and biases staying on AdamW
- use Muon's default learning rate and momentum as sweep starting points

Success signal:

- Muon runs correctly on `tab-foundry` with proper parameter-group assignment
- optimizer sweeps compare Muon and AdamW under matched budgets

### Optimizer Watchlist

These are intentionally not yet curated as primary-source references:

- `Polar Express`: keep as a watchlist item until a primary paper or official technical source is available
- general rule: expensive optimizers are more viable here than in frontier-scale LLM settings because the target transformer family is small
- watchlist items should graduate into the curated reference list only after a primary source is available

## Usage Contract

- Major architecture tickets should cite the relevant references from this directory.
- Each reference note should say why it matters and what signal would count as success or failure.
- New papers should be added here before they inform architecture changes (literature-first construction).
- New entries should say whether they are `likely adopt`, `benchmark first`, or `probably low relevance`.
- Cross-reference with `reference/evidence.md` for roadmap item mappings.

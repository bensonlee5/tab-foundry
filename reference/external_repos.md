# External Repo References

Structured notes on adjacent implementations that inform tab-foundry's architecture and training decisions.

Related docs:

- Paper index: `reference/PAPERS.md`
- Architecture strategy: `docs/ARCHITECTURE_STRATEGY.md`
- Roadmap: `docs/development/roadmap.md`

Default transfer rule for repo references:

- borrow compact-transformer recipe ideas when they do not depend on sequence order
- prefer set- and permutation-aware references for row and column structure
- treat language-sequence positional machinery as low priority by default

## nanoTabPFN

- **URL:** https://github.com/automl/nanoTabPFN
- **Roadmap relevance:** BL-173 (external baseline configs), BL-155 (control baseline)

### What it does well

- Clean, minimal TabPFN training implementation with good defaults.
- Establishes a concrete performance baseline for tabular PFN training at short-run budgets.
- Model sizing choices (depth, width, head count) provide a known-good starting point.

### What to absorb

- **Training recipe:** Optimizer choice, learning rate, warmup schedule, and batch size as a named baseline config in tab-foundry.
- **Model sizing:** Depth/width/head configuration as a reference point for the model size sweep (BL-164).
- **Evaluation protocol:** Benchmark dataset selection and evaluation metrics for direct comparison.

### Success signal

- tab-foundry achieves performance parity with nanoTabPFN on the same benchmark suite and training budget.
- At least one tab-foundry config is directly derived from nanoTabPFN's recipe and runs through the benchmark pipeline.

______________________________________________________________________

## nanochat

- **URL:** https://github.com/karpathy/nanochat
- **Roadmap relevance:** BL-173 (external baseline configs), BL-160 (schedule sweep), BL-161 (optimizer sweep), BL-169 (QASS and encoder simplification)

### What it does well

- Minimalist, high-quality reference for modern transformer training pipelines.
- Clean integration of Muon optimizer with standard training loop.
- Strong compact-transformer backbone hygiene: pre-norm residual layout, simple block structure, and small-model sizing discipline.
- Readable model definition and training loop that separates concerns well.
- Good LR schedule implementation (warmup + cosine decay).

### What to absorb

- **FFN recipe:** Treat `ReLU^2` as a concrete candidate for the compact-transformer baseline rather than defaulting automatically to GELU.
- **Residual and normalization structure:** Favor the same kind of pre-norm residual simplicity and block readability where it fits tabular modeling.
- **Muon optimizer integration:** How Muon is configured and applied alongside standard AdamW for different parameter groups (e.g., embeddings vs. attention weights).
- **LR schedule patterns:** Warmup duration, cosine decay shape, and end-of-training LR ratio.
- **Model sizing conventions:** How depth, width, and head count scale together.
- **Training loop structure:** Clean separation of forward pass, loss computation, gradient accumulation, and optimizer step.
- **Value/token encoding patterns:** Treat scalar-to-vector input encoding ideas as relevant only where they translate cleanly to tabular value tokenization.

### What not to copy by default

- Causal-LM assumptions.
- Sequence-order positional encodings such as RoPE.
- Autoregressive masking or next-token machinery whose value depends on token order.
- Any mechanism whose main benefit is language-style sequence structure rather than unordered row/column interactions.

### Success signal

- tab-foundry's training loop achieves comparable code clarity and separation of concerns.
- Muon optimizer integration works correctly for the tabular architecture's parameter groups.
- LR schedule patterns from nanochat are available as named schedule options in the sweep infrastructure.
- A compact tabular baseline can resemble nanochat's recipe choices while still using set-structured inductive biases instead of language-sequence assumptions.

______________________________________________________________________

## Set / Permutation-Aware Priority Note

- **Type:** Paper-backed priority rule rather than a single external repo
- **Roadmap relevance:** BL-165 (feature tokenization ablation), BL-167 (row-attention ablation), BL-169 (QASS and encoder simplification)

### Why it matters

- Rows and columns in tabular models are much closer to sets than to text sequences.
- When repo references and paper references disagree, set- and permutation-aware work should usually win for row/column structure.
- This is the main reason to elevate `Deep Sets` and `Set Transformer` ahead of generic LLM positional tricks.

### What to absorb

- Permutation invariance as the default test for row/column mechanisms.
- Set-style attention patterns before language-style positional patterns.
- Factorized or late interaction ideas only as benchmark-first hypotheses, not as default structure.

### Success signal

- Architecture ablations are phrased in terms of set structure and interaction patterns, not in terms of porting LLM sequence machinery.
- New row/column blocks justify any positional assumptions explicitly instead of inheriting them by default.

______________________________________________________________________

## Muon Optimizer

- **URL:** https://github.com/KellerJordan/Muon
- **Roadmap relevance:** BL-161 (optimizer sweep)

### What it does well

- Treats weight matrices as elements of the orthogonal group; optimizer updates preserve orthogonality.
- Demonstrated faster convergence and lower post-warmup variance compared to AdamW in compact transformer settings.

### What to absorb

- **Parameter group handling:** Muon applies to weight matrices only; embeddings and biases still use AdamW. This split must be handled correctly in the optimizer setup.
- **Hyperparameter defaults:** Muon's default learning rate and momentum settings as starting points for the optimizer sweep.

### Success signal

- Muon runs correctly on tab-foundry's architecture with proper parameter group assignment.
- Optimizer sweep (BL-161) includes Muon vs. AdamW comparison with matched training budgets.

______________________________________________________________________

## Optimizer Watchlist

These are intentionally not yet curated in `reference/PAPERS.md`.

- `Polar Express`
  - Treat as a watchlist item only until a primary paper or official technical source is added to the reference set.
  - Relevant because the repo's target transformer family is small enough to seriously consider more expensive optimizers.
- General rule
  - Expensive optimizers are more viable here than in frontier-scale LLM settings.
  - Watchlist items should graduate into the curated reference list only after a primary source is available.

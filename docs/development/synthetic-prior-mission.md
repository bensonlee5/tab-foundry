# Problem Formulation

The primary objects are:

- data-side prior parameters $\\phi$ for the synthetic task generator
- model-side parameters $\\theta$ for the sandwich model

Everything else in the repo should be understood as machinery for sampling
from the induced task distribution and optimizing the resulting objectives.

Use these alongside this page:

- architecture reference:
  [Model Architecture](model-architecture.md)
- canonical roadmap:
  [Roadmap](roadmap.md)
- workflow runbooks:
  [Workflows](../workflows.md)
- sweep contract:
  [Sweep Contract](../../program.md)
- reference index:
  [References](../../reference/README.md)

## Problem Setup

Let a supervised tabular task be

$$ T = (X\_{\\mathrm{tr}}, Y\_{\\mathrm{tr}}, X\_{\\mathrm{te}}, Y\_{\\mathrm{te}}, \\tau). $$

with:

- $X\_{\\mathrm{tr}} \\in \\mathbb{R}^{N\_{\\mathrm{tr}} \\times C}$ and
  $X\_{\\mathrm{te}} \\in \\mathbb{R}^{N\_{\\mathrm{te}} \\times C}$
- $Y\_{\\mathrm{tr}}$ and $Y\_{\\mathrm{te}}$ the train and test labels in the
  current classification lane
- $\\tau$ the vector of per-column feature-type annotations
- $R = N\_{\\mathrm{tr}} + N\_{\\mathrm{te}}$ the total row count
- $C$ the observed feature count

`dagzoo` defines a parameterized distribution over such tasks:

$$ T \\sim p\_{\\phi}(T). $$

The overall program is bilevel:

$$ \\theta^{\*}(\\phi) = \\arg\\min\_{\\theta} L\_{\\mathrm{in}}(\\theta; \\phi) $$

$$ \\phi^{*} = \\arg\\max\_{\\phi} J\\bigl(\\phi, \\theta^{*}(\\phi)\\bigr) $$

The inner problem trains the sandwich model on tasks drawn from $p\_{\\phi}$.
The outer problem chooses the prior parameters $\\phi$ so that the resulting
trained model behaves well on held-out task families.

## Prior Parameterization

A useful factorization is

$$ \\phi = \\bigl( \\phi\_{\\mathrm{layout}}, \\phi\_{\\mathrm{mechanism}}, \\phi\_{\\mathrm{observation}}, \\phi\_{\\mathrm{task}}, \\phi\_{\\mathrm{shift}}, \\phi\_{\\mathrm{curriculum}} \\bigr). $$

Interpret these blocks as:

- $\\phi\_{\\mathrm{layout}}$: latent dependency structure, graph depth, density,
  and connectivity
- $\\phi\_{\\mathrm{mechanism}}$: functional families, interaction order,
  monotonicity, thresholding, and noise laws
- $\\phi\_{\\mathrm{observation}}$: $C$, feature-type composition $\\tau$,
  cardinality structure, missingness, and nuisance features
- $\\phi\_{\\mathrm{task}}$: label structure, class imbalance, Bayes error, label
  noise, and context/query sizes $N\_{\\mathrm{tr}}, N\_{\\mathrm{te}}$
- $\\phi\_{\\mathrm{shift}}$: discrepancy between train and test regimes
- $\\phi\_{\\mathrm{curriculum}}$: how task difficulty is reweighted or scheduled
  during training

This factorization matters because the sandwich model does not observe an
abstract prior. It observes tasks whose geometry and difficulty are induced by
these blocks.

## Grid Model As A Composition

The active hypothesis class is `grid_sandwich`, viewed mathematically as a
composition of maps over one task $T$. It inherits the previous
`tabfoundry_sandwich` tokenizer and feature-conditioning contract, but keeps
cell states arranged as a `[row, feature]` grid through the core mixers instead
of collapsing them into compressed context first.

First form the joint table

$$ X = [X\_{\\mathrm{tr}}; X\_{\\mathrm{te}}] \\in \\mathbb{R}^{R \\times C}. $$

Then define the grid computation in stages.

Cell encoding:

$$ E = e\_{\\theta}(X, \\tau). $$

where $E$ is the tensor of encoded cell states.

Label-conditioned grid state:

$$ G^{(0)} = g\_{\\theta}^{(0)}(E, Y\_{\\mathrm{tr}}). $$

Alternating row-wise and column-wise mixing:

$$ G^{(\\ell+1)} = c\_{\\theta}^{(\\ell)}(r\_{\\theta}^{(\\ell)}(G^{(\\ell)})). $$

where $r\_{\theta}^{(\ell)}$ mixes feature bundles within each row and
$c\_{\theta}^{(\ell)}$ mixes row states within each feature column.

Test-row bundle pooling and readout:

$$ O = h\_{\\theta}(G^{(L)}\_{\\mathrm{te}}). $$

where $O$ denotes the test-row logits. Conceptually:

- $E$ preserves high-bandwidth cell evidence
- $G^{(0)}$ attaches train labels only to train-row feature tokens
- row mixers let each observation form feature-level context
- column mixers let each feature share row-level evidence
- the readout pools each test-row feature bundle directly

The grid architecture is therefore a structured map from one task $T$ to one
matrix of test-row logits $O$.

## Inner Objective

For the current classification lane, the grid model induces a conditional
distribution over the test labels:

$$ p\_{\\theta}(Y\_{\\mathrm{te}} \\mid X\_{\\mathrm{tr}}, Y\_{\\mathrm{tr}}, X\_{\\mathrm{te}}, \\tau). $$

The inner supervised training objective is

$$ L\_{\\mathrm{in}}(\\theta; \\phi) = \\mathbb{E}_{T \\sim p_{\\phi}} \\left[ - \\sum\_{i=1}^{N\_{\\mathrm{te}}} \\log p\_{\\theta}(Y\_{\\mathrm{te},i} \\mid X\_{\\mathrm{tr}}, Y\_{\\mathrm{tr}}, X\_{\\mathrm{te}}, \\tau) \\right]. $$

This is the mathematical core of the current training problem: learn
$\\theta$ so that test labels are predictable from the train rows, test rows,
and feature-type metadata under tasks drawn from $p\_{\\phi}$.

## Outer Objective

The outer problem is not to minimize the inner loss on one fixed task family.
It is to choose $\\phi$ so that the model trained under $p\_{\\phi}$ generalizes
well across held-out task families.

Let $\\theta^{\*}(\\phi)$ denote the inner optimum and let
$\\mathcal{F}\_{\\mathrm{val}}$ be the held-out validation family distribution.
Then one useful scalarization is

$$ J(\\phi) = \\mathbb{E}_{F \\sim \\mathcal{F}_{\\mathrm{val}}} \\left\[ \\operatorname{Perf}(F; \\theta^{*}(\\phi)) - \\lambda\_{\\mathrm{cal}} \\operatorname{Cal}(F; \\theta^{*}(\\phi)) - \\lambda\_{\\mathrm{stab}} \\operatorname{Stab}(F; \\theta^{*}(\\phi)) - \\lambda\_{\\mathrm{worst}} \\operatorname{Worst}(F; \\theta^{*}(\\phi)) \\right\]. $$

The precise metrics may vary, but mathematically the outer objective should
reward:

- strong average predictive performance
- good calibration under shift
- stability across seeds and budgets
- low worst-family failure

The outer problem is therefore a prior-selection problem over $\\phi$, not
merely a training problem over $\\theta$.

## Current Implementation Status

This section is not part of the mathematical statement. It records which
pieces of the outer-objective story are implemented in the repo today and
which pieces would still have to be built.

### Built Today

- one primary objective metric is persisted per task or loss-surface lane; the
  runtime telemetry records `objective_metric` rather than a full weighted
  objective specification
- benchmark and registry artifacts already retain aggregate metrics that could
  feed a future composite score, including BPC, log loss, Brier score, ROC
  AUC, CRPS, pinball loss, and training-time summaries
- sweep tooling already retains some guardrail-style telemetry such as clipped
  step fraction and local stability diagnostics
- current classification benchmark policy is implemented operationally as one
  primary ranking metric plus guardrails; the active roadmap ranks rows by
  `final_log_loss_at_matched_regime_budget`, with calibration, runtime, and
  stability treated as guardrails rather than folded into one weighted scalar

### Not Built Yet

- there is no repo-level $\\lambda\_{\\mathrm{cal}}$, $\\lambda\_{\\mathrm{stab}}$,
  or $\\lambda\_{\\mathrm{worst}}$
  parameter surface today
- there is no canonical composite-score engine that combines multiple metrics
  into one weighted outer objective used for row ranking
- $\\operatorname{Cal}$, $\\operatorname{Stab}$, and
  $\\operatorname{Worst}$ are not frozen as one canonical scalar each in the
  current artifact contract
- the benchmark registry does not currently persist an explicit worst-family or
  per-task risk summary suitable for a true $\\operatorname{Worst}$ term
- the practical outer loop today is sweep-based keep/defer comparison, not an
  automated optimizer over $\\phi$

### What Must Be True To Build Different Paths

- To build a lightweight weighted scorer on top of the current benchmark
  contract:
  define one concrete metric for each term, define its direction and
  normalization, decide how missing values are handled, and store the resulting
  composite score plus its component breakdown in the registry or sweep row.
- To make that weighted scorer the canonical ranking rule:
  add an explicit objective-spec surface to the benchmark or sweep contract,
  update matrix or reporting code to rank by the composite score
  deterministically, and keep older runs interpretable when the score is not
  present.
- To build a real calibration term:
  freeze one calibration scalar for the lane instead of treating calibration as
  a generic guardrail category.
- To build a real stability term:
  freeze one stability scalar and collect enough repeatability evidence, such
  as reruns or multi-seed summaries, for that scalar to be meaningful.
- To build a real worst-family term:
  persist per-task or per-family benchmark metrics, define the family
  partition, and define the aggregation rule used to convert those retained
  metrics into one worst-case penalty.
- To build a true outer optimizer over $\\phi$ rather than a sweep-based
  decision process:
  make the admissible prior space machine-readable, make evaluation
  repeatable enough for noisy comparisons, and add search orchestration rather
  than relying on fixed manual sweep rows.

## Interpreting φ Through Grid Demands

The factorization of $\\phi$ matters because different parts of the prior alter
different parts of the grid computation.

| Prior block | Effect on the task distribution | Effect on grid demands |
| --- | --- | --- |
| $\\phi\_{\\mathrm{observation}}$ | changes $C$ and the distribution of $\\tau$ | changes cell-state dimensional burden, type-conditioned encoding burden, and effective token budget |
| $\\phi\_{\\mathrm{task}}$ | changes $N\_{\\mathrm{tr}}, N\_{\\mathrm{te}}$, class structure, and label noise | changes context size, query count, and classification difficulty |
| $\\phi\_{\\mathrm{shift}}$ | changes the relation between train and test laws | changes how hard row/column evidence sharing and test-row pooling are |
| $\\phi\_{\\mathrm{mechanism}}$ | changes interaction order, smoothness, saturation, and noise | changes how much structure row-wise and column-wise grid mixers must preserve |
| $\\phi\_{\\mathrm{layout}}$ | changes latent dependency structure and long-range coupling | changes how much multi-step or distributed structure the explicit grid must carry |
| $\\phi\_{\\mathrm{curriculum}}$ | changes which task families are emphasized during training | changes the gradient mixture seen during optimization |

Two especially important derived quantities are:

- $C$, because it controls the width of the observed table
- $R = N\_{\\mathrm{tr}} + N\_{\\mathrm{te}}$, because it controls the total row
  budget that the column mixers must share evidence across

At the mathematical level, `dagzoo` matters because it changes the law of
these quantities and the conditional structure coupling them to the labels.

## Repo Correspondence

This note is not part of the formulation itself, but it explains how the
symbols map back to the current repo:

- `dagzoo` is the implementation that parameterizes and samples from
  $p\_{\\phi}(T)$
- `grid_sandwich` is the active implementation of the map $f\_{\\theta}$
- `tabfoundry_sandwich` remains previous carried comparison evidence, not the
  current default map
- corpus ids, manifests, and loader objects are implementation artifacts used
  to materialize samples from $p\_{\\phi}(T)$; they are not part of the
  mathematical statement of the problem

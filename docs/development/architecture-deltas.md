# Architecture Deltas

This document compares the current benchmark-facing `tab-foundry` anchor to two
external reference points:

- `nanoTabPFN` / TabPFN-style cell-table encoder
- TabICLv2's row-first architecture

The goal is not to restate every implementation detail. It is to make the
structural deltas visible enough that future anchor work can choose a direction
deliberately instead of living in a hybrid middle state.

## Scope

The "current anchor" here means the active large-CUDA benchmark-facing surface
under investigation in
`reference/system_delta_sweeps/cuda_stability_followup/queue.yaml` row `1`:

- `stage=nano_exact`
- `module_overrides.table_block_style=prenorm`
- `module_overrides.row_pool=row_cls`
- `d_icl=512`
- `tficl_n_layers=12`
- `head_hidden_dim=1024`

Code landing zones:

- frozen TabPFN-like anchor:
  `src/tab_foundry/model/architectures/tabfoundry_simple.py`
- current staged anchor wiring:
  `src/tab_foundry/model/architectures/tabfoundry_staged/forward_common.py`
- current staged block and pooling implementations:
  `src/tab_foundry/model/architectures/tabfoundry_staged/subsystems.py`
- staged recipe and override surface:
  `src/tab_foundry/model/architectures/tabfoundry_staged/recipes.py`
  and `src/tab_foundry/model/architectures/tabfoundry_staged/resolved.py`
- parked TabICL-flavored reusable components:
  `src/tab_foundry/model/components/blocks.py` and
  `src/tab_foundry/model/components/qass.py`

## Current Anchor At A Glance

```mermaid
flowchart LR
    x[train/test table] --> fe[nano feature encoder]
    y[y_train] --> tc[mean-padded target conditioner]
    fe --> cat[concat into full cell table]
    tc --> cat
    cat --> blk[12 x prenorm cell blocks]
    blk --> pool[row CLS pool]
    pool --> head[binary direct head]

    classDef base fill:#eef5ff,stroke:#3b6ea8,color:#0f1f33;
    classDef delta fill:#fff1d6,stroke:#c67a00,color:#3d2a00;
    class blk,pool,head delta;
    class x,fe,y,tc,cat base;
```

The important point is that the current anchor still keeps the full cell table
alive through the main stack. It has changed the block math and the readout, but
it has not yet switched to a row-embedding-then-ICL architecture.

## Delta Vs TabPFN

Shared backbone traits:

- one feature value maps to one cell embedding on the exact path
- target values are conditioned into a target column before the main stack
- the main stack alternates feature-wise and row-wise interaction over the full
  cell table
- prediction still happens in one forward pass over train and test rows

Key structural deltas:

- TabPFN uses a post-norm cell block; the current anchor uses a prenorm cell
  block
- TabPFN reads predictions from the target column; the current anchor first
  collapses each row with `row_cls`
- TabPFN is architecturally monolithic; the current anchor lives inside a staged
  resolved surface with overrideable subsystems

```mermaid
flowchart LR
    subgraph TP[TabPFN / nanoTabPFN]
        tp_x[train/test table] --> tp_fe[feature encoder]
        tp_y[y_train] --> tp_tc[mean-padded target column]
        tp_fe --> tp_cat[full cell table]
        tp_tc --> tp_cat
        tp_cat --> tp_blk[post-norm cell blocks]
        tp_blk --> tp_read[target-column readout]
        tp_read --> tp_head[binary decoder]
    end

    subgraph TA[Current tabfoundry anchor]
        ta_x[train/test table] --> ta_fe[nano feature encoder]
        ta_y[y_train] --> ta_tc[mean-padded target conditioner]
        ta_fe --> ta_cat[full cell table]
        ta_tc --> ta_cat
        ta_cat --> ta_blk[prenorm cell blocks]
        ta_blk --> ta_pool[row CLS pool]
        ta_pool --> ta_head[binary direct head]
    end

    tp_cat -. same cell-table lineage .- ta_cat
    tp_blk -. block style changed .- ta_blk
    tp_read -. readout replaced .- ta_pool

    classDef shared fill:#eef5ff,stroke:#3b6ea8,color:#0f1f33;
    classDef delta fill:#fff1d6,stroke:#c67a00,color:#3d2a00;
    class tp_x,tp_fe,tp_y,tp_tc,tp_cat,ta_x,ta_fe,ta_y,ta_tc,ta_cat shared;
    class tp_blk,tp_read,tp_head,ta_blk,ta_pool,ta_head delta;
```

### What This Means

Relative to TabPFN, the current anchor is still in the same architectural
family. The main design question is therefore not "should it become more like a
transformer for tables?" It already is. The real question is whether the anchor
should stay on the cell-table path and become a cleaner TabPFN-derived model, or
stop paying cell-table costs and move to a row-first design.

## Delta Vs TabICLv2

TabICLv2's defining move is to compress the table into row embeddings before the
final ICL stage. The current anchor does not do that.

Key structural deltas:

- TabICLv2 has an explicit column-wise embedding stage based on set-style /
  induced attention; the current anchor does not use that on the live path
- TabICLv2 has a distinct row-interaction stage before final ICL; the current
  anchor keeps one main cell-table stack instead
- TabICLv2 performs the last stage of in-context learning on row embeddings and
  uses QASS-family attention there; the current anchor's live path has no active
  context encoder
- TabICLv2 is designed for classification and regression; the current anchor is
  still a binary direct benchmark surface

```mermaid
flowchart LR
    subgraph TA[Current tabfoundry anchor]
        ta_x[train/test table] --> ta_fe[nano feature encoder]
        ta_y[y_train] --> ta_tc[target conditioner]
        ta_fe --> ta_cat[full cell table]
        ta_tc --> ta_cat
        ta_cat --> ta_blk[prenorm cell stack]
        ta_blk --> ta_pool[row CLS pool]
        ta_pool --> ta_head[direct head]
    end

    subgraph TI[TabICLv2]
        ti_x[train/test table] --> ti_col[column-wise set embedding]
        ti_col --> ti_row[row transformer plus CLS]
        ti_row --> ti_rows[row embeddings]
        ti_y[y_train] --> ti_ctx[label injection at ICL stage]
        ti_rows --> ti_icl[final row-level ICL transformer]
        ti_ctx --> ti_icl
        ti_icl --> ti_head[classification or regression head]
    end

    ta_cat -. no early row collapse .- ti_rows
    ta_pool -. current path ends here .- ti_icl

    classDef anchor fill:#fff1d6,stroke:#c67a00,color:#3d2a00;
    classDef tabicl fill:#e9f8ef,stroke:#2d8a57,color:#123322;
    class ta_x,ta_fe,ta_y,ta_tc,ta_cat,ta_blk,ta_pool,ta_head anchor;
    class ti_x,ti_col,ti_row,ti_rows,ti_y,ti_ctx,ti_icl,ti_head tabicl;
```

### What This Means

Relative to TabICLv2, the current anchor is still missing the architectural move
that makes TabICLv2 scalable: "embed rows first, then do ICL on rows." The repo
does contain some ingredients for that direction in reusable components:

- `TFColEncoder` in `src/tab_foundry/model/components/blocks.py`
- `TFRowEncoder` in `src/tab_foundry/model/components/blocks.py`
- `QASSTransformerEncoder` in `src/tab_foundry/model/components/qass.py`

But those pieces are not yet the default live anchor path.

## Directional Read

```mermaid
flowchart TD
    cur[current anchor<br/>cell-table stack + prenorm + row CLS] --> pfn[PFN-cleanup path]
    cur --> hybrid[explicit post-pool ICL path]
    cur --> ticl[TabICLv2-style row-first path]

    pfn --> pfn1[keep cell-table core<br/>stabilize block math and readout]
    hybrid --> hy1[retain current table stack<br/>add a real row-level context stage]
    ticl --> ti1[promote column encoder plus row encoder<br/>make final ICL operate on rows]

    classDef neutral fill:#eef5ff,stroke:#3b6ea8,color:#0f1f33;
    classDef branch fill:#f7f7f7,stroke:#777,color:#222;
    class cur neutral;
    class pfn,pfn1,hybrid,hy1,ticl,ti1 branch;
```

The least coherent long-term state is to keep `row_cls` as a readout tweak while
still treating the full cell-table stack as the main architecture. That is a
hybrid worth benchmarking, but probably not a strong destination.

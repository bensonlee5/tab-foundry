# TF-RD-009 Upper-Family Selection

## Decision

- Selected continuation: `192x7 -> 208x8 -> 224x9 -> 248x10`
- Primary score: maximize D-optimal information gain on the current validation `L(N,S)` fit.
- Secondary score: minimize projected parameter-uncertainty width for `alpha_n` and `alpha_s`.
- Tie-breaks: fewer new geometries, then lower max predicted reserved VRAM.

## Current Validation L(N,S)

- `alpha_n = 0.030256524`
- `alpha_s = 0.331429581`
- `Nc = 258222763.013`
- `Sc = 608.501`
- `irreducible_loss = 0.000300470729`

## Candidate Scores

| Continuation | Rows | D-opt gain | alpha width | Max predicted reserved GB |
| --- | --- | ---: | ---: | ---: |
| `192x7->208x8->224x9->248x10` | `192x7 -> 208x8 -> 224x9 -> 248x10` | 2.904636 | 30.570378 | 40.481 |
| `200x7->224x8->256x9` | `200x7 -> 224x8 -> 256x9` | 2.481342 | 32.219487 | 39.652 |
| `216x7->272x8` | `216x7 -> 272x8` | 1.999156 | 34.213596 | 40.354 |

## Policy

- Run the selected rows first at the carried fixed-budget gate row.
- Promote only health=`ok` survivors into the full `{625,1250,2500,5000}` NS ladder.
- Keep health=`warn` rows as upper-family evidence only.
- Require a fresh one-row large-rung validation before replacing the frozen preferred RTX 8000 baseline.

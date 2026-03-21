# Dataset License Review Summary

This file summarizes the current repo-tracked real-data dataset approvals.

Primary source of truth:

- `reference/dataset_license_reviews.csv`

Current execution note:

- The current repo-tracked OpenML benchmark backfill approves datasets from
  OpenML license metadata unless contradictory evidence appears.

## Scope

- benchmark bundles reviewed:
  - `src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json`
  - `src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`
  - `src/tab_foundry/bench/nanotabpfn_openml_classification_small_v1.json`
  - `src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json`
  - `src/tab_foundry/bench/nanotabpfn_openml_binary_large_no_missing_v1.json`
- distinct OpenML datasets reviewed: 84
- manifest-backed external comparator sets reviewed: 0

## Result

- approved: 84
- rejected: 0
- manual_review_required: 0
- rows with original upstream URL present: 26
- rows without original upstream URL: 58
- rows whose OpenML `original_data_url` was non-URL text: 2

## License Counts

- `Public`: 70
- `CC BY 4.0`: 6
- `CC0: Public Domain`: 3
- `Public Domain`: 3
- `MIT License`: 1
- `public`: 1

## Notes

- Approval for this pass is based on OpenML `licence` metadata recorded in the ledger.
- `original_upstream_url` is included when OpenML exposed a usable URL; otherwise the ledger notes whether it was absent or non-URL text.
- No bundle-replacement follow-up issues were required during this backfill pass.

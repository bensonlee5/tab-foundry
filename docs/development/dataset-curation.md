# Dataset Curation And License Review

This document defines the repo-level acceptance gate for real-data datasets
used in `tab-foundry`.

Related surfaces:

- canonical roadmap: `docs/development/roadmap.md`
- operator workflow runbook: `docs/workflows.md`
- review ledger: `reference/dataset_license_reviews.csv`

## Policy

- No real-data dataset may be added to an OpenML benchmark bundle,
  manifest-backed external set, or other curated benchmark ladder until its
  license review is complete.
- Keep one canonical OpenML baseline where the benchmark tooling is already
  native, and use vetted manifest-backed external real-data datasets as
  augmentation lanes when they add regimes that OpenML does not cover cleanly.
- The source of truth for approval status is the repo-tracked review ledger at
  `reference/dataset_license_reviews.csv`.
- This gate applies to OpenML datasets and vetted external real-data sources
  such as Kaggle mirrors.
- `dagzoo` remains the synthetic-data generation lane. It is not an external
  real-data source, and its synthetic corpora do not replace the review
  requirement for any real-data comparator set.

## Review Record

Each dataset review record must include these fields:

- `dataset_key`
- `source_platform`
- `source_url`
- `original_upstream_url`
- `license_id_or_text`
- `license_url`
- `review_status`
- `reviewer`
- `review_date`
- `commercial_use_notes`
- `derivative_use_notes`
- `notes`

Allowed `review_status` values:

- `approved`
- `rejected`
- `manual_review_required`

## Baseline And Augmentation Model

- OpenML remains the canonical benchmark-backed baseline for curated real-data
  ladders because the benchmark bundle tooling is already OpenML-native.
- External real-data sources may be used in the same wave as manifest-backed
  augmentation sets when they add decision-relevant regimes that OpenML does
  not cover well enough.
- External real-data augmentations do not justify a new loader or downloader
  path. They should enter through the existing manifest-backed data surface.
- `dagzoo` comparisons should treat these curated real-data ladders as the
  comparator surface rather than as interchangeable source-ingestion work.

## Review Workflow

For every candidate dataset:

1. Read the platform license metadata.
1. For the current repo-tracked OpenML benchmark backfill, OpenML dataset
   metadata is sufficient unless contradictory evidence appears.
1. When an original upstream page or publisher terms URL is available, record
   it in the review ledger.
1. If the platform metadata and upstream terms disagree, reject the dataset or
   mark it `manual_review_required`.
1. For vetted external real-data sources such as Kaggle mirrors, do not rely on
   the platform page alone; verify the original upstream terms before approval.
1. Only datasets with `review_status=approved` may be promoted into curated
   bundles, manifests, or benchmark ladders.

## Admission Rules

Auto-approve only these clearly permissive licenses:

- `CC0-1.0`
- `PDDL-1.0`
- `CC-BY-4.0`
- `CC BY 4.0`
- `CC0: Public Domain`
- `Public Domain`
- `MIT License`
- OpenML `Public` and `public` metadata for the current repo-tracked benchmark
  backfill pass

Reject these cases:

- any `NC` or `ND` Creative Commons license
- `unknown`
- `other`
- `copyright-authors`
- blank or missing license fields
- custom terms that do not clearly permit both commercial use and
  modifications or derivative works
- terms such as `research only`, `academic use only`, `personal use only`,
  `no redistribution`, or `no modification`

Require manual review for:

- `CC-BY-SA-*`
- `ODbL-1.0`
- `DbCL-1.0`
- government or public-sector terms with conditions
- source-specific custom agreements

## Operational Notes

- Keep this gate outside runtime data loading in v1. Bundles and manifests
  should only reference datasets that are already cleared in the ledger.
- Do not add a new dataset loader or downloader path to enforce licensing.
  Real-data additions should continue to use the existing OpenML benchmark
  bundle path or the existing manifest-backed data path.
- Treat the OpenML benchmark ladder as the canonical closure surface when one
  exists, and use approved manifest-backed external datasets as augmentation
  evidence rather than as an unstructured replacement path.
- Existing benchmark bundles should be treated as pending license-review
  backfill until their datasets are added to the ledger with `approved`
  records.
- The current backfill issue for repo-tracked OpenML benchmark datasets may
  approve rows directly from OpenML metadata and should record whether an
  `original_data_url` was present, absent, or non-URL.
- This is an operational compliance filter, not legal advice. Escalate
  ambiguous or high-value commercial cases to counsel.

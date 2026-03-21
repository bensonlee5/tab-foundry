# Reference Index

`reference/` is the canonical landing zone for repo-local literature notes,
evidence maps, and adjacent-repo summaries that shape architecture and
benchmark decisions in `tab-foundry`.

Structural rules:

- start new literature or evidence material here instead of scattering it under
  `docs/`
- keep curated source summaries in `papers.md`
- keep roadmap-to-source justification and acceptance signals in `evidence.md`
- add future repo notes or issue maps as standalone Markdown files in this
  directory and link them from this index
- treat generated sweep queues and matrices as research-evidence surfaces that
  may reflect diagnostic or historical PFN-adjacent work rather than the
  normative architecture target

Contents:

- `papers.md`: curated paper list, adoption tiers, and borrowing rules for
  architecture and training ideas
- `evidence.md`: roadmap-to-reference mapping and per-epic evidence notes
- `dataset_license_reviews.csv`: review ledger for approving or rejecting
  real-data datasets before they enter curated bundles or manifests
- `dataset_license_review_summary.md`: current status summary for the repo's
  reviewed real-data datasets
- `system_delta_catalog.yaml`: reusable delta definitions for the active
  system-delta workflow
- `system_delta_campaign_template.md`: required research-package template for
  one queue row
- `stage_research_sources.yaml`: pinned repo-local, sibling-workspace, and
  external reference manifest for research packages
- `system_delta_sweeps/`: canonical sweep metadata, queue instances, and
  rendered matrices for research evidence; completed sweeps remain historical
  evidence even when the roadmap direction moves on
- `system_delta_queue.yaml`: generated active-sweep queue alias
- `system_delta_matrix.md`: generated active-sweep matrix alias

Keeping this material under one indexed home gives future architecture and
benchmark work a stable citation surface without mixing research notes into the
operator-facing docs. The live architecture source of truth still lives in
`docs/development/`.

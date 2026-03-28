# Reference Index

Start here when you need the papers, evidence notes, and supporting research
artifacts behind roadmap and architecture decisions in `tab-foundry`.

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
- `roadmap_evidence/`: canonical long-form evidence notes for TF-RD-018 and
  later research-oriented roadmap epics, indexed by roadmap order
- `dataset_license_reviews.csv`: review ledger for approving or rejecting
  real-data datasets before they enter curated bundles or manifests
- `dataset_license_review_summary.md`: current status summary for the repo's
  reviewed real-data datasets
- `system_delta_catalog.yaml`: reusable delta definitions for the
  system-delta workflow
- `system_delta_campaign_template.md`: required research-package template for
  one queue row
- `stage_research_sources.yaml`: pinned repo-local, sibling-workspace, and
  external reference manifest for research packages
- `system_delta_sweeps/`: canonical sweep metadata, queue instances, and
  rendered matrices for research evidence; completed sweeps remain historical
  evidence even when the roadmap direction moves on

## Research Epic Notes

Long-form research evidence for the current epic and later research-oriented
roadmap items now lives under `reference/roadmap_evidence/`. `papers.md`
remains the shared bibliography, and `evidence.md` remains the compact
cross-epic map.

- [`roadmap_evidence/README.md`](roadmap_evidence/README.md): conventions and
  roadmap-order index for per-epic evidence notes
- [`roadmap_evidence/tf_rd_018_training_surface_adequacy.md`](roadmap_evidence/tf_rd_018_training_surface_adequacy.md):
  canonical TF-RD-018 batch-size, LR, optimizer, and training-surface note
- [`roadmap_evidence/tf_rd_020_harder_dagzoo_corpus_fronts.md`](roadmap_evidence/tf_rd_020_harder_dagzoo_corpus_fronts.md):
  canonical TF-RD-020 harder-front handoff note on the historical
  staged-control line
- [`roadmap_evidence/tf_rd_021_steering_derived_dagzoo_corpus_fronts.md`](roadmap_evidence/tf_rd_021_steering_derived_dagzoo_corpus_fronts.md):
  canonical TF-RD-021 steering-derived synthetic follow-on note on the carried
  sandwich dagzoo slice
- [`roadmap_evidence/tf_rd_022_training_runtime_vram_efficiency.md`](roadmap_evidence/tf_rd_022_training_runtime_vram_efficiency.md):
  canonical TF-RD-022 runtime-and-VRAM efficiency note for the carried
  sandwich classification family before scaling
- [`roadmap_evidence/tf_rd_021a_latent_bank_sandwich_prototype.md`](roadmap_evidence/tf_rd_021a_latent_bank_sandwich_prototype.md):
  fixed-latent sandwich candidate note and closed immediate nanoTabPFN screen
- [`roadmap_evidence/tf_rd_021b_hybrid_full_cell_sandwich_successor.md`](roadmap_evidence/tf_rd_021b_hybrid_full_cell_sandwich_successor.md):
  hybrid full-cell sandwich successor note and simplified-parent handoff
- [`roadmap_evidence/tf_rd_010_many_class_promotion.md`](roadmap_evidence/tf_rd_010_many_class_promotion.md):
  canonical TF-RD-010 carried-slice note for the first sandwich many-class plus
  missingness gate
- [`roadmap_evidence/tf_rd_009_scaling_law_measurement.md`](roadmap_evidence/tf_rd_009_scaling_law_measurement.md):
  canonical TF-RD-009 scaling-specific note and classification-first sandwich
  handoff target

Keeping this material under one indexed home gives architecture and benchmark
work a stable citation surface without mixing research notes into the
operator-facing docs. The live architecture source of truth still lives in
`docs/development/`.

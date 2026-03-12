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

Contents:

- `papers.md`: curated paper list, adoption tiers, and borrowing rules for
  architecture and training ideas
- `evidence.md`: roadmap-to-reference mapping and per-epic evidence notes

Keeping this material under one indexed home gives future architecture and
benchmark work a stable citation surface without mixing research notes into the
operator-facing docs.

# TF-RD-014: Missingness Robustness On The Promoted Anchor

This is the canonical long-form evidence note for
[TF-RD-014](../../docs/development/roadmap.md#tf-rd-014-missingness-robustness-on-the-promoted-anchor).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows [TF-RD-018](tf_rd_018_training_surface_adequacy.md)
  as one of the first preferred harder benchmark-backed ladders and feeds into
  [TF-RD-016](tf_rd_016_architecture_surface_adequacy.md)

## External Evidence

- Shared bibliography: [reference/papers.md](../papers.md)
- Current curated context is still shared architecture context rather than a
  dedicated missingness bibliography: `TabICLv2`, `TabPFN`, `FT-Transformer`,
  and the repo's set-structured modeling references
- Dedicated missingness-robustness literature is not yet curated in this repo
- External evidence to curate next: explicit missing-token or missing-mask
  handling papers, synthetic missingness training studies, and benchmark-policy
  references for mixed missingness regimes

## Repo-Local Evidence

- `missingness_followup` already exists, but it is anchored on the older
  stabilized prenorm hybrid surface rather than the promoted row-first base
- the repo already maintains separate no-missing and allow-missing benchmark
  bundle contracts
- TF-RD-008 settled the row-first default on the allow-missing benchmark
  surface, but there is still no explicit row-first missingness recommendation

## Current Interpretation

- re-anchor missingness work on the promoted row-first base rather than
  reopening hybrid diagnostic lines
- keep one pinned OpenML missingness ladder as the canonical benchmark surface
  and use license-cleared manifest-backed external datasets only when they add
  missingness regimes OpenML does not cover cleanly
- separate missingness-mechanism adequacy from synthetic missingness training
  and from benchmark-surface evaluation

## Open Evidence Gaps

- the repo still lacks a benchmark-backed missingness recommendation for the
  row-first family
- there is no curated missingness-specific literature set yet
- review-ledger coverage for any future external missingness augmentations is
  still prospective rather than settled

## Exit Signals

- the repo has a benchmark-backed missingness recommendation for the promoted
  row-first family
- regime identity remains explicit in task-source, bundle, manifest, and
  curation artifacts instead of being hidden inside benchmark schema changes

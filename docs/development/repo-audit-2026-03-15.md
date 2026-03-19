# Repo Audit — 2026-03-15

Historical snapshot: this document records an audit that was run against branch
`codex/system-dimension-delta-sweep` on 2026-03-15, not against the current
branch tip. Current-state repo truth should come from the live docs and code,
not from this dated snapshot.

Audit method:

- `./.venv/bin/python -m pytest -q`
- `uv run pre-commit run --all-files`
- `./.venv/bin/python scripts/audit/check_repo_paths.py`
- `./.venv/bin/python scripts/audit/check_markdown_links.py`
- `./.venv/bin/python scripts/audit/module_graph.py --fail-on-doc-drift`
- `uv run deptry .`
- `uv run vulture src scripts tests --min-confidence 80`
- targeted `rg`/`sed` inspection for CLI/doc references and
  orphan-surface candidates

## Status Summary

The original 2026-03-15 audit findings were addressed on the audited branch:

- `docs/development/codebase-navigation.md` was updated to reflect that
  branch's then-live model surface plus the `preprocessing` and `research`
  package areas.
- `docs/development/module-dependency-map.md` now separates observed graph
  facts from dependency-direction policy and is guarded by the module-graph
  audit script.
- Benchmark-only third-party libraries are now modeled as optional install
  extras instead of being present only through the dev group by accident.
- The two identified orphan candidates have been removed, and the repeated
  unused benchmark-test lambda parameters are gone.

No `P0`, `P1`, or `P2` cleanup findings remain open from the original audit.

## Surface Classification

| Surface | Classification | Notes |
| ---- | ---- | ---- |
| `src/tab_foundry/`, `configs/`, packaged CLI, and shell/audit scripts | canonical and live | Includes live `preprocessing/`, `research/`, and nested CLI package surfaces in addition to data/model/training/export. |
| `README.md`, `docs/workflows.md`, `docs/inference.md`, `program.md` | canonical and live | Operator-facing docs and the active staged-research contract. |
| `reference/system_delta_catalog.yaml`, `reference/system_delta_sweeps/` | canonical and live | Multi-sweep system-delta source of truth. |
| `reference/system_delta_queue.yaml`, `reference/system_delta_matrix.md` | generated alias | Active-sweep compatibility views, not canonical editable sources. |
| benchmark registries, frozen bundles, export fixtures | historical/immutable | Accuracy-only surfaces unless they are still presented as canonical. |

## Residual Watchpoints

These are not current cleanup findings, but they remain worth watching:

- The observed package graph is still a hand-maintained doc surface, even
  though it now has a drift guardrail. Any future top-level package addition
  should update `docs/development/module-dependency-map.md` in the same PR.
- The benchmark/install boundary is explicit now, but it still spans optional
  surfaces under `src/tab_foundry/bench/`. New benchmark-facing imports should
  be reviewed against the `benchmark` extra instead of drifting back into
  “dev-only by accident.”
- The active system-delta alias files are intentionally duplicated by design.
  They should continue to be treated as generated views rather than canonical
  sources.

## Not A Finding

- The wrapper-surface reset is intentional: packaged nested CLI commands now
  own workflow entrypoints, while `scripts/` is limited to shell helpers and
  audit tooling.
- The audited branch intentionally kept three model families at that time:
  `tabfoundry`, `tabfoundry_simple`, and `tabfoundry_staged`.
- The current repo has since removed legacy `tabfoundry`; see the live model
  docs for the current staged/simple split.
- The current top-level import graph has no cycle candidates according to
  `scripts/audit/module_graph.py`.
- Repo-path and local Markdown-link hygiene are clean according to
  `scripts/audit/check_repo_paths.py` and
  `scripts/audit/check_markdown_links.py`.

## Doc Status Matrix

| Document | Status | Notes |
| ---- | ---- | ---- |
| `README.md` | accurate | Setup now distinguishes repo-local `uv sync` from minimal installs with optional extras. |
| `docs/workflows.md` | accurate on the audited branch | This row describes the 2026-03-15 snapshot, not the current branch by itself. |
| `docs/inference.md` | accurate | Schema, producer commands, and compatibility notes match the current export contract. |
| `program.md` | accurate | Correctly describes the active-sweep model and generated alias behavior. |
| `docs/development/design-decisions.md` | accurate but incomplete | Directional policy is sound; current-state package graph lives in the dependency-map doc instead. |
| `docs/development/codebase-navigation.md` | accurate on the audited branch | Reflected the audited branch's then-live package layout and model-family split. |
| `docs/development/module-dependency-map.md` | accurate | Observed graph and policy are now separated and checked by tooling. |
| `docs/development/model-architecture.md` | accurate on the audited branch | Captured the audited branch's model-family state at that time. |
| `docs/development/model-config.md` | accurate | Reflects `stage_label` and `module_overrides`. |
| `docs/development/roadmap.md` | accurate but incomplete | Planning state looks current; dependency-direction details live in the dedicated dependency-map doc. |
| `reference/README.md` | accurate | Correctly indexes active system-delta surfaces and generated aliases. |
| `reference/system_delta_queue.yaml` | generated alias, not canonical | Active-sweep compatibility view. |
| `reference/system_delta_matrix.md` | generated alias, not canonical | Active-sweep compatibility view. |

## Duplicate / Dead-Surface Table

| Surface | Current assessment | Recommended action |
| ---- | ---- | ---- |
| packaged nested CLI entrypoints into `bench/` and `research/` | intentional orchestration indirection | keep |
| three-family audited model split | historical snapshot only | keep this doc as historical context, but use the live docs for the current staged/simple split |
| `reference/system_delta_queue.yaml`, `reference/system_delta_matrix.md` | intentional generated aliases | keep |
| benchmark-only third-party packages under `src/tab_foundry/bench/` | optional install surface | keep, but extend the `benchmark` extra when new imports are added |

## Cleanup Backlog

### PR 1 — Optional graph-doc automation

Scope:

- teach `scripts/audit/module_graph.py` to emit a copy-paste-ready observed
  graph block or update the block automatically in a dry-run/format mode

Risk: low

Payoff: medium

Independent: yes

### PR 2 — Benchmark dependency review on new helpers

Scope:

- add a lightweight checklist or test around new `bench/` imports so optional
  dependency ownership stays explicit when new benchmark helpers land

Risk: low

Payoff: medium

Independent: yes

### PR 3 — Optional CI smoke for audit scripts

Scope:

- decide whether the repo-path, link, and module-graph drift checks should run
  in CI as explicit smoke checks instead of only via local verification

Risk: low

Payoff: medium

Independent: yes

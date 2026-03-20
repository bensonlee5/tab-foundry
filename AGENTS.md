# Development Patterns

- Use `.venv/` for commands and tests in this repo.
- For fast agent triage, prefer the existing narrow inspection surfaces before broad greps or full-suite runs: `./scripts/dev verify paths <PATH...>`, `tab-foundry dev resolve-config ...`, `tab-foundry dev forward-check ...`, `tab-foundry dev diff-config ...`, `tab-foundry dev export-check --checkpoint ...`, `tab-foundry data manifest-inspect --manifest ...`, `tab-foundry dev run-inspect --run-dir ...`, and `tab-foundry research sweep inspect|diff ...`.
- Prefer breaking dependency cycles and centralizing shared wiring in the existing role-based library modules under `src/tab_foundry/`; avoid "legacy" pathways, duplicate pathways, and shims. Do not introduce parallel implementations of the same logic in different layers of the codebase.
- We optimize for iteration speed: internal Python APIs and internal config structure may change without backward-compat guarantees.
- If CLI flags, persisted metadata schema, or dataset artifact contract changes, treat it as a user-facing break and call it out explicitly.
- For behavior/schema changes under `src/tab_foundry`, bump version in `pyproject.toml` just before merging into main so that the version reflects the latest changes (patch by default; minor for intentionally broad user-facing breaks). Docs/tests-only changes do not require a bump.
- On every version bump, update `CHANGELOG.md` in the same PR.
- If a PR resolves a GitHub issue, include explicit closing syntax such as `Closes #74` in the PR body so the issue is linked and auto-closes on merge.
- We prefer shared utility packages over hand-rolled helpers to keep invariants centralized
- We don’t probe data “YOLO-style”—we validate boundaries or rely on typed SDKs
- Prior to declaring a branch ready for review, compare branch to main and verify that all intended changes are included and no unintended changes are included
- Always log results to wandb when executing sweeps

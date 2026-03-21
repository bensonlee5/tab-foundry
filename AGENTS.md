# Development Patterns

- Use `.venv/` for commands and tests in this repo.
- Treat `./scripts/dev` as the fast path for repo-local bootstrap, verification, and Iris smoke only.
- For anything else, discover the packaged CLI via `.venv/bin/tab-foundry --help`, `.venv/bin/tab-foundry <group> --help`, and `.venv/bin/tab-foundry <group> <command> --help`.
- For fast agent triage, prefer the existing narrow inspection surfaces before broad greps or full-suite runs: `tab-foundry dev resolve-config ...`, `tab-foundry dev forward-check ...`, `tab-foundry dev diff-config ...`, `tab-foundry dev export-check --checkpoint ...`, `tab-foundry data manifest-inspect --manifest ...`, `tab-foundry dev run-inspect --run-dir ...`, `tab-foundry research sweep inspect ...`, `tab-foundry research sweep diff ...`, and `./scripts/dev verify paths <PATH...>`.
- Only fall back to broader greps, full verification, or codebase-wide exploration after those surfaces do not answer the question.
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
- Update roadmap.md for sweeps when a sweep is complete, and also update associated GitHub issues

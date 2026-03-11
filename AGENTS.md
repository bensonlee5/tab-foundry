# tab-foundry Agent Guide

## Environment

- Python `3.13` from `.python-version`
- Package and task runner: `uv`
- Bootstrap a fresh clone with `uv sync --frozen`
- Git remote uses SSH: `git@github.com:bensonlee5/tab-foundry.git`

## Validation

Run targeted checks while iterating, then run the full gate before push or review handoff:

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## Repo Rules

- Keep changes narrowly scoped to the ticket.
- Prefer smoke experiments and targeted validation over workstation-scale training unless the task explicitly requires heavier runs.
- `DAGZOO_DATA_ROOT` is optional. Only treat it as required when the task needs real dataset-backed commands.
- Do not commit generated artifacts or local runtime outputs from `data/`, `outputs/`, `wandb/`, caches, or local env files.
- If you change export/runtime contract behavior, update `docs/INFERENCE_CONTRACT.md` in the same change.
- Training and evaluation commands may write outputs under `outputs/`; those outputs are not source files and should not be committed.
- If Muon is unavailable and the task only needs a smoke proof, use the documented override `optimizer=adamw`.

## Git and PR Expectations

- Start by syncing with `origin/main` before edits on long-lived branches.
- Keep PR descriptions concise and concrete; this repo does not require a PR template.
- Address both top-level and inline PR feedback before moving work to review-complete states.

## Secrets and External Access

- Never read secrets from home-directory files from inside repo automation.
- Assume required secrets are already present in environment variables provided by the launcher.
- If required auth is missing, record the exact failing command and treat it as a blocker instead of inventing a workaround.

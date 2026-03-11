#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to bootstrap tab-foundry workspaces." >&2
  exit 1
fi

uv sync --frozen

if [[ -z "${DAGZOO_DATA_ROOT:-}" ]]; then
  echo "Note: DAGZOO_DATA_ROOT is unset. This is fine unless the ticket needs dataset-backed commands." >&2
fi

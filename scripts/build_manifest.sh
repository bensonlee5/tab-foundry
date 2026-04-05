#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${DAGZOO_DATA_ROOT:-}}"
OUT="${2:-data/manifests/default.parquet}"

if [[ -z "$ROOT" ]]; then
  echo "usage: $0 <dagzoo-data-root> [out-manifest]" >&2
  echo "or set DAGZOO_DATA_ROOT to an explicit dagzoo data directory" >&2
  exit 1
fi

uv run tab-foundry dev data build-manifest \
  --data-root "$ROOT" \
  --out-manifest "$OUT"

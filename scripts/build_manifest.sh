#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${DAGZOO_DATA_ROOT:-$HOME/dev/dagzoo/data}}"
OUT="${2:-data/manifests/default.parquet}"

uv run tab-foundry data build-manifest \
  --data-root "$ROOT" \
  --out-manifest "$OUT"

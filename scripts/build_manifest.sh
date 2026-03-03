#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${CAUCHY_DATA_ROOT:-$HOME/dev/cauchy-generator/data}}"
OUT="${2:-data/manifests/default.parquet}"

uv run tab-foundry build-manifest \
  --data-root "$ROOT" \
  --out-manifest "$OUT"

#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:?checkpoint path required}"
EXP="${2:-cls_smoke}"

uv run tab-foundry eval --checkpoint "$CKPT" "experiment=${EXP}"

#!/usr/bin/env bash
set -euo pipefail

EXP="${1:-cls_smoke}"

uv run tab-foundry train "experiment=${EXP}"

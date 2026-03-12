#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-main}"
shift || true

if [ "$#" -gt 0 ]; then
  CHECK_CONTEXTS=("$@")
else
  CHECK_CONTEXTS=("quality-and-unit" "iris-smoke")
fi

origin_url="$(git config --get remote.origin.url)"
if [ -z "${origin_url}" ]; then
  echo "remote.origin.url is not configured" >&2
  exit 1
fi

case "${origin_url}" in
  git@github.com:*)
    repo="${origin_url#git@github.com:}"
    ;;
  https://github.com/*)
    repo="${origin_url#https://github.com/}"
    ;;
  *)
    echo "unsupported GitHub remote URL: ${origin_url}" >&2
    exit 1
    ;;
esac
repo="${repo%.git}"

payload_file="$(mktemp)"
trap 'rm -f "${payload_file}"' EXIT

python3 - "${BRANCH}" "${CHECK_CONTEXTS[@]}" > "${payload_file}" <<'PY'
import json
import sys

branch = sys.argv[1]
contexts = list(sys.argv[2:])
payload = {
    "required_status_checks": {
        "strict": True,
        "contexts": contexts,
    },
    "enforce_admins": False,
    "required_pull_request_reviews": None,
    "restrictions": None,
    "allow_force_pushes": False,
    "allow_deletions": False,
    "block_creations": False,
    "required_conversation_resolution": False,
    "lock_branch": False,
    "allow_fork_syncing": True,
}
json.dump(payload, sys.stdout)
PY

echo "Configuring protection for ${repo}:${BRANCH}"
echo "Required checks: ${CHECK_CONTEXTS[*]}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  cat "${payload_file}"
  echo
  exit 0
fi

gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "repos/${repo}/branches/${BRANCH}/protection" \
  --input "${payload_file}" >/dev/null

echo "Branch protection updated."

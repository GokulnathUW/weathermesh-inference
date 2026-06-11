#!/usr/bin/env bash
# Cron wrapper: load .env, use project Python, append all output to pipeline.log.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$REPO_ROOT/outputs/pipeline.log"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/outputs"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.venv/bin/activate"
else
  export PATH="$HOME/.local/bin:/usr/bin:$PATH"
fi

if [[ ! -f "$REPO_ROOT/.env" ]]; then
  echo "ERROR: missing $REPO_ROOT/.env" >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "$REPO_ROOT/.env"
set +a

exec >> "$LOG_FILE" 2>&1
python3 "$REPO_ROOT/pipeline.py"

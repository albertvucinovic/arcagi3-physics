#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$ROOT/.." && pwd)
PYTHON=${PYTHON:-"$PROJECT_ROOT/venv/bin/python"}

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing ARC Python environment: $PYTHON" >&2
  exit 1
fi
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.env"
  set +a
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m leaderboard_submission "$@"

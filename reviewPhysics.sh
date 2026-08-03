#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-"$ROOT/venv/bin/python"}

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing ARC Python environment: $PYTHON" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import arcagi3_physics, eggthreads' >/dev/null 2>&1; then
  echo "Missing ARC Physics dependencies in $PYTHON." >&2
  echo "Install this project first:" >&2
  echo "  $PYTHON -m pip install ." >&2
  exit 1
fi

cd "$ROOT"
exec "$PYTHON" -m arcagi3_physics.review \
  --run-dir "${ARC_RUN_DIR:-$ROOT/runs/physics-ls20-seed0}" "$@"

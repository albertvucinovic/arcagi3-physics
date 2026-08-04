#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-"$ROOT/venv/bin/python"}

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing ARC Python environment: $PYTHON" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import arc_agi, eggopt, eggthreads' >/dev/null 2>&1; then
  echo "Missing ARC Physics dependencies in $PYTHON." >&2
  echo "Install this project first:" >&2
  echo "  $PYTHON -m pip install ." >&2
  exit 1
fi
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT/.env"
  set +a
fi

cd "$ROOT"
args=(
  -m arcagi3_physics.benchmark
  --environments-dir "${ARC_ENVIRONMENTS_DIR:-$ROOT/environment_files}"
  --run-dir "${ARC_BENCHMARK_RUN_DIR:-$ROOT/runs/luna-public-benchmark}"
  --actor-model "${ARC_ACTOR_MODEL:-Pro: GPT-5.6 Luna max}"
  --seed "${ARC_SEED:-0}"
  --max-parallel "${ARC_MAX_PARALLEL:-8}"
  --max-actions "${ARC_MAX_ACTIONS:-500}"
  --max-cycles "${ARC_MAX_CYCLES:-100}"
  --actor-context-limit "${ARC_ACTOR_CONTEXT_LIMIT:-3000000}"
  --default-search-depth "${ARC_DEFAULT_SEARCH_DEPTH:-12}"
  --default-max-nodes "${ARC_DEFAULT_MAX_NODES:-10000}"
  --critic-timeout "${ARC_CRITIC_TIMEOUT:-300}"
)
if [[ $# -gt 0 ]]; then
  args+=("$@")
fi
exec "$PYTHON" "${args[@]}"

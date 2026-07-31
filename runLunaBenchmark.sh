#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EGG_MONO=${EGG_MONO:-/home/albert/Private/Projekti/ai/egg/egg-mono}
PYTHON=${PYTHON:-"$ROOT/venv/bin/python"}

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing ARC Python environment: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$EGG_MONO/eggopt" ]]; then
  echo "Missing egg-mono checkout: $EGG_MONO" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import aiohttp' >/dev/null 2>&1; then
  echo "Missing aiohttp in $PYTHON." >&2
  echo "Install the runner dependencies with:" >&2
  echo "  $PYTHON -m pip install 'aiohttp>=3.9'" >&2
  exit 1
fi
if [[ -f "$EGG_MONO/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$EGG_MONO/.env"
  set +a
fi

export PYTHONPATH="$ROOT:$EGG_MONO/eggopt:$EGG_MONO/eggflow:$EGG_MONO/eggthreads:$EGG_MONO/eggconfig:$EGG_MONO/eggllm${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
args=(
  -m arcagi3_physics.benchmark
  --environments-dir "${ARC_ENVIRONMENTS_DIR:-$ROOT/environment_files}"
  --run-dir "${ARC_BENCHMARK_RUN_DIR:-$ROOT/runs/luna-public-benchmark}"
  --actor-model "${ARC_ACTOR_MODEL:-Pro: GPT-5.6 Luna max}"
  --seed "${ARC_SEED:-0}"
  --max-parallel "${ARC_MAX_PARALLEL:-3}"
  --max-actions "${ARC_MAX_ACTIONS:-50}"
  --max-cycles "${ARC_MAX_CYCLES:-100}"
  --actor-context-limit "${ARC_ACTOR_CONTEXT_LIMIT:-300000}"
  --max-plan-depth "${ARC_MAX_PLAN_DEPTH:-8}"
  --max-plan-nodes "${ARC_MAX_PLAN_NODES:-10000}"
  --critic-timeout "${ARC_CRITIC_TIMEOUT:-300}"
)
if [[ $# -gt 0 ]]; then
  args+=("$@")
fi
exec "$PYTHON" "${args[@]}"

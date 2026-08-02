#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EGG_MONO=${EGG_MONO:-/home/albert/Private/Projekti/ai/egg/egg-mono}
PYTHON=${PYTHON:-"$ROOT/venv/bin/python"}

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing ARC Python environment: $PYTHON" >&2
  echo "Create $ROOT/venv with Python 3.12 and install arc-agi first." >&2
  exit 1
fi
if [[ ! -d "$EGG_MONO/eggopt" ]]; then
  echo "Missing egg-mono checkout: $EGG_MONO" >&2
  echo "Set EGG_MONO=/path/to/egg-mono." >&2
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
  -m arcagi3_physics.run
  --game "${ARC_GAME:-ls20}"
  --seed "${ARC_SEED:-0}"
  --environments-dir "${ARC_ENVIRONMENTS_DIR:-$ROOT/environment_files}"
  --run-dir "${ARC_RUN_DIR:-$ROOT/runs/physics-ls20-seed0}"
  --actor-model "${ARC_ACTOR_MODEL:-Pro: GPT-5.6 Sol max}"
  --max-actions "${ARC_MAX_ACTIONS:-500}"
  --max-cycles "${ARC_MAX_CYCLES:-100}"
  --actor-context-limit "${ARC_ACTOR_CONTEXT_LIMIT:-0}"
  --max-plan-depth "${ARC_MAX_PLAN_DEPTH:-12}"
  --max-plan-nodes "${ARC_MAX_PLAN_NODES:-10000}"
  --critic-timeout "${ARC_CRITIC_TIMEOUT:-300}"
)
if [[ $# -gt 0 ]]; then
  args+=("$@")
fi
exec "$PYTHON" "${args[@]}"

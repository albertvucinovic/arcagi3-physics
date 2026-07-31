#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EGG_MONO=${EGG_MONO:-/home/albert/Private/Projekti/ai/egg/egg-mono}
PYTHON=${PYTHON:-"$ROOT/venv/bin/python"}

export PYTHONPATH="$ROOT:$EGG_MONO/eggopt:$EGG_MONO/eggflow:$EGG_MONO/eggthreads:$EGG_MONO/eggconfig:$EGG_MONO/eggllm${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON" -m arcagi3_physics.review \
  --run-dir "${ARC_RUN_DIR:-$ROOT/runs/physics-ls20-seed0}" "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-"$ROOT/venv/bin/python"}

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing ARC Python environment: $PYTHON" >&2
  echo "Create it with Python 3.12+, then install this project." >&2
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
environments_dir=${ARC_ENVIRONMENTS_DIR:-$ROOT/environment_files}
leaderboard=${ARC_LEADERBOARD:-$ROOT/leaderboard-submission/leaderboard.sh}
explicit_game=false
for argument in "$@"; do
  if [[ $argument == --game || $argument == --game=* ]]; then
    explicit_game=true
    break
  fi
done
if [[ -n ${ARC_GAME:-} ]]; then
  game=$ARC_GAME
elif [[ $explicit_game == true ]]; then
  game=ls20
else
  game=$(
    "$leaderboard" \
      --api-key "${ARC_API_KEY:-}" \
      --base-url "${ARC_BASE_URL:-https://three.arcprize.org}" \
      current-game ls20 \
      --environments-dir "$environments_dir"
  )
  echo "ARC API current game: $game"
fi
args=(
  -m arcagi3_physics.run
  --game "$game"
  --seed "${ARC_SEED:-0}"
  --environments-dir "$environments_dir"
  --run-dir "${ARC_RUN_DIR:-$ROOT/runs/physics-${game}-astar}"
  --actor-model "${ARC_ACTOR_MODEL:-Pro: GPT-5.6 Sol max}"
  --max-actions "${ARC_MAX_ACTIONS:-1000}"
  --max-cycles "${ARC_MAX_CYCLES:-200}"
  --actor-context-limit "${ARC_ACTOR_CONTEXT_LIMIT:-0}"
  --default-search-depth "${ARC_DEFAULT_SEARCH_DEPTH:-12}"
  --default-max-nodes "${ARC_DEFAULT_MAX_NODES:-10000}"
  --critic-timeout "${ARC_CRITIC_TIMEOUT:-300}"
)
if [[ $# -gt 0 ]]; then
  args+=("$@")
fi
exec "$PYTHON" "${args[@]}"

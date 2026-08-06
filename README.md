# ARC-AGI-3 Physics domain

This repository is the thin ARC adapter for Eggopt's complete Git-backed
`PhysicsStrategy`. Eggopt owns the ActorCritic loop, Git repositories, Timeline,
`step_<suffix>` / `reward_<suffix>` convention, all-model backtesting, goal and
discrimination planning, Actor instruments, committed-plan validation, trusted
Critic-thread sandbox execution, and execute-until-resolution semantics.

ARC supplies only:

- `FrameDataRaw` → finite public state conversion;
- initial environment creation/reset;
- real action submission;
- one live offline session during normal play;
- reset plus verified Timeline replay only after session loss;
- trusted ARC win detection;
- ARC-specific Actor guidance and CLI composition.

## Installation

Prerequisites:

- Python 3.12 or newer (required by the ARC-AGI-3 toolkit);
- Git (used both by pip for the Eggmono dependency and by PhysicsStrategy);
- Docker (used for isolated Actor and trusted Critic tool execution).

From a clone of this repository:

```bash
python3.12 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install .
```

This installs `arc-agi` and a reproducible, commit-pinned Eggmono release
directly from
[`albertvucinovic/egg-mono`](https://github.com/albertvucinovic/egg-mono).
No sibling Eggmono checkout and no hand-written `PYTHONPATH` are required.

Eggmono's component distributions remain independently installable through
their existing Git subdirectories (`eggllm`, `eggconfig`, `eggthreads`,
`eggflow`, and `eggopt`); the root package is an additional convenience, not a
replacement for that interface.

Configure the provider key used by your selected Actor model, for example in a
private `.env` file:

```bash
export OPENAI_API_KEY=...
```

The launchers source this repository's `.env` when present. Never commit it.
Downloaded ARC environments are expected under `environment_files/` by
default and are intentionally ignored by Git.

Runtime topology:

```text
Physics
└── Critic                    trusted Eggthread sandbox and environment authority
    └── Actor                 persistent theorist/planner Git workspace
```

The Actor repository is `<run-dir>/workspace/innerContext`; the Critic's pulled
history is `<run-dir>/workspace/critic-repository`.

ARC adds `gridToPng.py` to each Actor repository. It renders a 2-D ARC color
grid, public state, or the latest state in `canonical-input.json` to a PNG. The
Actor can then use its image-only `add_local_file_to_model_context` tool to view
that PNG in the next model turn.

The ARC Physics Actor allowlist also includes `bash`, `python_exec`,
`python_repl`, `answer_user_while_preserving_llm_turn`,
`read_long_tool_output`, `skill`, and `tool_help`. Generic Physics instructions
ask the Actor to use interim Assistant Notes as a visible lab notebook while it
continues working; only a later plain assistant answer ends and submits the turn.

Run or resume:

```bash
./runPhysics.sh
```

With no `ARC_GAME` override, the launcher asks the official API for the current
`ls20` version, verifies that exact version exists in `environment_files`, and
uses a versioned run directory. It fails before starting an expensive run if the
API and local files do not match. Synchronize first when instructed:

```bash
./leaderboard-submission/leaderboard.sh environments \
  --environments-dir environment_files --sync
```

Only set `ARC_GAME` (or pass `--game`) when intentionally selecting another
base game or an older exact version, for example
`ARC_GAME=ls20-9607627b ./runPhysics.sh`. Either explicit override deliberately
skips API-current-version selection.

Equivalent installed entry point:

```bash
arcagi3-physics --game ls20 \
  --environments-dir environment_files \
  --run-dir runs/physics-ls20-seed0
```

Review the current or completed run from the Critic's authoritative Git copy:

```bash
./reviewPhysics.sh
```

The `arcagi3-physics-review` and `arcagi3-physics-benchmark` entry points are
also installed. See [LUNA-BENCHMARK.md](LUNA-BENCHMARK.md) for the public-suite
benchmark workflow.

The terminal viewer starts at the latest public observation. Use left/right
arrows (or `h`/`l`) to traverse the append-only Timeline, Home/End to jump, `r`
to reload a run that is still progressing, and `q` to quit. Its textual stats
follow the selected action rather than always describing the latest report. They
identify the Actor turn that produced the action, that turn's proposed and
executed plan lengths, the action's position within the executed prefix and its
prediction result, plus historical Actor/Critic, evaluation, and model totals as
of that action. For logs or non-interactive inspection, print one frame with:

```bash
./reviewPhysics.sh --frame 0
```

Scoring and scorecard submission are intentionally separate from the solver.
All local RHAE reporting, official environment synchronization, Competition
Mode trajectory gathering/replay, exact current environment lists, and the
complete Community Leaderboard procedure live in:

```text
leaderboard-submission/
```

Start with [leaderboard-submission/README.md](leaderboard-submission/README.md).
In particular, completed independent `runPhysics.sh` runs can be listed together
for one scorecard, provided they cover every exact version currently advertised
by the API. The submission tooling preflights coverage without opening the
one-shot scorecard. None of this code or score data is exposed to the Actor.

Animated GIF export remains a normal review feature:

```bash
./reviewPhysics.sh --gif runs/physics-ls20-seed0/played.gif
```

Use `--gif-scale`, `--gif-duration-ms`, and `--gif-level-pause-ms` to control
nearest-neighbor size and timing.

Useful overrides:

```bash
ARC_ACTOR_MODEL='local:your-model' \
ARC_MAX_ACTIONS=20 \
ARC_DEFAULT_SEARCH_DEPTH=8 \
ARC_CRITIC_TIMEOUT=300 \
ARC_RUN_DIR="$PWD/runs/physics-ls20-local" \
./runPhysics.sh
```

`ARC_CRITIC_TIMEOUT` bounds each trusted Critic evaluator subprocess. If Actor
`world_model.py` hangs or plans for too long, Eggthreads terminates that isolated
tool call and the Critic returns revision feedback instead of blocking the run.

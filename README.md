# ARC-AGI-3 Physics solver

This repository proves Eggopt's general `PhysicsStrategy` against a real local
ARC-AGI-3 environment without placing ARC concepts in Eggopt.

```text
Physics
├── Environment              Observe/Execute effect history
├── Hypotheses
│   └── Backtest
│       └── Modeler          persistent Actor
└── Plan
    └── Plan Review
        └── Planner          persistent Actor
```

`arcagi3_physics.arc_physics(...)` composes:

- durable reset/replay/one-action environment Tasks;
- one Modeler `ActorCritic` that edits a normal `world_model.py` as one
  provisional hypothesis and repairs it from complete-Timeline counterexamples;
- one Planner `ActorCritic` that emits a model-backed plan or one cheap
  falsifying/discovery action with frozen pre-action predictions;
- compile/backtest and deterministic BFS over the one accepted model.

The Modeler's chat response is only a completion signal. The accepted Task
result is an immutable snapshot of `world_model.py`; the mutable workspace file
is its editable projection. The Timeline lives in Eggflow results and is
append-only. Before each Modeler or Planner turn, authoritative game-generated
values are published into that role's persistent `python_repl`; prompts name the
variables instead of duplicating grids and histories into model context. Every
real action reconstructs the local game from its seed, replays the complete Timeline, and
verifies reality before taking exactly one new action. A prediction mismatch
aborts the unexecuted queue and returns evidence to the same Modeler thread.

The existing `arcagi3_cli.py` remains independent and unchanged by this solver.
No benchmark or model run is launched automatically.

## Run one offline game

`runPhysics.sh` runs `ls20`, seed `0`, and resumes from the same run directory
when launched again:

```bash
./runPhysics.sh
```

The ARC venv needs Eggllm's async streaming dependency:

```bash
venv/bin/python -m pip install 'aiohttp>=3.9'
```

The script checks this before opening or resuming a run.

It uses `Pro: GPT-5.6 Sol max` for Modeler and Planner by default, sources API
keys from the egg-mono `.env` when present, and never starts a model server.
Override settings without editing the script:

```bash
ARC_MODELER_MODEL='local:your-model' \
ARC_PLANNER_MODEL='local:your-model' \
ARC_MAX_ACTIONS=20 \
ARC_RUN_DIR="$PWD/runs/physics-ls20-local" \
./runPhysics.sh
```

The accepted model is readable at
`<run-dir>/workspaces/hypotheses/innerContext/world_model.py`; durable task and
thread state live under `<run-dir>/.egg/`.

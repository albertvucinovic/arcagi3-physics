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
- one Modeler `ActorCritic` that edits a normal `world_model.py` with tools,
  keeps several hypotheses inside that program, and repairs it from
  complete-Timeline counterexamples;
- one Planner `ActorCritic` that emits a model-backed plan or a discriminating
  experiment with frozen pre-action predictions;
- compile/backtest, deterministic BFS, and predictive-disagreement search.

The Modeler's chat response is only a completion signal. The accepted Task
result is an immutable snapshot of `world_model.py`; the mutable workspace file
is its editable projection. The Timeline lives in Eggflow results and is
append-only. Every real action
reconstructs the local game from its seed, replays the complete Timeline, and
verifies reality before taking exactly one new action. A prediction mismatch
aborts the unexecuted queue and returns evidence to the same Modeler thread.

The existing `arcagi3_cli.py` remains independent and unchanged by this solver.
No benchmark or model run is launched automatically.

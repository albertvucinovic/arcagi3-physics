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

Runtime topology:

```text
Physics
└── Critic                    trusted Eggthread sandbox and environment authority
    └── Actor                 persistent theorist/planner Git workspace
```

The Actor repository is `<run-dir>/workspace/innerContext`; the Critic's pulled
history is `<run-dir>/workspace/critic-repository`.

Run or resume:

```bash
./runPhysics.sh
```

Review the current or completed run from the Critic's authoritative Git copy:

```bash
./reviewPhysics.sh
```

The terminal viewer starts at the latest public observation. Use left/right
arrows (or `h`/`l`) to traverse the append-only Timeline, Home/End to jump, `r`
to reload a run that is still progressing, and `q` to quit. Its header reports
Actor/Critic turns and commits, real actions, durable evaluation reports, plan
models, compatible models, and the latest trusted resolution. For logs or
non-interactive inspection, print one frame with:

```bash
./reviewPhysics.sh --frame 0
```

Useful overrides:

```bash
ARC_ACTOR_MODEL='local:your-model' \
ARC_MAX_ACTIONS=20 \
ARC_MAX_PLAN_DEPTH=8 \
ARC_RUN_DIR="$PWD/runs/physics-ls20-local" \
./runPhysics.sh
```

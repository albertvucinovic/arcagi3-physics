# ARC-AGI-3 Git Physics solver

This repository maps ARC-AGI-3 onto Eggopt's Git-backed `PhysicsStrategy`.
The strategy is one canonical ActorCritic hierarchy:

```text
Physics
└── Critic                    trusted validation and environment authority
    └── Actor                 persistent theorist and planner
```

The Actor owns a writable Git repository at:

```text
<run-dir>/workspace/innerContext/
```

The Critic keeps a pulled history copy at:

```text
<run-dir>/workspace/critic-repository/
```

## Actor contract

`world_model.py` is the current theory. Multiple competing models coexist as
matching `step_<suffix>` and `reward_<suffix>` functions. The Actor reads
`INSTRUCTIONS.md` and uses:

- `python backtest.py` — replay every model over the Timeline;
- `python plan.py` — find goal plans for all valid models and multi-action
  discrimination plans for model subsets;
- `python commit.py PLAN_ID` — write `committed-plan.json` and make the Actor's
  Git commit.

Every turn must finish with a clean new Git HEAD and a non-empty plan. Scratch
files may be `.gitignore`d. If the Actor deletes or corrupts `.git`, the Critic
restores its pulled history and rehydrates the latest canonical game state.

## Trusted Critic

The Critic pulls Actor HEAD, evaluates that committed snapshot in its own clone,
and independently reruns the canonical backtest and planner. It rejects dirty
repositories, missing commits, invalid model APIs, empty plans, and plans not
returned by the trusted planner. Planning reports include every valid model,
even a model that currently contradicts the Timeline; such a model must be
repaired before its plan may cross the real-action boundary.

An intent combines an action with predictions keyed by model suffix. An
experiment may have a common setup prefix. The Critic executes one intent at a
time and stops on the first wrong prediction or after executing the first intent
whose predictions actually branch. Only the Critic can touch the environment.

Offline play uses one live environment session: reset once at initial creation,
then ordinary actions reuse that session. Reset plus verified Timeline replay is
used only to recover a lost process-local session after interruption.

## Run

```bash
./runPhysics.sh
```

Defaults: game `ls20`, seed `0`, Actor `Pro: GPT-5.6 Sol max`. Override without
editing the script:

```bash
ARC_ACTOR_MODEL='local:your-model' \
ARC_MAX_ACTIONS=20 \
ARC_MAX_PLAN_DEPTH=8 \
ARC_RUN_DIR="$PWD/runs/physics-ls20-local" \
./runPhysics.sh
```

The script sources Egg's `.env`, uses the existing ARC virtual environment, and
resumes from the same run-owned `.egg` and Git repositories.

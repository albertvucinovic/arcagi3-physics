from __future__ import annotations

from pathlib import Path

from eggopt import Agent, PhysicsStrategy

from .tasks import ARCCritic, PrepareARC

ARC_DOMAIN_PROMPT = """You are reverse engineering an ARC-AGI-3 game from public observations.
A public state contains a 64x64 color-index grid, currently legal actions, visible
game state, levels completed, and levels needed to win. Never inspect the real
environment implementation or hidden state.

In `world_model.py`, define one or more matching `step_<suffix>` and
`reward_<suffix>` pairs. `step_*` must predict the complete next public state;
`reward_*` is that model's inferred goal/utility. Use `backtest.py`, `plan.py`,
and `commit.py` exactly as documented in `INSTRUCTIONS.md`.
"""


def arc_physics(
    *,
    game: str,
    seed: int,
    environments_dir: str | Path,
    actor: Agent,
    max_depth: int = 8,
    max_nodes: int = 10_000,
) -> PhysicsStrategy:
    """Compose Git-backed PhysicsStrategy with the ARC domain instruments."""

    environments_dir = str(Path(environments_dir).resolve())
    if not actor.auto_approve_tools:
        raise ValueError("ARC Physics Actor must auto-approve its tools")
    if not {"bash", "python_exec"} <= set(actor.allowed_tools):
        raise ValueError("ARC Physics Actor needs bash and python_exec")
    return PhysicsStrategy(
        actor=actor,
        prepare=lambda **_: PrepareARC(game, seed, environments_dir),
        critic=ARCCritic(
            game,
            seed,
            environments_dir,
            max_depth=max_depth,
            max_nodes=max_nodes,
        ),
        identity={
            "domain": "arc-agi-3",
            "version": 2,
            "game": game,
            "seed": seed,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
        },
    )

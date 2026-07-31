from __future__ import annotations

from pathlib import Path

from eggopt import Agent, PhysicsStrategy

from .environment import Execute, Initialize

ARC_DOMAIN_PROMPT = """You are reverse engineering an ARC-AGI-3 game from public observations.
A public state contains a 64x64 color-index grid, currently legal actions, visible
game state, levels completed, and levels needed to win. Never inspect the real
environment implementation or hidden state.

In `world_model.py`, define matching `step_<suffix>` and `reward_<suffix>` pairs.
`step_*` predicts the complete next public state; `reward_*` is that model's
inferred goal/utility. Use the generic Physics `backtest.py`, `plan.py`, and
`commit.py` instruments documented in `INSTRUCTIONS.md`.
"""


def arc_physics(
    *,
    game: str,
    seed: int,
    environments_dir: str | Path,
    actor: Agent,
    max_depth: int = 8,
    max_nodes: int = 10_000,
    evaluator_timeout_sec: float = 300.0,
) -> PhysicsStrategy:
    """Map ARC environment effects and state semantics onto PhysicsStrategy."""

    environments_dir = str(Path(environments_dir).resolve())
    if not actor.auto_approve_tools:
        raise ValueError("ARC Physics Actor must auto-approve its tools")
    if not {"bash", "python_exec"} <= set(actor.allowed_tools):
        raise ValueError("ARC Physics Actor needs bash and python_exec")
    return PhysicsStrategy(
        actor=actor,
        observe=lambda **_: Initialize(game, seed, environments_dir),
        execute=lambda timeline, intent, **_: Execute(
            game, seed, environments_dir, timeline, intent
        ),
        is_goal=arc_goal,
        identity={"domain": "arc-agi-3", "version": 3, "game": game, "seed": seed},
        domain_information=ARC_DOMAIN_PROMPT,
        legal_actions_key="legal_actions",
        max_depth=max_depth,
        max_nodes=max_nodes,
        evaluator_timeout_sec=evaluator_timeout_sec,
    )


def arc_goal(state) -> bool:
    return state.get("state") == "WIN" or (
        state.get("win_levels", 0) > 0
        and state.get("levels_completed", 0) >= state.get("win_levels", 0)
    )

from __future__ import annotations

from pathlib import Path

from eggopt import Agent, PhysicsStrategy

from .environment import Execute, Initialize, validate_action

ARC_DOMAIN_PROMPT = """You are reverse engineering an ARC-AGI-3 game from public observations.
A public state contains a 64x64 color-index grid, currently legal action IDs,
visible game state, levels completed, and levels needed to win. Never inspect the
real environment implementation or hidden state.

Every ARC action is a JSON object. The available numeric IDs in the public state
tell you which action objects may currently be used. Simple controls and Undo
are represented as {"action": 1} through {"action": 5}, and {"action": 7}.
These objects contain exactly the `action` field. Action 6 is a mouse click
represented as {"action": 6, "data": {"x": X, "y": Y}}, where X and Y are
integers from 0 through 63. (0, 0) is the upper-left; x increases rightward and
y downward. When action 6 is currently legal, choose complete coordinates from
public visual evidence. Never submit a bare action identifier.

In `world_model.py`, define `step_<suffix>` hypotheses. You may additionally
define `reward_<suffix>` for any hypothesis you want the advisory planner to
search. Use the Physics instruments documented in `INSTRUCTIONS.md`.

Complete all required game levels using as few real actions as possible:

- do not waste actions on unnecessary experimentation;
- prefer direct completion when confidence is adequate;
- when uncertainty blocks progress, seek high-information experiments with
  short prefixes;
- remember that Undo is also a real action, not free planning.
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
        execute=lambda timeline, action, **_: Execute(
            game, seed, environments_dir, timeline, action
        ),
        validate_action=validate_action,
        is_goal=arc_goal,
        identity={"domain": "arc-agi-3", "version": 3, "game": game, "seed": seed},
        domain_information=ARC_DOMAIN_PROMPT,
        planner_actions=tuple({"action": action} for action in (1, 2, 3, 4, 5, 7)),
        max_depth=max_depth,
        max_nodes=max_nodes,
        evaluator_timeout_sec=evaluator_timeout_sec,
    )


def arc_goal(state) -> bool:
    return state.get("state") == "WIN" or (
        state.get("win_levels", 0) > 0
        and state.get("levels_completed", 0) >= state.get("win_levels", 0)
    )

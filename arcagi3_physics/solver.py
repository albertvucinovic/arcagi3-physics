from __future__ import annotations

from pathlib import Path

from eggopt import Agent, PhysicsStrategy, TerminalOutcome

from .environment import Execute, Initialize, validate_action
from .grid_to_png import ARC_DOMAIN_FILES

ARC_ACTOR_TOOLS = frozenset(
    {
        "add_local_file_to_model_context",
        "answer_user_while_preserving_llm_turn",
        "bash",
        "python_exec",
        "python_repl",
        "read_long_tool_output",
        "skill",
        "tool_help",
    }
)

ARC_DOMAIN_PROMPT = """You are reverse engineering an ARC-AGI-3 game from public observations.
A public state contains a 64x64 color-index grid, currently legal action IDs,
visible game state, levels completed, and levels needed to win. Never inspect the
real environment implementation or hidden state.

`gridToPng.py` is an ARC domain helper in your repository. Use it to render any
2-D ARC color-index grid, a public state, or the latest state in
`canonical-input.json`, for example:

    python gridToPng.py canonical-input.json scratch/current-grid.png

Then call `add_local_file_to_model_context` with the PNG path to view the image
in your next model context. That tool accepts images only. Use this visual route
whenever it helps you understand spatial structure; keep generated PNGs under
`scratch/` so they are not committed.

Your objective is to pass every required level and reach the public `WIN` state
using as few real actions as possible. Do not treat exploration, a plausible
world model, or completion of only one level as success. `WIN` means the task is
complete. `GAME_OVER` means the run is lost and no further action can recover it.

Every ARC action is a JSON object. The available numeric IDs in the public state
tell you which action objects may currently be used. Simple controls and Undo
are represented as {"action": 1} through {"action": 5}, and {"action": 7}.
These objects contain exactly the `action` field. Action 6 is a mouse click
represented as {"action": 6, "data": {"x": X, "y": Y}}, where X and Y are
integers from 0 through 63. (0, 0) is the upper-left; x increases rightward and
y downward. When action 6 is currently legal, choose complete coordinates from
public visual evidence. Never submit a bare action identifier.

In `world_model.py`, define `step_<suffix>` hypotheses and normally define a
matching `reward_<suffix>` that rewards progress through levels toward `WIN`.
Run `python plan.py` and use its productive suggestions whenever bounded search
can find them. Use the Physics instruments documented in `INSTRUCTIONS.md`.

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
    if not ARC_ACTOR_TOOLS <= set(actor.allowed_tools):
        missing = sorted(ARC_ACTOR_TOOLS - set(actor.allowed_tools))
        raise ValueError(
            f"ARC Physics Actor is missing required tools: {', '.join(missing)}"
        )
    return PhysicsStrategy(
        actor=actor,
        observe=lambda **_: Initialize(game, seed, environments_dir),
        execute=lambda timeline, action, **_: Execute(
            game, seed, environments_dir, timeline, action
        ),
        validate_action=validate_action,
        is_goal=arc_goal,
        identity={"domain": "arc-agi-3", "version": 5, "game": game, "seed": seed},
        terminal_outcome=arc_terminal_outcome,
        domain_information=ARC_DOMAIN_PROMPT,
        domain_files=ARC_DOMAIN_FILES,
        planner_actions=tuple({"action": action} for action in (1, 2, 3, 4, 5, 7)),
        max_depth=max_depth,
        max_nodes=max_nodes,
        evaluator_timeout_sec=evaluator_timeout_sec,
    )


def arc_goal(state) -> bool:
    return state.get("state") == "WIN"


def arc_terminal_outcome(state) -> TerminalOutcome | None:
    """Return why no further ARC action should be attempted, or ``None``."""

    if arc_goal(state):
        return None
    if state.get("state") == "GAME_OVER":
        return TerminalOutcome("game_over")
    if not state.get("legal_actions"):
        return TerminalOutcome("no_legal_actions")
    return None

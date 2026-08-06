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

ARC_DOMAIN_PROMPT = Path(__file__).with_name("systemprompt.md").read_text().strip()


def arc_physics(
    *,
    game: str,
    seed: int,
    environments_dir: str | Path,
    actor: Agent,
    default_search_depth: int = 8,
    default_max_nodes: int = 10_000,
    evaluator_timeout_sec: float = 300.0,
    strategy: str = "verified",
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
    try:
        latent, verified, planner = {
            "latent": (True, False, False),
            "latent-verified": (True, True, False),
            "verified": (False, True, True),
        }[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown Physics strategy: {strategy!r}") from exc
    return PhysicsStrategy(
        actor=actor,
        observe=lambda **_: Initialize(game, seed, environments_dir),
        execute=lambda timeline, action, **_: Execute(
            game, seed, environments_dir, timeline, action
        ),
        validate_action=validate_action,
        is_goal=arc_goal,
        identity={"domain": "arc-agi-3", "version": 5, "game": game, "seed": seed},
        latent=latent,
        verified=verified,
        planner=planner,
        terminal_outcome=arc_terminal_outcome,
        domain_information=ARC_DOMAIN_PROMPT,
        domain_files=ARC_DOMAIN_FILES,
        planner_actions=tuple({"action": action} for action in (1, 2, 3, 4, 5, 7)),
        default_search_depth=default_search_depth,
        default_max_nodes=default_max_nodes,
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

"""Thin ARC-AGI-3 domain adapter for Eggopt PhysicsStrategy."""

from .environment import (
    Execute,
    Initialize,
    clear_live_sessions,
    observation,
    validate_action,
)
from .grid_to_png import ARC_DOMAIN_FILES, GRID_TO_PNG
from .solver import (
    ARC_ACTOR_TOOLS,
    ARC_DOMAIN_PROMPT,
    arc_goal,
    arc_physics,
    arc_terminal_outcome,
)

__all__ = [
    "ARC_ACTOR_TOOLS",
    "ARC_DOMAIN_FILES",
    "ARC_DOMAIN_PROMPT",
    "GRID_TO_PNG",
    "Execute",
    "Initialize",
    "arc_goal",
    "arc_physics",
    "arc_terminal_outcome",
    "clear_live_sessions",
    "observation",
    "validate_action",
]

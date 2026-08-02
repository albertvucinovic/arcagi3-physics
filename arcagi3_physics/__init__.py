"""Thin ARC-AGI-3 domain adapter for Eggopt PhysicsStrategy."""

from .environment import (
    Execute,
    Initialize,
    clear_live_sessions,
    observation,
    validate_action,
)
from .solver import ARC_DOMAIN_PROMPT, arc_goal, arc_physics

__all__ = [
    "ARC_DOMAIN_PROMPT",
    "Execute",
    "Initialize",
    "arc_goal",
    "arc_physics",
    "clear_live_sessions",
    "observation",
    "validate_action",
]

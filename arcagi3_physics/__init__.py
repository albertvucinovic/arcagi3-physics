"""ARC-AGI-3 domain for Eggopt's Git-backed PhysicsStrategy."""

from .environment import Execute, Initialize, clear_live_sessions, observation
from .solver import ARC_DOMAIN_PROMPT, arc_physics
from .tasks import ARCCritic, PrepareARC
from .world import (
    COMMITTED_PLAN,
    WORLD_MODEL,
    WORLD_MODEL_TEMPLATE,
    canonical_plans,
    discover_models,
    ensure_world_model,
    load_committed_plan,
    load_model,
    run_backtest,
    run_planner,
    snapshot_world_model,
)

__all__ = [
    "ARC_DOMAIN_PROMPT",
    "COMMITTED_PLAN",
    "WORLD_MODEL",
    "WORLD_MODEL_TEMPLATE",
    "ARCCritic",
    "Execute",
    "Initialize",
    "PrepareARC",
    "arc_physics",
    "canonical_plans",
    "clear_live_sessions",
    "discover_models",
    "ensure_world_model",
    "load_committed_plan",
    "load_model",
    "observation",
    "run_backtest",
    "run_planner",
    "snapshot_world_model",
]

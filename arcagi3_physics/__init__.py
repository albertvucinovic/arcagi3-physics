"""ARC-AGI-3 proving adapter for Eggopt PhysicsStrategy."""

from .environment import Execute, Observe, observation
from .solver import arc_physics
from .tasks import Backtest, Deliberate, Hypothesize, Test, deterministic_commitment
from .world import (
    WORLD_MODEL_TEMPLATE,
    ensure_world_model,
    load_model,
    run_backtest,
    run_bfs,
    snapshot_world_model,
)

__all__ = [
    "WORLD_MODEL_TEMPLATE",
    "Backtest",
    "Deliberate",
    "Execute",
    "Hypothesize",
    "Observe",
    "Test",
    "arc_physics",
    "deterministic_commitment",
    "ensure_world_model",
    "load_model",
    "observation",
    "run_backtest",
    "run_bfs",
    "snapshot_world_model",
]

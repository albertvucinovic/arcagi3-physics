"""ARC-AGI-3 proving adapter for Eggopt PhysicsStrategy."""

from .environment import Execute, Observe, observation
from .solver import arc_physics
from .tasks import Backtest, Deliberate, Hypothesize, Test, deterministic_commitment
from .world import (
    choose_experiment,
    load_model,
    run_backtest,
    run_bfs,
    snapshot_world_model,
)

__all__ = [
    "Backtest",
    "Deliberate",
    "Execute",
    "Hypothesize",
    "Observe",
    "Test",
    "arc_physics",
    "choose_experiment",
    "deterministic_commitment",
    "load_model",
    "observation",
    "run_backtest",
    "run_bfs",
    "snapshot_world_model",
]

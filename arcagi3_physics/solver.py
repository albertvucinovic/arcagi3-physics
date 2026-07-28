from __future__ import annotations

from pathlib import Path

from eggopt import Agent, PhysicsEffect, PhysicsStrategy

from .environment import Execute, Observe
from .tasks import Deliberate, Hypothesize, Test


def arc_physics(
    *,
    game: str,
    seed: int,
    environments_dir: str | Path,
    modeler: Agent,
    planner: Agent,
) -> PhysicsStrategy:
    """Compose the generic PhysicsStrategy into an ARC-AGI-3 solver."""

    environments_dir = str(Path(environments_dir).resolve())
    if not modeler.auto_approve_tools:
        raise ValueError("modeler must auto-approve tools to edit world_model.py")
    if not ({"bash", "python_exec"} & set(modeler.allowed_tools)):
        raise ValueError("modeler needs bash or python_exec to edit world_model.py")

    def observe(*, thread_id, **_):
        return PhysicsEffect(
            Observe(game, seed, environments_dir),
            thread_id,
            "arc_observe",
            {"game": game, "seed": seed},
        )

    def hypothesize(*, timeline, hypotheses, evidence, workspace, **_):
        return Hypothesize(modeler, timeline, hypotheses, evidence, workspace)

    def test(*, hypotheses, timeline, commitment, workspace, **_):
        return Test(hypotheses, timeline, commitment, workspace)

    def deliberate(*, timeline, hypotheses, evidence, workspace, **_):
        return Deliberate(planner, timeline, hypotheses, evidence, workspace)

    def execute(*, timeline, intent, thread_id, **_):
        return PhysicsEffect(
            Execute(game, seed, environments_dir, timeline, intent),
            thread_id,
            "arc_act_observe",
            {"game": game, "seed": seed, "intent": intent},
        )

    return PhysicsStrategy(
        observe=observe,
        hypothesize=hypothesize,
        test=test,
        deliberate=deliberate,
        execute=execute,
        identity={
            "domain": "arc-agi-3",
            "version": 1,
            "game": game,
            "seed": seed,
        },
    )

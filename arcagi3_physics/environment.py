from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eggflow import Task


def observation(frame: Any) -> dict[str, Any]:
    """Return only public ARC observation fields as durable Python values."""

    layers = tuple(
        tuple(tuple(int(cell) for cell in row) for row in layer.tolist())
        for layer in frame.frame
    )
    return {
        "grid": layers,
        "legal_actions": tuple(int(action) for action in frame.available_actions),
        "state": frame.state.value,
        "levels_completed": int(frame.levels_completed),
        "win_levels": int(frame.win_levels),
    }


@dataclass
class Observe(Task):
    game: str
    seed: int
    environments_dir: str | Path

    def run(self):
        env = _environment(self.game, self.seed, self.environments_dir)
        return observation(env.reset())


@dataclass
class Execute(Task):
    game: str
    seed: int
    environments_dir: str | Path
    timeline: tuple[Any, ...]
    intent: Any

    def run(self):
        env = _environment(self.game, self.seed, self.environments_dir)
        current = observation(env.reset())
        if current != self.timeline[0]:
            raise RuntimeError(
                "ARC reset does not reproduce the recorded initial observation"
            )
        for recorded in self.timeline[1:]:
            current = _step(env, recorded["intent"])
            if current != recorded["observation"]:
                raise RuntimeError("ARC replay contradicts the immutable Timeline")
        actual = _step(env, self.intent)
        return {"intent": self.intent, "observation": actual}


def _environment(game: str, seed: int, environments_dir: str | Path):
    from arc_agi import Arcade, OperationMode

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(environments_dir),
    )
    env = arcade.make(game, seed=seed, render_mode=None)
    if env is None:
        raise ValueError(f"ARC environment is unavailable: {game}")
    return env


def _step(env, intent):
    from arcengine import GameAction

    action = intent["action"] if isinstance(intent, dict) else intent
    data = intent.get("data", {}) if isinstance(intent, dict) else {}
    action = GameAction.from_id(int(action))
    if action not in env.action_space:
        raise ValueError(f"ARC action is not currently legal: {action.name}")
    frame = env.step(action, data=data)
    if frame is None:
        raise RuntimeError("ARC environment returned no observation")
    return observation(frame)

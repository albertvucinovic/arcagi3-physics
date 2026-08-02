from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eggflow import Task

_SESSIONS: dict[tuple[str, int, str], Any] = {}
_SESSIONS_LOCK = threading.Lock()


def observation(frame: Any) -> dict[str, Any]:
    """Return only public ARC observation fields as durable Python values."""

    if frame is None:
        raise RuntimeError("ARC environment returned no observation")
    layers = [
        [[int(cell) for cell in row] for row in layer.tolist()] for layer in frame.frame
    ]
    return {
        "grid": layers,
        "legal_actions": [int(action) for action in frame.available_actions],
        "state": frame.state.value,
        "levels_completed": int(frame.levels_completed),
        "win_levels": int(frame.win_levels),
    }


@dataclass
class Initialize(Task):
    game: str
    seed: int
    environments_dir: str | Path

    def run(self):
        key = _key(self.game, self.seed, self.environments_dir)
        with _SESSIONS_LOCK:
            env = _SESSIONS.get(key)
            if env is None:
                env = _environment(*key)
                initial = observation(env.reset())
                _SESSIONS[key] = env
            else:
                last = getattr(env, "last_response", None)
                initial = (
                    observation(last) if last is not None else observation(env.reset())
                )
        return initial


@dataclass
class Execute(Task):
    game: str
    seed: int
    environments_dir: str | Path
    timeline: tuple[Any, ...]
    action: Any

    cacheable = False

    def run(self):
        key = _key(self.game, self.seed, self.environments_dir)
        with _SESSIONS_LOCK:
            env = _SESSIONS.get(key)
            if env is None:
                env = _recover(key, self.timeline)
                _SESSIONS[key] = env
            current = self.timeline[-1].get("next_state", self.timeline[-1])
            validate_action(current, self.action)
            return _step(env, self.action)


def clear_live_sessions() -> None:
    """Forget process-local sessions; the next action recovers by verified replay."""

    with _SESSIONS_LOCK:
        _SESSIONS.clear()


def _recover(key, timeline):
    env = _environment(*key)
    current = observation(env.reset())
    if current != timeline[0]:
        raise RuntimeError(
            "ARC reset does not reproduce the recorded initial observation"
        )
    for recorded in timeline[1:]:
        validate_action(current, recorded["action"])
        current = _step(env, recorded["action"])
        if current != recorded["next_state"]:
            raise RuntimeError("ARC replay contradicts the immutable Timeline")
    return env


def _key(game, seed, environments_dir):
    return game, int(seed), str(Path(environments_dir).resolve())


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


def validate_action(state, action) -> None:
    """Validate one complete ARC action against a public state."""

    if (
        not isinstance(action, dict)
        or "action" not in action
        or set(action) - {"action", "data"}
    ):
        raise ValueError(
            "ARC actions must be objects containing action and optional data"
        )
    identifier = action.get("action")
    if type(identifier) is not int or identifier not in state.get("legal_actions", ()):
        raise ValueError("ARC action is not currently legal")
    data = action.get("data")
    if identifier == 6:
        valid_click = isinstance(data, dict) and set(data) == {"x", "y"}
        valid_click = valid_click and all(
            type(data[key]) is int and 0 <= data[key] <= 63 for key in ("x", "y")
        )
        if not valid_click:
            raise ValueError(
                "ARC action 6 requires integer click coordinates x and y in [0, 63]"
            )
    elif set(action) != {"action"}:
        raise ValueError("ARC simple actions must contain exactly action")


def _step(env, action):
    from arcengine import GameAction

    data = action.get("data", {})
    game_action = GameAction.from_id(action["action"])
    if game_action not in env.action_space:
        raise ValueError(f"ARC action is not currently legal: {game_action.name}")
    if game_action.is_complex():
        try:
            game_action.validate_data(data)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ARC action 6 requires integer click coordinates x and y in [0, 63]"
            ) from exc
    return observation(env.step(game_action, data=data))

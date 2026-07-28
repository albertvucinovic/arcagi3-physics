from __future__ import annotations

import hashlib
import importlib.util
from collections import deque
from pathlib import Path
from types import ModuleType
from typing import Any

WORLD_MODEL = "world_model.py"
WORLD_MODEL_TEMPLATE = '''"""One provisional model of an observed ARC-AGI-3 world.

Public observation shape::

    {
        "grid": tuple[layers][rows][columns] of color indices 0..15,
        "legal_actions": tuple[int, ...],
        "state": "NOT_PLAYED" | "NOT_FINISHED" | "WIN" | "GAME_OVER",
        "levels_completed": int,
        "win_levels": int,
    }

Timeline shape::

    timeline[0] = initial observation
    timeline[i] = {"intent": intent, "observation": observation}  # i >= 1

An intent always has ``action: int`` and may have ``data`` for ACTION6. During
model search it may also carry a frozen ``prediction`` field; ignore fields that
do not affect the environment transition.

The program is one hypothesis. Revise any of grounding, dynamics, rendering,
or goal when reality contradicts it. Internal state may be any pickle-safe
Python value. For a stochastic world, state may contain a belief distribution.
"""


def current_observation(history):
    """Return the latest public observation from a non-empty Timeline prefix."""
    latest = history[-1]
    return latest.get("observation", latest)


def ground(history):
    """Complete observed Timeline prefix -> current latent state.

    ``history`` is an immutable tuple with the shapes documented above. Infer
    all objects, variables, relations, memory, and the provisional goal needed
    to predict and plan. Do not mutate ``history``.
    """
    raise NotImplementedError("Modeler must implement ground")


def step(state, action):
    """Current latent state + canonical action -> predicted next latent state.

    ``action`` is normally an integer 1..7. It may instead be an intent mapping;
    if so, read ``action['action']`` and optional ``action.get('data', {})``.
    This function must be pure: it must not call the real ARC environment.
    """
    raise NotImplementedError("Modeler must implement step")


def render(state):
    """Predicted latent state -> complete public observation mapping.

    Return the same five-field observation shape documented above so recorded
    reality can be compared exactly with the prediction.
    """
    raise NotImplementedError("Modeler must implement render")


def is_goal(state):
    """Return whether the latent state satisfies the currently inferred goal."""
    raise NotImplementedError("Modeler must implement is_goal")
'''


def ensure_world_model(workspace: str | Path) -> Path:
    """Create the documented editable skeleton once, preserving later revisions."""

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / WORLD_MODEL
    if not path.exists():
        path.write_text(WORLD_MODEL_TEMPLATE)
    return path


def snapshot_world_model(workspace: str | Path) -> str:
    """Return the durable contents of the Modeler's editable world_model.py."""

    path = Path(workspace) / WORLD_MODEL
    if not path.is_file():
        raise FileNotFoundError(f"{WORLD_MODEL} was not created")
    source = path.read_text()
    if not source.strip():
        raise ValueError(f"{WORLD_MODEL} is empty")
    return source


def load_model(source: str, workspace: str | Path) -> ModuleType:
    """Load one snapshotted world-model program in its own module namespace."""

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    path = workspace / f"world_model_{digest[:12]}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"arc_world_{digest}", path)
    if spec is None or spec.loader is None:
        raise ValueError("world model could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("ground", "step", "render", "is_goal"):
        if not callable(getattr(module, name, None)):
            raise TypeError(f"world model must define {name}()")
    return module


def run_backtest(
    source: str, timeline: tuple[Any, ...], workspace: str | Path
) -> dict[str, Any]:
    """Replay every recorded transition and return all counterexamples."""

    model = load_model(source, workspace)
    counterexamples = []
    matches = 0
    for index, transition in enumerate(timeline[1:], start=1):
        history = timeline[:index]
        try:
            state = model.ground(history)
            predicted_state = model.step(state, transition["intent"])
            prediction = model.render(predicted_state)
            actual = transition["observation"]
            if prediction == actual:
                matches += 1
            else:
                counterexamples.append(
                    {"transition": index, "prediction": prediction, "actual": actual}
                )
        except (TypeError, ValueError, RuntimeError, KeyError, AttributeError) as exc:
            counterexamples.append({"transition": index, "error": str(exc)})
    return {"matches": matches, "counterexamples": counterexamples}


def run_bfs(
    source: str,
    timeline: tuple[Any, ...],
    legal_actions: tuple[Any, ...],
    workspace: str | Path,
    *,
    max_depth: int = 12,
) -> tuple[dict[str, Any], ...] | None:
    """Search one deterministic model without spending real actions."""

    model = load_model(source, workspace)
    start = model.ground(timeline)
    frontier = deque([(start, ())])
    seen = {_freeze(start)}
    while frontier:
        state, path = frontier.popleft()
        if model.is_goal(state):
            return tuple(path)
        if len(path) >= max_depth:
            continue
        for action in legal_actions:
            next_state = model.step(state, action)
            key = _freeze(next_state)
            if key in seen:
                continue
            seen.add(key)
            frontier.append(
                (
                    next_state,
                    path
                    + (
                        {
                            "action": action,
                            "prediction": model.render(next_state),
                        },
                    ),
                )
            )
    return None


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value

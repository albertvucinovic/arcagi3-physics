from __future__ import annotations

import hashlib
import importlib.util
from collections import deque
from pathlib import Path
from types import ModuleType
from typing import Any

WORLD_MODEL = "world_model.py"


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


def branches(model: ModuleType) -> tuple[Any, ...]:
    """Return named hypotheses exposed by a model, or one implicit hypothesis."""

    values = getattr(model, "HYPOTHESES", None)
    if values is None:
        return (None,)
    if not isinstance(values, (tuple, list)) or not values:
        raise TypeError("HYPOTHESES must be a non-empty tuple or list")
    return tuple(values)


def run_backtest(
    source: str, timeline: tuple[Any, ...], workspace: str | Path
) -> dict[str, Any]:
    """Replay every transition under every hypothesis and return counterexamples."""

    model = load_model(source, workspace)
    reports = tuple(
        _backtest_branch(model, hypothesis, timeline) for hypothesis in branches(model)
    )
    return {
        "branches": reports,
        "counterexamples": [
            {"hypothesis": report["hypothesis"], **counterexample}
            for report in reports
            for counterexample in report["counterexamples"]
        ],
    }


def run_bfs(
    source: str,
    timeline: tuple[Any, ...],
    legal_actions: tuple[Any, ...],
    workspace: str | Path,
    *,
    max_depth: int = 12,
) -> tuple[dict[str, Any], ...] | None:
    """Search all consistent deterministic hypotheses without real actions."""

    model = load_model(source, workspace)
    for hypothesis in branches(model):
        plan = _run_bfs_branch(
            model, hypothesis, timeline, legal_actions, max_depth=max_depth
        )
        if plan:
            return plan
    return None


def choose_experiment(
    source: str,
    timeline: tuple[Any, ...],
    legal_actions: tuple[Any, ...],
    workspace: str | Path,
) -> dict[str, Any] | None:
    """Choose the action inducing greatest disagreement among hypotheses."""

    model = load_model(source, workspace)
    hypotheses = branches(model)
    best = None
    for order, action in enumerate(legal_actions):
        predictions = tuple(
            model.render(
                model.step(model.ground(timeline, hypothesis), action, hypothesis),
                hypothesis,
            )
            for hypothesis in hypotheses
        )
        groups = len({_freeze(value) for value in predictions})
        candidate = (
            groups,
            -order,
            {
                "action": action,
                "hypotheses": hypotheses,
                "predictions": predictions,
            },
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best is not None else None


def _backtest_branch(model, hypothesis, timeline):
    counterexamples = []
    matches = 0
    for index, transition in enumerate(timeline[1:], start=1):
        history = timeline[:index]
        try:
            state = model.ground(history, hypothesis)
            predicted_state = model.step(state, transition["intent"], hypothesis)
            prediction = model.render(predicted_state, hypothesis)
            actual = transition["observation"]
            if prediction == actual:
                matches += 1
            else:
                counterexamples.append(
                    {"transition": index, "prediction": prediction, "actual": actual}
                )
        except (TypeError, ValueError, RuntimeError, KeyError, AttributeError) as exc:
            counterexamples.append({"transition": index, "error": str(exc)})
    return {
        "hypothesis": hypothesis,
        "matches": matches,
        "counterexamples": counterexamples,
    }


def _run_bfs_branch(model, hypothesis, timeline, legal_actions, *, max_depth):
    start = model.ground(timeline, hypothesis)
    frontier = deque([(start, ())])
    seen = {_freeze(start)}
    while frontier:
        state, path = frontier.popleft()
        if model.is_goal(state, hypothesis):
            return tuple(path)
        if len(path) >= max_depth:
            continue
        for intent in legal_actions:
            next_state = model.step(state, intent, hypothesis)
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
                            "action": intent,
                            "hypothesis": hypothesis,
                            "prediction": model.render(next_state, hypothesis),
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

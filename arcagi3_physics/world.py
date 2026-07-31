from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from collections import deque
from pathlib import Path
from types import ModuleType
from typing import Any

WORLD_MODEL = "world_model.py"
COMMITTED_PLAN = "committed-plan.json"

WORLD_MODEL_TEMPLATE = '''"""Current competing hypotheses for one observed ARC-AGI-3 world.

Each model is a matching pair named ``step_<suffix>`` and ``reward_<suffix>``.
Shared helpers and arbitrary internal state representations are allowed.

``step_<suffix>(state, action)`` receives one complete public observation and an
intent/action, and returns the next complete predicted public observation.
``reward_<suffix>(state)`` returns a finite utility; larger is better.
"""


def step_1(state, action):
    """Current transition hypothesis; replace this placeholder."""
    raise NotImplementedError


def reward_1(state):
    """Current goal/reward hypothesis; replace this placeholder."""
    return 0.0
'''


def ensure_world_model(workspace: str | Path) -> Path:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / WORLD_MODEL
    if not path.exists():
        path.write_text(WORLD_MODEL_TEMPLATE)
    return path


def snapshot_world_model(workspace: str | Path) -> str:
    path = Path(workspace) / WORLD_MODEL
    if not path.is_file():
        raise FileNotFoundError(f"{WORLD_MODEL} was not created")
    source = path.read_text()
    if not source.strip():
        raise ValueError(f"{WORLD_MODEL} is empty")
    return source


def load_model(source: str, workspace: str | Path) -> ModuleType:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.encode()).hexdigest()
    path = workspace / f"world_model_{digest[:12]}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"arc_world_{digest}", path)
    if spec is None or spec.loader is None:
        raise ValueError("world model could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_models(module: ModuleType) -> dict[str, tuple[Any, Any]]:
    steps = {
        name[5:]: value
        for name, value in vars(module).items()
        if name.startswith("step_") and name[5:] and callable(value)
    }
    rewards = {
        name[7:]: value
        for name, value in vars(module).items()
        if name.startswith("reward_") and name[7:] and callable(value)
    }
    if not steps:
        raise ValueError("world_model.py defines no step_<suffix> functions")
    missing_rewards = sorted(set(steps) - set(rewards))
    orphan_rewards = sorted(set(rewards) - set(steps))
    if missing_rewards or orphan_rewards:
        raise ValueError(
            "step/reward suffixes must match; "
            f"missing rewards={missing_rewards}, orphan rewards={orphan_rewards}"
        )
    return {suffix: (steps[suffix], rewards[suffix]) for suffix in sorted(steps)}


def transitions(timeline: tuple[Any, ...]):
    for item in timeline[1:]:
        yield item["state"], item["action"], item["next_state"]


def run_backtest(
    source: str, timeline: tuple[Any, ...], workspace: str | Path
) -> dict[str, Any]:
    module = load_model(source, workspace)
    models = discover_models(module)
    reports = {}
    for suffix, (step, _reward) in models.items():
        mismatches = []
        matches = 0
        for index, (state, action, actual) in enumerate(transitions(timeline), start=1):
            try:
                predicted = step(state, action)
                if predicted == actual:
                    matches += 1
                else:
                    mismatches.append(
                        {"transition": index, "prediction": predicted, "actual": actual}
                    )
            except (
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
                AttributeError,
            ) as exc:
                mismatches.append({"transition": index, "error": str(exc)})
        reports[suffix] = {"matches": matches, "mismatches": mismatches}
    return {
        "models": reports,
        "surviving_models": [
            suffix for suffix, report in reports.items() if not report["mismatches"]
        ],
    }


def run_planner(
    source: str,
    timeline: tuple[Any, ...],
    workspace: str | Path,
    *,
    max_depth: int = 8,
    max_nodes: int = 10_000,
) -> dict[str, Any]:
    module = load_model(source, workspace)
    models = discover_models(module)
    current = current_state(timeline)
    goal_plans = {
        suffix: _goal_plan(step, reward, current, max_depth, max_nodes)
        for suffix, (step, reward) in models.items()
    }
    discrimination = []
    suffixes = tuple(models)
    for size in range(2, len(suffixes) + 1):
        for subset in itertools.combinations(suffixes, size):
            found = _distinguishing_plan(
                {suffix: models[suffix][0] for suffix in subset},
                current,
                max_depth,
                max_nodes,
            )
            if found is not None:
                discrimination.append({"models": subset, "plan": found})
    report = {
        "goal_plans": goal_plans,
        "discrimination_plans": discrimination,
    }
    report["plans"] = canonical_plans(report)
    return report


def current_state(timeline: tuple[Any, ...]) -> dict[str, Any]:
    latest = timeline[-1]
    return latest.get("next_state", latest)


def load_committed_plan(workspace: str | Path) -> dict[str, Any]:
    path = Path(workspace) / COMMITTED_PLAN
    if not path.is_file():
        raise ValueError(f"{COMMITTED_PLAN} is missing")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{COMMITTED_PLAN} is not valid JSON") from exc
    return _canonical_plan(value)


def _canonical_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"purpose", "models", "intents"}:
        raise ValueError("plan must contain exactly purpose, models, and intents")
    if value["purpose"] not in {"goal", "experiment"}:
        raise ValueError("plan purpose must be goal or experiment")
    models = value["models"]
    intents = value["intents"]
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(item, str) and item for item in models)
    ):
        raise ValueError("plan models must be a non-empty string list")
    if not isinstance(intents, list) or not intents:
        raise ValueError("committed plan must contain at least one intent")
    for intent in intents:
        if not isinstance(intent, dict) or "action" not in intent:
            raise ValueError("every intent must contain an action")
        predictions = intent.get("prediction")
        if not isinstance(predictions, dict) or set(predictions) != set(models):
            raise ValueError("every intent must predict once for every plan model")
    return {"purpose": value["purpose"], "models": models, "intents": intents}


def _goal_plan(step, reward, start, max_depth, max_nodes):
    frontier = deque([(start, ())])
    seen = {_freeze(start)}
    baseline = float(reward(start))
    best = (baseline, ())
    nodes = 0
    while frontier and nodes < max_nodes:
        state, path = frontier.popleft()
        nodes += 1
        score = float(reward(state))
        if score > best[0]:
            best = (score, path)
        if len(path) >= max_depth:
            continue
        for action in _legal_actions(state):
            next_state = step(state, action)
            key = _freeze(next_state)
            if key in seen:
                continue
            seen.add(key)
            frontier.append(
                (
                    next_state,
                    path + ({"action": action, "prediction": next_state},),
                )
            )
    return list(best[1]) if best[0] > baseline and best[1] else None


def _distinguishing_plan(steps, start, max_depth, max_nodes):
    suffixes = tuple(steps)
    frontier = deque([({suffix: start for suffix in suffixes}, ())])
    seen = {_freeze({suffix: start for suffix in suffixes})}
    nodes = 0
    while frontier and nodes < max_nodes:
        states, path = frontier.popleft()
        nodes += 1
        if len(path) >= max_depth:
            continue
        legal_sets = [set(_legal_actions(state)) for state in states.values()]
        common = set.intersection(*legal_sets) if legal_sets else set()
        for action in sorted(common, key=repr):
            next_states = {
                suffix: steps[suffix](states[suffix], action) for suffix in suffixes
            }
            intent = {"action": action, "prediction": next_states}
            next_path = path + (intent,)
            if len({_freeze(value) for value in next_states.values()}) > 1:
                return list(next_path)
            key = _freeze(next_states)
            if key in seen:
                continue
            seen.add(key)
            frontier.append((next_states, next_path))
    return None


def goal_plan_for(suffix: str, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"action": item["action"], "prediction": {suffix: item["prediction"]}}
        for item in plan
    ]


def canonical_plans(report: dict[str, Any]) -> list[dict[str, Any]]:
    plans = []
    for suffix, plan in report["goal_plans"].items():
        if plan:
            plans.append(
                {
                    "purpose": "goal",
                    "models": [suffix],
                    "intents": goal_plan_for(suffix, plan),
                }
            )
    plans.extend(
        {
            "purpose": "experiment",
            "models": list(item["models"]),
            "intents": item["plan"],
        }
        for item in report["discrimination_plans"]
    )
    return [_canonical_plan(plan) for plan in plans]


def _legal_actions(state):
    if not isinstance(state, dict):
        raise TypeError("model states must expose legal_actions in a mapping")
    actions = state.get("legal_actions", ())
    return tuple(actions)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value

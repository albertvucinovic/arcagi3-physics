from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eggflow import Task
from eggopt import Critique
from eggopt.identity import digest_payload
from eggthreads import set_thread_working_directory

from .environment import Execute, Initialize
from .instruments import write_actor_files
from .world import (
    canonical_plans,
    load_committed_plan,
    run_backtest,
    run_planner,
    snapshot_world_model,
)


@dataclass
class PrepareARC(Task):
    game: str
    seed: int
    environments_dir: str
    workspace: str | None = None
    outer_context: str | None = None

    def get_cache_key(self):
        return digest_payload(
            "arcagi3.physics.prepare.v1",
            {
                "game": self.game,
                "seed": self.seed,
                "environments": self.environments_dir,
            },
        )

    def run(self):
        initial = yield Initialize(self.game, self.seed, self.environments_dir)
        workspace = Path(self.workspace)
        from .world import ensure_world_model

        ensure_world_model(workspace)
        write_actor_files(workspace, (initial,))
        _write_state(workspace, (initial,), 0, None)
        if self.outer_context:
            _write_state(Path(self.outer_context), (initial,), 0, None)


@dataclass
class ARCCritic(Task):
    game: str
    seed: int
    environments_dir: str
    max_depth: int = 8
    max_nodes: int = 10_000
    workspace: str | None = None
    actor_workspace: str | None = None
    head: str | None = None
    max_actions: int = 100
    critic_thread_id: str | None = None
    outer_context: str | None = None

    def get_cache_key(self):
        return digest_payload(
            "arcagi3.physics.critic.v1",
            {
                "game": self.game,
                "seed": self.seed,
                "environments": self.environments_dir,
                "max_depth": self.max_depth,
                "max_nodes": self.max_nodes,
                "head": self.head,
                "max_actions": self.max_actions,
            },
        )

    def run(self):
        repository = Path(self.workspace)
        if self.critic_thread_id is not None:
            from eggopt.context import _current_operation, _operation_runtime

            context = _current_operation()
            set_thread_working_directory(
                _operation_runtime(str(context["_runtime_key"])),
                self.critic_thread_id,
                str(repository),
                reason="ARC trusted Critic clone",
            )
        state_root = Path(self.outer_context) if self.outer_context else repository
        state = _read_state(state_root)
        timeline = tuple(state["timeline"])
        actions = int(state["actions"])
        source = snapshot_world_model(repository)
        try:
            backtest = run_backtest(
                source, timeline, repository / ".trusted" / "backtest"
            )
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
            return self._revise(repository, timeline, actions, "backtest", str(exc))

        try:
            planning = run_planner(
                source,
                timeline,
                repository / ".trusted" / "planner",
                max_depth=self.max_depth,
                max_nodes=self.max_nodes,
            )
            plans = canonical_plans(planning)
            committed = load_committed_plan(repository)
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
            return self._revise(
                repository, timeline, actions, "planning", str(exc), backtest
            )

        if committed not in plans:
            return self._revise(
                repository,
                timeline,
                actions,
                "committed-plan",
                "committed-plan.json is not one of the trusted planner's returned plans",
                backtest,
                planning,
            )
        if not set(committed["models"]) <= set(backtest["surviving_models"]):
            return self._revise(
                repository,
                timeline,
                actions,
                "committed-plan",
                "The trusted planner reports this plan, but one or more referenced models "
                "already contradict the Timeline. Use the all-model report to repair or "
                "replace them before committing real actions.",
                backtest,
                planning,
            )

        current = timeline[-1].get("next_state", timeline[-1])
        legal = set(current.get("legal_actions", ()))
        if int(committed["intents"][0]["action"]) not in legal:
            return self._revise(
                repository,
                timeline,
                actions,
                "committed-plan",
                "The first committed action is not legal in the canonical current state.",
                backtest,
                planning,
            )

        executed = []
        compatible = set(committed["models"])
        resolution = "plan_exhausted"
        for intent in committed["intents"]:
            if actions >= self.max_actions:
                resolution = "max_actions"
                break
            transition = yield Execute(
                self.game,
                self.seed,
                self.environments_dir,
                timeline,
                intent,
            )
            timeline += (transition,)
            executed.append(transition)
            actions += 1
            actual = transition["next_state"]
            predictions = intent["prediction"]
            matching = {name for name in compatible if predictions[name] == actual}
            branched = len({_freeze(value) for value in predictions.values()}) > 1
            if not matching:
                compatible.clear()
                resolution = "wrong_prediction"
                break
            compatible = matching
            if branched:
                resolution = "models_discriminated"
                break
            if _won(actual):
                resolution = "won"
                break

        report = {
            "stage": "execution",
            "head": self.head,
            "backtest": backtest,
            "planning": planning,
            "committed_plan": committed,
            "executed": executed,
            "resolution": resolution,
            "compatible_models": sorted(compatible),
            "actions": actions,
        }
        self._sync(repository, state_root, timeline, actions, report)
        if resolution == "won":
            return Critique.accept(
                {
                    "stopping_reason": "won",
                    "timeline": timeline,
                    "actions": actions,
                    "report": report,
                },
                "Game won.",
            )
        if resolution == "max_actions":
            return Critique.accept(
                {
                    "stopping_reason": "max_actions",
                    "timeline": timeline,
                    "actions": actions,
                    "report": report,
                },
                "The real-action budget is exhausted.",
            )
        return Critique.revise(
            "The trusted plan executed until resolution. Read trusted-report.json and "
            "canonical-input.json, improve world_model.py, then commit another non-empty "
            f"trusted planner result. Resolution: {resolution}."
        )

    def _revise(
        self,
        repository,
        timeline,
        actions,
        stage,
        error,
        backtest=None,
        planning=None,
    ):
        report = {
            "stage": stage,
            "head": self.head,
            "error": error,
            "backtest": backtest,
            "planning": planning,
        }
        state_root = Path(self.outer_context) if self.outer_context else repository
        self._sync(repository, state_root, timeline, actions, report)
        return Critique.revise(
            f"Trusted {stage} failed: {error}. Read trusted-report.json, fix the current "
            "theory or plan, make a clean Git commit, and answer again."
        )

    @staticmethod
    def _sync(repository, state_root, timeline, actions, report):
        _write_json(repository / "trusted-report.json", report)
        _write_state(repository, timeline, actions, report)
        if state_root != repository:
            _write_state(state_root, timeline, actions, report)
        write_actor_files(repository, timeline)
        # The generic Git Critic commits and pulls these trusted files afterward.


def _read_state(repository):
    path = repository / ".trusted" / "state.json"
    if not path.is_file():
        raise RuntimeError("trusted canonical state is missing")
    return json.loads(path.read_text())


def _write_state(repository, timeline, actions, report):
    trusted = repository / ".trusted"
    trusted.mkdir(parents=True, exist_ok=True)
    _write_json(
        trusted / "state.json",
        {"timeline": timeline, "actions": actions, "last_report": report},
    )
    _write_json(repository / "canonical-input.json", {"timeline": timeline})
    if report is not None:
        _write_json(repository / "trusted-report.json", report)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=repr) + "\n")


def _won(observation):
    return observation.get("state") == "WIN" or (
        observation.get("win_levels", 0) > 0
        and observation.get("levels_completed", 0) >= observation.get("win_levels", 0)
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value

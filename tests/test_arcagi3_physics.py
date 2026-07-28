from __future__ import annotations

from pathlib import Path

import pytest
from eggflow import FlowExecutor, Task, TaskStore
from eggopt import ActorCritic, Agent, PhysicsStrategy

from arcagi3_physics.environment import Execute, Observe
from arcagi3_physics.run import build_parser
from arcagi3_physics.solver import arc_physics
from arcagi3_physics.tasks import Backtest, Hypothesize, deterministic_commitment
from arcagi3_physics.world import (
    choose_experiment,
    run_backtest,
    run_bfs,
    snapshot_world_model,
)

WORLD_MODEL = """
HYPOTHESES = ("right", "double")

def ground(history, hypothesis):
    latest = history[-1]
    observation = latest.get("observation", latest)
    return observation["position"]

def step(state, action, hypothesis):
    value = action["action"] if isinstance(action, dict) else action
    return state + value * (2 if hypothesis == "double" else 1)

def render(state, hypothesis):
    return {"position": state, "legal_actions": (1, 2)}

def is_goal(state, hypothesis):
    return state >= (4 if hypothesis == "double" else 2)
"""

CONSISTENT_MODEL = WORLD_MODEL.replace(
    'HYPOTHESES = ("right", "double")', 'HYPOTHESES = ("right",)'
)


class ScriptedLLM:
    current_model_key = "test-model"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0

    def set_model(self, key):
        self.current_model_key = key

    def set_model_with_config(self, key, _config):
        self.current_model_key = key

    async def astream_chat(self, _messages, **_kwargs):
        self.calls += 1
        yield {
            "type": "message",
            "role": "assistant",
            "content": next(self.replies),
            "stop_reason": "end_turn",
        }


def test_world_model_file_backtest_plan_and_discriminating_experiment(tmp_path):
    model_path = tmp_path / "world_model.py"
    model_path.write_text(WORLD_MODEL)
    source = snapshot_world_model(tmp_path)
    timeline = (
        {"position": 0, "legal_actions": (1, 2)},
        {
            "intent": {
                "action": 1,
                "prediction": {"position": 1, "legal_actions": (1, 2)},
            },
            "observation": {"position": 1, "legal_actions": (1, 2)},
        },
    )

    report = run_backtest(source, timeline, tmp_path / "backtest")
    assert report["branches"][0] == {
        "hypothesis": "right",
        "matches": 1,
        "counterexamples": [],
    }
    assert report["branches"][1]["counterexamples"][0]["transition"] == 1

    plan = run_bfs(source, timeline, (1, 2), tmp_path / "bfs")
    assert plan == (
        {
            "action": 1,
            "hypothesis": "right",
            "prediction": {"position": 2, "legal_actions": (1, 2)},
        },
    )

    experiment = choose_experiment(source, timeline, (1, 2), tmp_path / "experiment")
    assert experiment["action"] == 1
    assert len({_freeze(value) for value in experiment["predictions"]}) == 2


def test_deterministic_commitment_prefers_goal_plan(tmp_path):
    timeline = ({"position": 0, "legal_actions": (1, 2)},)
    assert (
        deterministic_commitment(CONSISTENT_MODEL, timeline, tmp_path)[0]["action"] == 2
    )


def test_backtest_missing_file_returns_actor_revision(tmp_path):
    critic = Backtest(({"position": 0},), str(tmp_path))
    store = TaskStore(str(tmp_path / "flow.db"))
    try:
        result = __import__("asyncio").run(FlowExecutor(store).run(critic))
    finally:
        store.close()
    assert result["decision"] == "revise"
    assert "world_model.py" in result["feedback"]


def test_hypothesize_returns_world_model_file_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    modeler = ScriptedLLM(["world_model.py saved"] * 10)

    class PrewrittenHypothesize(Hypothesize):
        def run(self):
            inner = Path(self.workspace, "innerContext")
            inner.mkdir(parents=True, exist_ok=True)
            Path(inner, "world_model.py").write_text(CONSISTENT_MODEL)
            result = yield ActorCritic(
                actor=self.agent,
                critic=Backtest(self.timeline, str(inner)),
                actor_prompt=self._prompt,
                max_rounds=self.max_rounds,
                names=("Modeler", "Backtest"),
            )
            return snapshot_world_model(result.workspace)

    class Value(Task):
        def __init__(self, value):
            self.value = value

        def run(self):
            return self.value

    result = PhysicsStrategy(
        observe=lambda **_: Value({"position": 0, "legal_actions": (1,)}),
        hypothesize=lambda timeline, hypotheses, evidence, workspace, **_: (
            PrewrittenHypothesize(
                Agent(modeler, {"role": "file-modeler"}),
                timeline,
                hypotheses,
                evidence,
                workspace,
                branches=1,
            )
        ),
        test=lambda **_: Value(None),
        deliberate=lambda **_: Value(None),
        execute=lambda **_: Value(None),
        identity={"test": "world-model-file"},
    ).run(run_dir=tmp_path / "run", max_cycles=1)

    assert result.hypotheses == CONSISTENT_MODEL
    assert modeler.calls == 1


class FakeState:
    value = "NOT_FINISHED"


class FakeFrame:
    def __init__(self, position):
        import numpy as np

        self.frame = [np.array([[position]], dtype=int)]
        self.available_actions = [1]
        self.state = FakeState()
        self.levels_completed = 0
        self.win_levels = 1


class FakeEnv:
    def __init__(self):
        self.position = 0
        self.action_space = [FakeAction(1)]

    def reset(self):
        self.position = 0
        return FakeFrame(0)

    def step(self, _action, data=None):
        self.position += 1
        return FakeFrame(self.position)


class FakeAction:
    def __init__(self, value):
        self.value = value
        self.name = f"ACTION{value}"

    def __eq__(self, other):
        return getattr(other, "value", None) == self.value


def test_offline_execute_replays_timeline_before_one_new_action(monkeypatch):
    from arcagi3_physics import environment

    monkeypatch.setattr(environment, "_environment", lambda *_: FakeEnv())
    monkeypatch.setattr(
        environment,
        "_step",
        lambda env, _intent: environment.observation(env.step(None)),
    )

    initial = Observe("fake", 0, ".").run()
    first = Execute("fake", 0, ".", (initial,), {"action": 1}).run()
    second = Execute("fake", 0, ".", (initial, first), {"action": 1}).run()

    assert first["observation"]["grid"] == (((1,),),)
    assert second["observation"]["grid"] == (((2,),),)


def test_real_offline_arc_observe_and_one_replayed_action():

    environments = Path("environment_files").resolve()
    if not environments.exists():
        pytest.skip("local ARC environment files are unavailable")

    initial = Observe("ls20", 0, environments).run()
    assert initial["legal_actions"]
    intent = {"action": initial["legal_actions"][0], "prediction": initial}
    transition = Execute("ls20", 0, environments, (initial,), intent).run()
    assert transition["intent"] == intent
    assert transition["observation"]["grid"]


def test_arc_physics_requires_file_editing_modeler(tmp_path):
    agent = Agent(ScriptedLLM([]), {"role": "modeler"})
    with pytest.raises(ValueError, match="auto-approve"):
        arc_physics(
            game="fake",
            seed=0,
            environments_dir=tmp_path,
            modeler=agent,
            planner=agent,
        )


def test_offline_runner_defaults_to_ls20_seed_zero():
    arguments = build_parser().parse_args([])

    assert arguments.game == "ls20"
    assert arguments.seed == 0
    assert arguments.run_dir == Path("runs/physics-ls20-seed0")
    assert arguments.modeler_model == "Pro: GPT-5.6 Sol max"
    assert arguments.planner_model == "Pro: GPT-5.6 Sol max"


def test_run_script_fails_fast_when_aiohttp_is_missing(tmp_path):
    import os
    import subprocess

    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", "runPhysics.sh", "--help"],
        env={**os.environ, "PYTHON": str(python)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Missing aiohttp" in result.stderr


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value

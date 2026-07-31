from __future__ import annotations

import json
from pathlib import Path

import pytest
from eggflow import FlowExecutor, TaskStore
from eggopt import Agent, physics_actor_system_prompt

from arcagi3_physics.environment import Execute, Initialize, clear_live_sessions
from arcagi3_physics.instruments import (
    actor_backtest,
    actor_commit,
    actor_plan,
    write_actor_files,
)
from arcagi3_physics.run import build_parser
from arcagi3_physics.solver import ARC_DOMAIN_PROMPT, arc_physics
from arcagi3_physics.tasks import ARCCritic
from arcagi3_physics.world import (
    WORLD_MODEL_TEMPLATE,
    canonical_plans,
    discover_models,
    ensure_world_model,
    load_model,
    run_backtest,
    run_planner,
)

MODEL = """
def step_left(state, action):
    value = action["action"] if isinstance(action, dict) else action
    return {"position": state["position"] + value, "legal_actions": [1, 2]}

def reward_left(state):
    return float(state["position"] >= 2)

def step_right(state, action):
    value = action["action"] if isinstance(action, dict) else action
    delta = value if state["position"] else value
    return {"position": state["position"] + delta, "legal_actions": [1, 2]}

def reward_right(state):
    return float(state["position"] >= 3)
"""

DISAGREEING = """
def step_a(state, action):
    return {"position": state["position"] + 1, "legal_actions": [1]}

def reward_a(state):
    return float(state["position"] >= 2)

def step_b(state, action):
    delta = 1 if state["position"] == 0 else 2
    return {"position": state["position"] + delta, "legal_actions": [1]}

def reward_b(state):
    return float(state["position"] >= 3)
"""


def test_discovers_matching_step_reward_pairs(tmp_path):
    module = load_model(MODEL, tmp_path)
    assert set(discover_models(module)) == {"left", "right"}

    bad = MODEL + "\ndef step_orphan(state, action): return state\n"
    with pytest.raises(ValueError, match="suffixes must match"):
        discover_models(load_model(bad, tmp_path / "bad"))


def test_backtest_reports_all_models_and_survivors(tmp_path):
    timeline = (
        {"position": 0, "legal_actions": [1, 2]},
        {
            "state": {"position": 0, "legal_actions": [1, 2]},
            "action": {"action": 1},
            "next_state": {"position": 1, "legal_actions": [1, 2]},
        },
    )
    report = run_backtest(MODEL, timeline, tmp_path)

    assert set(report["models"]) == {"left", "right"}
    assert report["surviving_models"] == ["left", "right"]


def test_planner_reports_goal_and_multistep_discrimination_for_all_models(tmp_path):
    timeline = ({"position": 0, "legal_actions": [1]},)
    report = run_planner(DISAGREEING, timeline, tmp_path, max_depth=4)

    assert set(report["goal_plans"]) == {"a", "b"}
    experiment = report["discrimination_plans"][0]
    assert experiment["models"] == ("a", "b")
    assert len(experiment["plan"]) == 2
    assert (
        experiment["plan"][0]["prediction"]["a"]
        == experiment["plan"][0]["prediction"]["b"]
    )
    assert (
        experiment["plan"][1]["prediction"]["a"]
        != experiment["plan"][1]["prediction"]["b"]
    )
    assert canonical_plans(report)


def test_planner_keeps_reports_for_models_that_fail_backtest(tmp_path):
    timeline = (
        {"position": 0, "legal_actions": [1]},
        {
            "state": {"position": 0, "legal_actions": [1]},
            "action": {"action": 1},
            "next_state": {"position": 99, "legal_actions": [1]},
        },
    )
    backtest = run_backtest(DISAGREEING, timeline, tmp_path / "backtest")
    planning = run_planner(DISAGREEING, timeline, tmp_path / "planning", max_depth=3)

    assert backtest["surviving_models"] == []
    assert set(planning["goal_plans"]) == {"a", "b"}
    assert planning["discrimination_plans"]


def test_actor_instruments_backtest_plan_and_commit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_world_model(tmp_path).write_text(DISAGREEING)
    write_actor_files(tmp_path, ({"position": 0, "legal_actions": [1]},))
    __import__("subprocess").run(["git", "init", "-b", "main"], check=True)
    __import__("subprocess").run(["git", "config", "user.name", "test"], check=True)
    __import__("subprocess").run(
        ["git", "config", "user.email", "test@test"], check=True
    )
    __import__("subprocess").run(["git", "add", "-A"], check=True)
    __import__("subprocess").run(["git", "commit", "-m", "initial"], check=True)

    actor_backtest()
    actor_plan()
    report = json.loads(Path("plan-report.json").read_text())
    plan_id = next(
        item["plan_id"]
        for item in report["canonical_plans"]
        if item["plan"]["purpose"] == "experiment"
    )
    actor_commit(plan_id)

    committed = json.loads(Path("committed-plan.json").read_text())
    assert committed["intents"]
    assert (
        __import__("subprocess")
        .run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        .stdout
        == ""
    )


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
        self.resets = 0
        self.steps = 0

    def reset(self):
        self.resets += 1
        self.position = 0
        return FakeFrame(0)

    def step(self, _action, data=None):
        self.steps += 1
        self.position += 1
        return FakeFrame(self.position)


class FakeAction:
    def __init__(self, value):
        self.value = value
        self.name = f"ACTION{value}"

    def __eq__(self, other):
        return getattr(other, "value", None) == self.value

    def __hash__(self):
        return hash(self.value)


def test_offline_session_reuses_live_env_and_replays_only_after_loss(monkeypatch):
    from arcagi3_physics import environment

    environments = []

    def make(*_):
        env = FakeEnv()
        environments.append(env)
        return env

    monkeypatch.setattr(environment, "_environment", make)
    monkeypatch.setattr(
        environment,
        "_step",
        lambda env, _intent: environment.observation(env.step(None)),
    )
    clear_live_sessions()

    initial = Initialize("fake", 0, ".").run()
    first = Execute("fake", 0, ".", (initial,), {"action": 1}).run()
    second = Execute("fake", 0, ".", (initial, first), {"action": 1}).run()

    assert len(environments) == 1
    assert environments[0].resets == 1
    assert environments[0].steps == 2

    clear_live_sessions()
    third = Execute("fake", 0, ".", (initial, first, second), {"action": 1}).run()
    assert len(environments) == 2
    assert environments[1].resets == 1
    assert environments[1].steps == 3  # two recovery replays plus one new action
    assert third["next_state"]["grid"] == (((3,),),)


class ScriptedLLM:
    current_model_key = "test"

    def set_model(self, key):
        self.current_model_key = key

    def set_model_with_config(self, key, _config):
        self.current_model_key = key


def test_arc_composition_requires_actor_tools(tmp_path):
    actor = Agent(ScriptedLLM(), {"role": "actor"})
    with pytest.raises(ValueError, match="auto-approve"):
        arc_physics(
            game="fake",
            seed=0,
            environments_dir=tmp_path,
            actor=actor,
        )


def test_runner_and_prompt_defaults():
    arguments = build_parser().parse_args([])
    assert arguments.game == "ls20"
    assert arguments.actor_model == "Pro: GPT-5.6 Sol max"
    assert arguments.max_plan_depth == 8
    prompt = physics_actor_system_prompt(ARC_DOMAIN_PROMPT)
    assert "Git repository" in prompt
    assert "step_<suffix>" in prompt
    assert "ARC-AGI-3" in prompt


def test_world_model_skeleton_documents_multiple_hypotheses(tmp_path):
    source = ensure_world_model(tmp_path).read_text()
    assert source == WORLD_MODEL_TEMPLATE
    assert "step_<suffix>" in source
    assert "reward_<suffix>" in source
    assert "def step_1" in source
    assert "def reward_1" in source


def test_trusted_critic_executes_experiment_through_first_branch(tmp_path, monkeypatch):
    from arcagi3_physics import environment

    monkeypatch.chdir(tmp_path)
    repository = tmp_path / "critic"
    repository.mkdir()
    (repository / "world_model.py").write_text(DISAGREEING)
    initial = {"position": 0, "legal_actions": [1]}
    write_actor_files(repository, (initial,))
    trusted = repository / ".trusted"
    trusted.mkdir()
    (trusted / "state.json").write_text(
        json.dumps({"timeline": [initial], "actions": 0, "last_report": None})
    )
    report = run_planner(DISAGREEING, (initial,), repository / "planning", max_depth=4)
    experiment = next(
        plan for plan in canonical_plans(report) if plan["purpose"] == "experiment"
    )
    (repository / "committed-plan.json").write_text(json.dumps(experiment))

    env = FakeEnv()
    monkeypatch.setattr(environment, "_environment", lambda *_: env)

    def step(fake, _intent):
        fake.position += 1
        return {"position": fake.position, "legal_actions": [1]}

    monkeypatch.setattr(environment, "_step", step)
    clear_live_sessions()
    environment._SESSIONS[environment._key("fake", 0, ".")] = env
    store = TaskStore(str(tmp_path / "flow.db"))
    try:
        result = __import__("asyncio").run(
            FlowExecutor(store).run(
                ARCCritic("fake", 0, ".", workspace=str(repository), max_actions=10)
            )
        )
    finally:
        store.close()

    assert result.decision == "revise"
    state = json.loads((trusted / "state.json").read_text())
    assert state["actions"] == 2
    assert state["last_report"]["resolution"] == "models_discriminated"
    assert state["last_report"]["compatible_models"] == ["a"]

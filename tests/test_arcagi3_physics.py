from __future__ import annotations

from pathlib import Path

import pytest
from eggopt import Agent, physics_actor_system_prompt

from arcagi3_physics.environment import Execute, Initialize, clear_live_sessions
from arcagi3_physics.run import build_parser
from arcagi3_physics.solver import ARC_DOMAIN_PROMPT, arc_goal, arc_physics


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
    t1 = {"state": initial, "action": {"action": 1}, "next_state": first}
    second = Execute("fake", 0, ".", (initial, t1), {"action": 1}).run()

    assert len(environments) == 1
    assert environments[0].resets == 1
    assert environments[0].steps == 2

    clear_live_sessions()
    t2 = {"state": first, "action": {"action": 1}, "next_state": second}
    third = Execute("fake", 0, ".", (initial, t1, t2), {"action": 1}).run()
    assert len(environments) == 2
    assert environments[1].resets == 1
    assert environments[1].steps == 3
    assert third["grid"] == (((3,),),)


def test_arc_goal_uses_trusted_public_completion_fields():
    assert arc_goal({"state": "WIN", "levels_completed": 0, "win_levels": 1})
    assert arc_goal({"state": "NOT_FINISHED", "levels_completed": 2, "win_levels": 2})
    assert not arc_goal(
        {"state": "NOT_FINISHED", "levels_completed": 1, "win_levels": 2}
    )


class ScriptedLLM:
    current_model_key = "test"

    def set_model(self, key):
        self.current_model_key = key

    def set_model_with_config(self, key, _config):
        self.current_model_key = key


def test_arc_composition_is_thin_and_requires_actor_tools(tmp_path):
    actor = Agent(ScriptedLLM(), {"role": "actor"})
    with pytest.raises(ValueError, match="auto-approve"):
        arc_physics(game="fake", seed=0, environments_dir=tmp_path, actor=actor)

    configured = Agent(
        ScriptedLLM(),
        {"role": "actor"},
        auto_approve_tools=True,
        allowed_tools=frozenset({"bash", "python_exec"}),
    )
    strategy = arc_physics(
        game="fake", seed=0, environments_dir=tmp_path, actor=configured
    )
    assert strategy.domain_information == ARC_DOMAIN_PROMPT
    assert strategy.legal_actions_key == "legal_actions"
    assert strategy.is_goal({"state": "WIN"})


def test_runner_and_prompt_defaults():
    arguments = build_parser().parse_args([])
    assert arguments.game == "ls20"
    assert arguments.actor_model == "Pro: GPT-5.6 Sol max"
    assert arguments.max_plan_depth == 8
    prompt = physics_actor_system_prompt(ARC_DOMAIN_PROMPT)
    assert "Git repository" in prompt
    assert "step_<suffix>" in prompt
    assert "ARC-AGI-3" in prompt


def test_real_offline_arc_initial_observation_when_available():
    environments = Path("environment_files").resolve()
    if not environments.exists():
        pytest.skip("local ARC environment files are unavailable")
    clear_live_sessions()
    initial = Initialize("ls20", 0, environments).run()
    assert initial["grid"]
    assert initial["legal_actions"]

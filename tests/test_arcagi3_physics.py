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


def test_reviewer_loads_critic_git_timeline_and_metadata(tmp_path):
    import json
    import subprocess

    from eggthreads import (
        ThreadsDB,
        append_message,
        create_child_thread,
        create_root_thread,
    )

    from arcagi3_physics.review import frame, load_review, render

    run = tmp_path / "run"
    repository = run / "workspace" / "critic-repository"
    repository.mkdir(parents=True)
    subprocess.run(["git", "-C", str(repository), "init", "-b", "main"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Physics"], check=True
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "physics@test"],
        check=True,
    )
    initial = {
        "grid": [[[0, 1], [2, 3]]],
        "legal_actions": [1],
        "state": "NOT_FINISHED",
        "levels_completed": 0,
        "win_levels": 1,
    }
    next_state = {
        **initial,
        "grid": [[[1, 1], [2, 3]]],
    }
    intent = {"action": 1, "prediction": {"a": next_state}}
    timeline = [
        initial,
        {"state": initial, "action": intent, "next_state": next_state},
    ]
    report = {
        "stage": "execution",
        "resolution": "plan_exhausted",
        "compatible_models": ["a"],
        "committed_plan": {
            "purpose": "goal",
            "models": ["a"],
            "intents": [intent],
        },
    }
    trusted = repository / ".trusted"
    (trusted / "evaluations").mkdir(parents=True)
    (trusted / "state.json").write_text(
        json.dumps({"timeline": timeline, "actions": 1, "last_report": report})
    )
    (trusted / "evaluations" / ("a" * 40 + ".json")).write_text(
        json.dumps(
            {
                "backtest": {
                    "models": {"a": {}},
                    "surviving_models": ["a"],
                },
                "planning": {"plans": [{"purpose": "goal"}]},
            }
        )
    )
    subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", "Actor proposal"],
        check=True,
    )
    (repository / "critic.txt").write_text("result")
    subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", "[physics] trusted result"],
        check=True,
    )
    (trusted / "state.json").write_text("not committed and not valid JSON")

    db = ThreadsDB(run / ".egg" / "threads.sqlite")
    db.init_schema()
    physics = create_root_thread(db, name="Physics")
    critic = create_child_thread(db, physics, name="Critic")
    actor = create_child_thread(db, critic, name="Actor")
    append_message(db, actor, "assistant", "proposal ready")
    append_message(db, critic, "assistant", "reviewed")
    db.close()

    review = load_review(run)
    assert review.transitions == 1
    assert review.actor_turns == 1
    assert review.critic_turns == 1
    assert review.actor_commits == 1
    assert review.critic_commits == 1
    assert review.evaluation_reports == 1
    assert review.evaluated_head == "a" * 40
    assert review.model_count == 1
    assert review.surviving_models == ("a",)
    assert review.generated_plans == 1
    assert frame(review, 1)["action"] == intent
    output = render(review, 1, color=False)
    assert "frame 1/1" in output
    assert "Actor turns: 1" in output
    assert "Critic turns: 1" in output
    assert "resolution: plan_exhausted" in output
    assert "models: 1" in output
    assert "surviving: ('a',)" in output
    assert "Arriving action:" in output


def test_reviewer_defaults_to_configured_run():
    from arcagi3_physics.review import _decode_key, build_parser

    assert build_parser().parse_args([]).run_dir == Path("runs/physics-ls20-seed0")
    assert _decode_key("\x1b[D") == "left"
    assert _decode_key("\x1b[C") == "right"
    assert _decode_key("\x1b[H") == "home"
    assert _decode_key("\x1b[F") == "end"

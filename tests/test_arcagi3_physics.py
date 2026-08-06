from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from eggopt import Agent, physics_actor_system_prompt

from arcagi3_physics.environment import (
    Execute,
    Initialize,
    clear_live_sessions,
    validate_action,
)
from arcagi3_physics.grid_to_png import ARC_DOMAIN_FILES, GRID_TO_PNG
from arcagi3_physics.run import build_parser
from arcagi3_physics.solver import (
    ARC_ACTOR_TOOLS,
    ARC_DOMAIN_PROMPT,
    arc_goal,
    arc_physics,
    arc_terminal_outcome,
)

ANSI = re.compile(r"\033\[[0-?]*[ -/]*[@-~]")


def strip_ansi(value: str) -> str:
    return ANSI.sub("", value)


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

    @property
    def observation_space(self):
        return FakeFrame(self.position)

    def step(self, _action, data=None):
        self.steps += 1
        self.position += 1
        return FakeFrame(self.position)


class ComplexAction:
    value = 6

    @property
    def name(self):
        return "ACTION6"

    def __eq__(self, other):
        return getattr(other, "value", None) == self.value

    def __hash__(self):
        return hash(self.value)


class ComplexEnv:
    def __init__(self):
        self.action_space = [ComplexAction()]
        self.received = []

    def step(self, action, data=None):
        self.received.append((action.value, data))
        return FakeFrame(1)


class FakeAction:
    def __init__(self, value):
        self.value = value
        self.name = f"ACTION{value}"

    def __eq__(self, other):
        return getattr(other, "value", None) == self.value

    def __hash__(self):
        return hash(self.value)


def test_offline_session_reuses_live_env_and_replays_only_after_loss(monkeypatch):
    import json

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
    assert environments[0].resets == 0
    assert environments[0].steps == 2

    clear_live_sessions()
    t2 = {"state": first, "action": {"action": 1}, "next_state": second}
    persisted = tuple(json.loads(json.dumps([initial, t1, t2])))
    third = Execute("fake", 0, ".", persisted, {"action": 1}).run()
    assert len(environments) == 2
    assert environments[1].resets == 0
    assert environments[1].steps == 3
    assert third["grid"] == [[[3]]]


def test_step_executes_a_parameterized_click_intent(monkeypatch):
    from arcagi3_physics import environment

    monkeypatch.setattr("arcengine.GameAction.from_id", lambda action: ComplexAction())
    monkeypatch.setattr(ComplexAction, "is_complex", lambda _self: True, raising=False)
    monkeypatch.setattr(
        ComplexAction,
        "validate_data",
        lambda _self, data: (
            True
            if set(data) == {"x", "y"}
            and all(isinstance(data[key], int) and 0 <= data[key] <= 63 for key in data)
            else (_ for _ in ()).throw(ValueError("bad click"))
        ),
        raising=False,
    )
    env = ComplexEnv()

    result = environment._step(
        env,
        {"action": 6, "data": {"x": 12, "y": 34}},
    )

    assert result["legal_actions"] == [1]
    assert env.received == [(6, {"x": 12, "y": 34})]
    environment._step(env, {"action": 6, "data": {"x": 56, "y": 7}})
    assert env.received[-1] == (6, {"x": 56, "y": 7})
    with pytest.raises(ValueError, match="requires integer click coordinates"):
        environment._step(env, {"action": 6})
    with pytest.raises(ValueError, match="requires integer click coordinates"):
        environment._step(env, {"action": 6, "data": {"x": -1, "y": 2}})


def test_domain_validates_unified_arc_action_objects():
    current = {"legal_actions": [1, 6]}
    assert validate_action(current, {"action": 1}) is None
    assert validate_action(current, {"action": 6, "data": {"x": 12, "y": 34}}) is None
    with pytest.raises(ValueError, match="must be objects"):
        validate_action(current, 1)
    with pytest.raises(ValueError, match="must be objects"):
        validate_action(current, {})
    with pytest.raises(ValueError, match="currently legal"):
        validate_action(current, {"action": 2})
    with pytest.raises(ValueError, match="coordinates"):
        validate_action(current, {"action": 6})
    with pytest.raises(ValueError, match="exactly action"):
        validate_action(current, {"action": 1, "data": {}})


def test_arc_goal_uses_trusted_public_completion_fields():
    assert arc_goal({"state": "WIN", "levels_completed": 0, "win_levels": 1})
    assert not arc_goal(
        {"state": "NOT_FINISHED", "levels_completed": 2, "win_levels": 2}
    )
    assert not arc_goal(
        {"state": "NOT_FINISHED", "levels_completed": 1, "win_levels": 2}
    )


def test_arc_terminal_outcome_uses_public_engine_state_and_actions():
    assert (
        arc_terminal_outcome({"state": "GAME_OVER", "legal_actions": [1]}).reason
        == "game_over"
    )
    assert (
        arc_terminal_outcome({"state": "NOT_FINISHED", "legal_actions": []}).reason
        == "no_legal_actions"
    )
    assert (
        arc_terminal_outcome(
            {
                "state": "WIN",
                "legal_actions": [],
                "levels_completed": 1,
                "win_levels": 1,
            }
        )
        is None
    )
    assert arc_terminal_outcome({"state": "NOT_FINISHED", "legal_actions": [1]}) is None


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
        allowed_tools=ARC_ACTOR_TOOLS,
    )
    strategy = arc_physics(
        game="fake", seed=0, environments_dir=tmp_path, actor=configured
    )
    assert configured.allowed_tools == ARC_ACTOR_TOOLS
    assert strategy.domain_information == ARC_DOMAIN_PROMPT
    assert strategy.domain_files == ARC_DOMAIN_FILES
    assert {"action": 1} in strategy.planner_actions
    assert strategy.validate_action is validate_action
    assert strategy.is_goal({"state": "WIN"})
    assert strategy.terminal_outcome({"state": "GAME_OVER"}).reason == "game_over"


def test_runner_and_prompt_defaults():
    arguments = build_parser().parse_args([])
    assert arguments.game == "ls20"
    assert arguments.actor_model == "Pro: GPT-5.6 Sol max"
    assert arguments.default_search_depth == 8
    assert arguments.default_max_nodes == 10_000
    assert arguments.critic_timeout == 300
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--critic-timeout", "0"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--critic-timeout", "nan"])
    prompt = physics_actor_system_prompt(ARC_DOMAIN_PROMPT)
    assert "Git repository" in prompt
    assert "step_<suffix>" in prompt
    assert "hypothesis you consider most likely" in prompt
    assert "using as few real actions as possible" in prompt
    assert "pass every required level" in prompt
    assert "answer_user_while_preserving_llm_turn" in prompt
    assert "plain assistant answer" in prompt
    assert "reach the public `WIN` state" in prompt
    assert "matching `reward_<suffix>`" in prompt
    assert "Run `python plan.py`" in prompt
    assert "gridToPng.py" in prompt
    assert "add_local_file_to_model_context" in prompt
    assert "accepts images only" in prompt
    assert "Undo is also a real action" in prompt
    assert "Planner suggestions are aids, not constraints" in prompt
    assert "need not have been found by `plan.py`" in prompt
    assert "ARC-AGI-3" in prompt
    assert "Action 6 is a mouse" in ARC_DOMAIN_PROMPT
    assert '{"action": 1}' in ARC_DOMAIN_PROMPT
    assert '"data": {"x": X, "y": Y}' in ARC_DOMAIN_PROMPT
    assert "actions_<suffix>" not in ARC_DOMAIN_PROMPT
    assert "actions_<suffix>" not in prompt
    assert "committed-plan" not in prompt
    assert "proposed-plans" not in prompt
    assert "(0, 0) is the upper-left" in ARC_DOMAIN_PROMPT
    assert '{"action": 6, "data": {"x": X, "y": Y}}' in ARC_DOMAIN_PROMPT


def test_real_offline_arc_initial_observation_when_available():
    environments = Path("environment_files").resolve()
    if not environments.exists():
        pytest.skip("local ARC environment files are unavailable")
    clear_live_sessions()
    initial = Initialize("ls20", 0, environments).run()
    assert initial["grid"]
    assert initial["legal_actions"]


def test_real_offline_arc_click_intent_when_available():
    environments = Path("environment_files").resolve()
    if not environments.exists():
        pytest.skip("local ARC environment files are unavailable")
    clear_live_sessions()
    initial = Initialize("lf52", 0, environments).run()
    click = {"action": 6, "data": {"x": 40, "y": 35}}

    result = Execute(
        "lf52",
        0,
        environments,
        (initial,),
        click,
    ).run()

    assert result["grid"]
    assert result["state"] == "NOT_FINISHED"


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
    action = {"action": 1}
    timeline = [
        initial,
        {"state": initial, "action": action, "next_state": next_state},
    ]
    report = {
        "stage": "execution",
        "resolution": "plan_exhausted",
        "matching_models": ["a"],
        "plan": [{"state": initial, "action": action, "next_state": next_state}],
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
                "planning": {"suggestions": [{"kind": "reward"}]},
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
    append_message(
        db,
        actor,
        "user",
        "begin turn",
        extra={"eggopt_actor_critic_key": "eggopt.actor-critic.turn.v1:test"},
    )
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
    assert len(review.turns) == 1
    assert review.turns[0].actor_turn == 1
    assert review.turns[0].proposed_actions == 1
    assert review.turns[0].executed_actions == 1
    assert review.turns[0].action_range == "1"
    assert frame(review, 1)["action"] == action
    assert frame(review, 1)["turn"] == review.turns[0]
    output = render(review, 1, color=False, columns=80, lines=24)
    assert "action 1/1" in output
    assert "Actor turns: 1/1" in output
    assert "Critic turns: 1/1" in output
    assert "Real actions: 1/1" in output
    assert "Critic HEAD" in output
    assert "latest" in output
    assert "resolution: plan_exhausted" in output
    assert "models: 1" in output
    assert "surviving: ('a',)" in output
    assert "Action 1/1:" in output
    assert "Actor turn: 1" in output
    assert "plan proposed: 1" in output
    assert "executed: 1 (actions 1)" in output

    initial_output = render(review, 0, color=False, columns=80, lines=24)
    assert "Actor turns: 0/1" in initial_output
    assert "Real actions: 0/1" in initial_output
    assert "Report stage: -" in initial_output
    assert "Critic commits 0/1" in initial_output
    assert "Actor turn: -" in initial_output


def test_reviewer_defaults_to_configured_run():
    from arcagi3_physics.review import (
        _decode_key,
        _render_grid,
        build_parser,
    )

    assert build_parser().parse_args([]).run_dir == Path("runs/physics-ls20-seed0")
    assert _decode_key("\x1b[D") == "left"
    assert _decode_key("\x1b[C") == "right"
    assert _decode_key("\x1b[H") == "home"
    assert _decode_key("\x1b[F") == "end"

    grid = tuple(
        tuple((row + column) % 16 for column in range(64)) for row in range(64)
    )
    full = _render_grid(grid, color=True, columns=100, lines=40)
    assert len(full) == 32
    assert all(len(strip_ansi(line)) == 64 for line in full)
    assert all(line.endswith("\033[0m") for line in full)
    assert "\033[38;2;255;255;255m" in full[0]
    assert "\033[48;2;204;204;204m" in full[0]

    compact = _render_grid(grid, color=True, columns=40, lines=10)
    assert len(compact) == 10
    assert all(len(strip_ansi(line)) == 20 for line in compact)

    medium = _render_grid(grid, color=True, columns=80, lines=40)
    assert len(medium) == 32
    assert all(len(strip_ansi(line)) == 64 for line in medium)

    plain = _render_grid(grid, color=False, columns=20, lines=5)
    assert len(plain) == 64
    assert all(len(line) == 64 for line in plain)


def test_reviewer_maps_each_action_to_historical_actor_turn_stats(tmp_path):
    import json
    import subprocess

    from arcagi3_physics.review import frame, load_review, render

    run = tmp_path / "run"
    repository = run / "workspace" / "critic-repository"
    trusted = repository / ".trusted"
    evaluations = trusted / "evaluations"
    evaluations.mkdir(parents=True)
    subprocess.run(["git", "-C", str(repository), "init", "-b", "main"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Physics"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "physics@test"],
        check=True,
    )

    def state(position):
        return {
            "grid": [[[position]]],
            "legal_actions": [1],
            "state": "NOT_FINISHED",
            "levels_completed": 0,
            "win_levels": 1,
        }

    timeline = [state(0)]
    (trusted / "state.json").write_text(
        json.dumps({"timeline": timeline, "actions": 0, "last_report": None})
    )
    subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "commit",
            "-m",
            "[physics] initialize canonical world state",
        ],
        check=True,
    )

    def commit_turn(number, proposed, actual_positions, resolution):
        actor_file = repository / f"actor-{number}.txt"
        actor_file.write_text(str(number))
        subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-m", f"Actor turn {number}"],
            check=True,
        )
        actor_head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        plan = []
        executed = []
        current = timeline[-1].get("next_state", timeline[-1])
        for offset in range(proposed):
            predicted = state(current["grid"][0][0][0] + 1)
            plan.append(
                {"state": current, "action": {"action": 1}, "next_state": predicted}
            )
            current = predicted
        current = timeline[-1].get("next_state", timeline[-1])
        for position in actual_positions:
            transition = {
                "state": current,
                "action": {"action": 1},
                "next_state": state(position),
            }
            timeline.append(transition)
            executed.append(transition)
            current = transition["next_state"]
        report = {
            "stage": "execution",
            "head": actor_head,
            "resolution": resolution,
            "plan": plan,
            "executed": executed,
            "matching_models": [f"model-{number}"],
        }
        (trusted / "state.json").write_text(
            json.dumps(
                {
                    "timeline": timeline,
                    "actions": len(timeline) - 1,
                    "last_report": report,
                }
            )
        )
        (evaluations / f"{actor_head}.json").write_text(
            json.dumps(
                {
                    "backtest": {
                        "models": {f"model-{number}": {}},
                        "surviving_models": [f"model-{number}"],
                    },
                    "planning": {"suggestions": [{}] * number},
                }
            )
        )
        subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "commit",
                "-m",
                f"[physics] trusted result {number}",
            ],
            check=True,
        )

    commit_turn(1, proposed=3, actual_positions=[1, 9], resolution="wrong_prediction")
    commit_turn(2, proposed=2, actual_positions=[10, 11], resolution="plan_exhausted")

    review = load_review(run)
    assert review.transitions == 4
    assert review.actor_turns == 2
    assert [turn.action_range for turn in review.turns] == ["1-2", "3-4"]
    assert [turn.proposed_actions for turn in review.turns] == [3, 2]
    assert [turn.executed_actions for turn in review.turns] == [2, 2]
    assert frame(review, 2)["turn"].actor_turn == 1
    assert frame(review, 3)["turn"].actor_turn == 2

    first = render(review, 2, color=False, columns=160, lines=40)
    assert "Actor turns: 1/2" in first
    assert "Real actions: 2/4" in first
    assert "Actor turn: 1" in first
    assert "plan proposed: 3" in first
    assert "executed: 2 (actions 1-2)" in first
    assert "selected: 2/2" in first
    assert "prediction: mismatched" in first
    assert "resolution: wrong_prediction" in first
    assert "model-1" in first

    second = render(review, 3, color=False, columns=160, lines=40)
    assert "Actor turns: 2/2" in second
    assert "Real actions: 3/4" in second
    assert "Actor turn: 2" in second
    assert "plan proposed: 2" in second
    assert "executed: 2 (actions 3-4)" in second
    assert "selected: 1/2" in second
    assert "resolution: plan_exhausted" in second
    assert "prediction: matched" in second
    assert "model-2" in second

    latest = render(review, 4, color=False, columns=160, lines=40)
    assert "selected: 2/2" in latest
    assert "prediction: matched" in latest


def test_reviewer_counts_actor_turn_prompts_not_tool_call_messages(tmp_path):
    from eggthreads import (
        ThreadsDB,
        append_message,
        create_child_thread,
        create_root_thread,
    )

    from arcagi3_physics.review import _thread_turns

    db = ThreadsDB(tmp_path / ".egg" / "threads.sqlite")
    db.init_schema()
    physics = create_root_thread(db, name="Physics")
    critic = create_child_thread(db, physics, name="Critic")
    actor = create_child_thread(db, critic, name="Actor")
    for number in (1, 2):
        append_message(
            db,
            actor,
            "user",
            f"turn {number}",
            extra={
                "eggopt_actor_critic_key": f"eggopt.actor-critic.turn.v1:{number}"
            },
        )
        append_message(
            db,
            actor,
            "assistant",
            "",
            extra={"tool_calls": [{"id": str(number)}]},
        )
        append_message(db, actor, "assistant", f"answer {number}")
    db.close()

    assert _thread_turns(tmp_path / ".egg" / "threads.sqlite") == (2, 0)


def test_interactive_reviewer_fits_terminal_viewport(monkeypatch, tmp_path):
    from arcagi3_physics.review import Review, _raw_input, _terminal_viewport, render

    grid = tuple(
        tuple((row + column) % 16 for column in range(64)) for row in range(64)
    )
    state = {
        "grid": (grid,),
        "legal_actions": (1, 2, 3, 4),
        "state": "NOT_FINISHED",
        "levels_completed": 0,
        "win_levels": 1,
    }
    review = Review(
        repository=tmp_path / ("very-long-directory-name-" * 10),
        timeline=(state,),
        report={},
        head="a" * 40,
        actor_turns=100,
        critic_turns=100,
        actor_commits=100,
        critic_commits=100,
        evaluation_reports=100,
        evaluated_head="b" * 40,
        model_count=100,
        surviving_models=("very-long-model-name",) * 10,
        generated_plans=100,
        turns=(),
    )

    for columns, lines in ((40, 24), (80, 24), (117, 124), (240, 100)):
        output = render(review, 0, color=True, columns=columns, lines=lines)
        visible = [strip_ansi(line) for line in output.splitlines()]
        assert len(visible) <= lines
        assert max(map(len, visible)) <= columns

    large = render(review, 0, color=True, columns=117, lines=124)
    grid_lines = [line for line in large.splitlines() if "▀" in strip_ansi(line)]
    assert len(grid_lines) == 32
    assert all(len(strip_ansi(line)) == 64 for line in grid_lines)

    class Terminal:
        def fileno(self):
            return 42

    monkeypatch.setattr(
        "arcagi3_physics.review.os.get_terminal_size",
        lambda _fd: __import__("os").terminal_size((80, 24)),
    )
    assert _terminal_viewport(Terminal()) == (76, 24)

    class RawTerminal(Terminal):
        pass

    terminal = RawTerminal()
    previous = [0, 10, 0, 0, 0, 0, []]
    raw = [0, 20, 0, 0, 0, 0, []]
    attributes = [previous, raw]
    monkeypatch.setattr(
        "arcagi3_physics.review.termios.tcgetattr", lambda _fd: attributes.pop(0)
    )
    monkeypatch.setattr("arcagi3_physics.review.tty.setraw", lambda _fd: None)
    configured: list[Any] = []
    monkeypatch.setattr(
        "arcagi3_physics.review.termios.tcsetattr",
        lambda _fd, _when, value: configured.append(value),
    )
    with _raw_input(terminal):
        pass
    assert configured == [[0, 10, 0, 0, 0, 0, []], previous]


def test_public_benchmark_discovers_all_local_environment_metadata(tmp_path):
    import json

    from arcagi3_physics.benchmark import discover_public_environments

    for game, version in (("zz99", "first"), ("aa00", "second")):
        directory = tmp_path / game / version
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(
            json.dumps({"game_id": f"{game}-{version}"})
        )

    assert discover_public_environments(tmp_path) == ("aa00", "zz99")


def test_luna_benchmark_defaults_and_single_game_selection():
    from eggthreads import RunnerConfig

    from arcagi3_physics.benchmark import (
        DEFAULT_MODEL,
        _require_complete_public_suite,
        _selected_games,
        build_parser,
        discover_public_environments,
    )

    arguments = build_parser().parse_args([])

    assert arguments.actor_model == DEFAULT_MODEL == "Pro: GPT-5.6 Luna max"
    assert arguments.max_parallel == 3
    assert arguments.actor_context_limit == 300_000
    assert arguments.critic_timeout == 300
    scheduler = RunnerConfig(
        max_concurrent_threads=arguments.max_parallel,
        max_concurrent_llm_threads=arguments.max_parallel,
        sticky_scheduling=False,
        sticky_idle_threshold_sec=5,
    )
    assert scheduler.effective_max_concurrent_llm_threads == 3
    assert scheduler.sticky_scheduling is False
    assert scheduler.sticky_idle_threshold_sec == 5
    environments = Path("environment_files")
    if environments.is_dir():
        assert len(discover_public_environments(environments)) == 25
    assert _selected_games(("aa00", "bb00"), ["bb00"]) == ("bb00",)
    with pytest.raises(ValueError, match="unknown public"):
        _selected_games(("aa00",), ["bb00"])
    _require_complete_public_suite(tuple(f"g{i:02d}" for i in range(25)))
    with pytest.raises(ValueError, match="requires all 25"):
        _require_complete_public_suite(("aa00",))


def test_luna_max_handle_has_max_reasoning_configuration():
    from eggconfig import get_all_models_path, get_models_path

    from arcagi3_physics.benchmark import DEFAULT_MODEL, _validate_luna_max

    _validate_luna_max(
        DEFAULT_MODEL,
        str(get_models_path()),
        str(get_all_models_path()),
    )


def test_prepare_selected_run_creates_selected_physics_tree(tmp_path):
    import asyncio
    import json

    from arcagi3_physics.benchmark import build_parser, prepare

    environments = tmp_path / "environment_files"
    for game in ("aa00", "bb00"):
        version = environments / game / "version"
        version.mkdir(parents=True)
        (version / "metadata.json").write_text(
            json.dumps({"game_id": f"{game}-version"})
        )
    arguments = build_parser().parse_args(
        [
            "--environments-dir",
            str(environments),
            "--run-dir",
            str(tmp_path / "run"),
            "--games",
            "aa00",
        ]
    )

    prepared = asyncio.run(prepare(arguments))

    assert prepared.games == ("aa00",)
    configuration = json.loads((tmp_path / "run" / "benchmark.json").read_text())
    assert configuration["games"] == ["aa00"]


def test_benchmark_root_and_physics_children_are_cached(tmp_path):
    import asyncio

    from eggflow import FlowExecutor, TaskStore
    from eggthreads import ThreadsDB, list_children_with_meta, list_root_threads

    from arcagi3_physics.benchmark import (
        ROOT_NAME,
        _EnsureBenchmarkRoot,
        _EnsurePhysicsRun,
    )

    db = ThreadsDB(tmp_path / "threads.sqlite")
    db.init_schema()
    flow = FlowExecutor(TaskStore(str(tmp_path / "flow.db")))

    async def create_twice():
        root_task = _EnsureBenchmarkRoot(db, "model", "models.json", "all.json")
        root = await flow.run(root_task)
        assert await flow.run(root_task) == root
        child_task = _EnsurePhysicsRun(
            db, root, "aa00", "model", "models.json", "all.json"
        )
        child = await flow.run(child_task)
        assert await flow.run(child_task) == child
        return root, child

    try:
        root, child = asyncio.run(create_twice())
        assert list_root_threads(db) == [root]
        assert db.get_thread(root).name == ROOT_NAME
        assert list_children_with_meta(db, root)[0][:2] == (child, "Physics aa00")
    finally:
        flow.store.close()
        db.close()


def test_benchmark_configuration_round_trips_tuple_games(tmp_path):
    from arcagi3_physics.benchmark import _ensure_configuration

    path = tmp_path / "benchmark.json"
    configuration = {"games": ("aa00", "bb00"), "model": "Luna"}

    _ensure_configuration(path, configuration)
    _ensure_configuration(path, configuration)

    with pytest.raises(ValueError, match="configuration changed"):
        _ensure_configuration(path, {"games": ("aa00",), "model": "Luna"})


def test_shared_benchmark_thread_counts_are_scoped_to_environment(
    tmp_path, monkeypatch
):
    from eggthreads import (
        ThreadsDB,
        append_message,
        create_child_thread,
        create_root_thread,
        set_thread_working_directory,
    )

    from arcagi3_physics.review import _thread_turns

    monkeypatch.chdir(tmp_path)
    benchmark = tmp_path / "benchmark"
    (benchmark / "benchmark.json").parent.mkdir(parents=True)
    (benchmark / "benchmark.json").write_text("{}")
    first_run = benchmark / "Physics aa00"
    second_run = benchmark / "Physics bb00"
    db = ThreadsDB(benchmark / ".egg" / "threads.sqlite")
    db.init_schema()
    root = create_root_thread(db, name="arc-agi-3-public")
    for run, game in ((first_run, "aa00"), (second_run, "bb00")):
        physics = create_child_thread(db, root, name=f"Physics {game}")
        critic = create_child_thread(db, physics, name="Critic")
        actor = create_child_thread(db, critic, name="Actor")
        set_thread_working_directory(
            db, critic, run / "workspace" / "critic-repository"
        )
        set_thread_working_directory(db, actor, run / "workspace" / "innerContext")
        append_message(db, critic, "assistant", "reviewed")
        append_message(
            db,
            actor,
            "user",
            "begin turn",
            extra={"eggopt_actor_critic_key": "eggopt.actor-critic.turn.v1:test"},
        )
        append_message(db, actor, "assistant", "acted")
    db.close()

    assert _thread_turns(first_run / ".egg" / "threads.sqlite", run_dir=first_run) == (
        1,
        1,
    )


def test_environment_failure_isolated_into_durable_status(tmp_path):
    import json

    from eggflow import TaskError

    from arcagi3_physics.benchmark import _failure, _RunEnvironment

    failure = _failure(
        "aa00",
        tmp_path / "Physics aa00",
        "thread-aa00",
        0.0,
        TaskError("provider failed", result=None),
    )

    assert failure.status == "failed"
    assert Path(failure.traceback_path).is_file()
    status = json.loads((tmp_path / "Physics aa00" / "status.json").read_text())
    assert status["status"] == "failed"
    assert "provider failed" in status["error"]

    assert _RunEnvironment.cacheable is False


def test_completed_environment_result_skips_reexecution(tmp_path):
    import json

    from arcagi3_physics.benchmark import _completed_result

    run_dir = tmp_path / "Physics aa00"
    run_dir.mkdir(parents=True)
    value = {
        "physics_thread_id": "thread-aa00",
        "stopping_reason": "won",
        "rounds": 3,
        "head": "abc",
        "value": {"actions": 7},
    }
    (run_dir / "result.json").write_text(json.dumps(value))

    result = _completed_result("aa00", run_dir, "thread-aa00", value)

    assert result.status == "completed"
    assert result.actions == 7
    assert result.stopping_reason == "won"

    value["stopping_reason"] = "game_over"
    terminal = _completed_result("aa00", run_dir, "thread-aa00", value)
    assert terminal.status == "terminal"
    assert terminal.stopping_reason == "game_over"


def test_interrupted_summary_recovers_durable_environment_status(tmp_path):
    import json

    from arcagi3_physics.benchmark import _status_results

    run = tmp_path / "benchmark"
    first = run / "Physics aa00"
    first.mkdir(parents=True)
    (first / "status.json").write_text(
        json.dumps(
            {
                "game": "aa00",
                "status": "completed",
                "run_dir": str(first),
                "physics_thread_id": "thread-aa00",
                "actions": 4,
            }
        )
    )

    results = _status_results(
        ("aa00", "bb00"),
        {"aa00": "thread-aa00", "bb00": "thread-bb00"},
        run,
    )

    assert results[0].status == "completed"
    assert results[0].actions == 4
    assert results[1].status == "interrupted"


def test_arc_requires_complete_actor_tool_set(tmp_path):
    actor = Agent(
        ScriptedLLM(),
        {"role": "actor"},
        auto_approve_tools=True,
        allowed_tools=frozenset({"bash", "python_exec"}),
    )

    with pytest.raises(
        ValueError,
        match="add_local_file_to_model_context.*answer_user_while_preserving_llm_turn",
    ):
        arc_physics(game="fake", seed=0, environments_dir=tmp_path, actor=actor)


def test_grid_to_png_renders_arc_palette_without_domain_imports(tmp_path):
    import json
    import struct
    import subprocess

    helper = tmp_path / "gridToPng.py"
    helper.write_text(GRID_TO_PNG)
    source = tmp_path / "state.json"
    source.write_text(json.dumps({"grid": [[[0, 8], [9, 15]]]}))
    output = tmp_path / "scratch" / "grid.png"

    completed = subprocess.run(
        ["python", "-I", str(helper), str(source), str(output), "--scale", "3"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    data = output.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (6, 6)


def test_grid_to_png_selects_latest_canonical_timeline_state(tmp_path):
    import json
    import subprocess

    helper = tmp_path / "gridToPng.py"
    helper.write_text(GRID_TO_PNG)
    source = tmp_path / "canonical-input.json"
    source.write_text(
        json.dumps(
            {
                "timeline": [
                    {"grid": [[[0]]]},
                    {
                        "state": {"grid": [[[0]]]},
                        "action": {"action": 1},
                        "next_state": {"grid": [[[8]]]},
                    },
                ]
            }
        )
    )
    output = tmp_path / "grid.png"

    completed = subprocess.run(
        ["python", "-I", str(helper), str(source), str(output)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_grid_to_png_accepts_initial_only_canonical_timeline(tmp_path):
    import json
    import subprocess

    helper = tmp_path / "gridToPng.py"
    helper.write_text(GRID_TO_PNG)
    source = tmp_path / "canonical-input.json"
    source.write_text(
        json.dumps(
            {
                "timeline": [
                    {
                        "grid": [[[0, 8], [9, 15]]],
                        "legal_actions": [1],
                        "levels_completed": 0,
                        "state": "NOT_FINISHED",
                        "win_levels": 1,
                    }
                ]
            }
        )
    )
    output = tmp_path / "grid.png"

    completed = subprocess.run(
        ["python", "-I", str(helper), str(source), str(output)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_single_run_persists_exact_local_environment_version(tmp_path):
    import json
    from argparse import Namespace

    from arcagi3_physics.run import _ensure_run_configuration

    environments = tmp_path / "environments"
    metadata = environments / "ls20" / "version" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps(
            {
                "game_id": "ls20-version",
                "baseline_actions": [1],
                "date_downloaded": "2026-01-01T00:00:00Z",
            }
        )
    )
    run_dir = tmp_path / "run"
    _ensure_run_configuration(
        Namespace(
            run_dir=run_dir,
            game="ls20",
            seed=7,
            environments_dir=environments,
        )
    )

    assert json.loads((run_dir / "run.json").read_text()) == {
        "environments_dir": str(environments.resolve()),
        "game": "ls20",
        "game_id": "ls20-version",
        "seed": 7,
    }


def test_environment_metadata_selects_newest_or_explicit_version(tmp_path):
    import json

    from arcagi3_physics.environment import environment_metadata

    for version, downloaded in (
        ("old", "2026-01-01T00:00:00Z"),
        ("current", "2026-08-01T00:00:00Z"),
    ):
        path = tmp_path / "ls20" / version / "metadata.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "game_id": f"ls20-{version}",
                    "date_downloaded": downloaded,
                }
            )
        )

    _, current = environment_metadata("ls20", tmp_path)
    _, old = environment_metadata("ls20-old", tmp_path)

    assert current["game_id"] == "ls20-current"
    assert old["game_id"] == "ls20-old"


def test_legacy_run_without_recorded_version_cannot_resume_after_sync(tmp_path):
    from argparse import Namespace

    from arcagi3_physics.run import _ensure_run_configuration

    run_dir = tmp_path / "legacy"
    (run_dir / "workspace" / "critic-repository" / ".git").mkdir(parents=True)

    with pytest.raises(ValueError, match="legacy ARC run.*new run directory"):
        _ensure_run_configuration(
            Namespace(
                run_dir=run_dir,
                game="ls20",
                seed=0,
                environments_dir=tmp_path / "environments",
            )
        )


def test_official_game_id_selects_api_version_and_requires_local_copy(tmp_path):
    from types import SimpleNamespace

    from arcagi3_physics.launch import official_game_id

    class Arcade:
        def get_environments(self):
            return [SimpleNamespace(game_id="ls20-current")]

    def factory(**_):
        return Arcade()

    with pytest.raises(FileNotFoundError, match="ls20-current"):
        official_game_id("ls20", tmp_path, arcade_factory=factory)

    metadata = tmp_path / "ls20" / "current" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"game_id": "wrong-version"}')

    with pytest.raises(ValueError, match="metadata does not match"):
        official_game_id("ls20", tmp_path, arcade_factory=factory)

    metadata.write_text('{"game_id": "ls20-current"}')

    assert official_game_id("ls20", tmp_path, arcade_factory=factory) == "ls20-current"


def test_official_game_id_rejects_ambiguous_or_explicit_default(tmp_path):
    from types import SimpleNamespace

    from arcagi3_physics.launch import official_game_id

    class Arcade:
        def get_environments(self):
            return [
                SimpleNamespace(game_id="ls20-one"),
                SimpleNamespace(game_id="ls20-two"),
            ]

    with pytest.raises(ValueError, match="ARC_GAME"):
        official_game_id("ls20-old", tmp_path)
    with pytest.raises(RuntimeError, match="multiple versions"):
        official_game_id("ls20", tmp_path, arcade_factory=lambda **_: Arcade())

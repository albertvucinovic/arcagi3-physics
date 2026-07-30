from __future__ import annotations

from pathlib import Path

import pytest
from eggflow import FlowExecutor, Task, TaskStore
from eggopt import ActorCritic, Agent, PhysicsStrategy
from eggthreads import ThreadsDB, ToolRegistry, list_threads, load_thread_projection

from arcagi3_physics.environment import Execute, Observe
from arcagi3_physics.run import build_parser
from arcagi3_physics.solver import arc_physics
from arcagi3_physics.tasks import (
    Backtest,
    Deliberate,
    Hypothesize,
    PublishData,
    deterministic_commitment,
)
from arcagi3_physics.world import (
    WORLD_MODEL_TEMPLATE,
    ensure_world_model,
    run_backtest,
    run_bfs,
    snapshot_world_model,
)

WORLD_MODEL = """
def ground(history):
    latest = history[-1]
    observation = latest.get("observation", latest)
    return observation["position"]

def step(state, action):
    value = action['action'] if isinstance(action, dict) else action
    return state + value

def render(state):
    return {"position": state, "legal_actions": (1, 2)}

def is_goal(state):
    return state >= 2
"""


class ScriptedLLM:
    current_model_key = "test-model"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0
        self.messages = []

    def set_model(self, key):
        self.current_model_key = key

    def set_model_with_config(self, key, _config):
        self.current_model_key = key

    async def astream_chat(self, messages, **_kwargs):
        self.calls += 1
        self.messages.append(messages)
        yield {
            "type": "message",
            "role": "assistant",
            "content": next(self.replies),
            "stop_reason": "end_turn",
        }


def test_role_prompts_reference_repl_without_embedding_game_data():
    marker = "GAME_PAYLOAD_MUST_NOT_ENTER_PROMPT"
    timeline = ({"marker": marker, "legal_actions": (1,)},)
    tools = ToolRegistry()
    agent = Agent(
        ScriptedLLM([]),
        {"role": "prompt-test"},
        tools=tools,
        allowed_tools=frozenset(),
    )
    state = {"actor_thread_id": "actor", "feedback": ""}

    modeler = Hypothesize(agent, timeline, None, {"marker": marker}, "workspace")
    planner = Deliberate(agent, timeline, marker, {"marker": marker}, "workspace")
    modeler_prompt = modeler._prompt(1, state)
    planner_prompt = planner._prompt(1, state)

    assert marker not in modeler_prompt.prompt
    assert marker not in planner_prompt.prompt
    assert "`timeline`" in modeler_prompt.prompt
    assert "`latest_observation`" in modeler_prompt.prompt
    assert "`world_model_source`" in planner_prompt.prompt
    assert "`legal_actions`" in planner_prompt.prompt
    assert marker in str(modeler_prompt.values)
    assert marker in str(planner_prompt.values)
    assert "python_repl" in modeler._prompt(2, {**state, "feedback": "Revise."})
    assert "python_repl" in planner._prompt(2, {**state, "feedback": "Revise."})


def test_publish_data_assigns_authoritative_repl_variables(tmp_path, monkeypatch):
    from eggopt.context import _bind_evaluation_runtime, _evaluation_scope
    from eggthreads import ThreadsDB, create_root_thread

    monkeypatch.chdir(tmp_path)
    db = ThreadsDB(tmp_path / "threads.sqlite")
    db.init_schema()
    thread_id = create_root_thread(db, name="Planner")
    runtime_key = "arc-publish-test"
    _bind_evaluation_runtime(runtime_key, db)
    captured = {}
    namespace = {}
    tools = ToolRegistry()

    def python_repl(arguments, context):
        assert context.thread_id == thread_id
        exec(arguments["code"], namespace)  # noqa: S102 - exercise generated REPL code
        if all(name in namespace for name in ("timeline", "legal_actions")):
            captured.update(
                {name: namespace[name] for name in ("timeline", "legal_actions")}
            )
        return "Published"

    tools.register(
        "python_repl",
        "Test REPL",
        {"type": "object", "properties": {"code": {"type": "string"}}},
        python_repl,
        accepts_context=True,
    )
    task = PublishData(
        thread_id,
        {"timeline": ({"position": 0},), "legal_actions": (1, 2)},
        1,
        "planner",
        tools,
    )
    store = TaskStore(str(tmp_path / "flow.db"))
    try:
        with _evaluation_scope(
            {
                "evaluation_thread_id": thread_id,
                "_runtime_key": runtime_key,
                "_evaluation_key": "publish",
                "_context_limit": None,
            }
        ):
            __import__("asyncio").run(FlowExecutor(store).run(task))
    finally:
        store.close()
        db.close()

    assert captured == {
        "timeline": [{"position": 0}],
        "legal_actions": [1, 2],
    }


def test_backtest_publishes_counterexample_instead_of_embedding_feedback(
    tmp_path, monkeypatch
):
    from eggopt.context import _bind_evaluation_runtime, _evaluation_scope
    from eggthreads import ThreadsDB, create_root_thread

    monkeypatch.chdir(tmp_path)
    Path(tmp_path, "world_model.py").write_text(WORLD_MODEL)
    marker = "COUNTEREXAMPLE_MUST_NOT_ENTER_FEEDBACK"
    timeline = (
        {"position": 0, "legal_actions": (1,)},
        {
            "intent": {"action": 1},
            "observation": {"position": marker, "legal_actions": (1,)},
        },
    )
    db = ThreadsDB(tmp_path / "threads.sqlite")
    db.init_schema()
    actor_id = create_root_thread(db, name="Modeler")
    runtime_key = "arc-counterexample-test"
    _bind_evaluation_runtime(runtime_key, db)
    namespace = {}
    tools = ToolRegistry()

    def python_repl(arguments, _context):
        exec(arguments["code"], namespace)  # noqa: S102 - exercise generated code
        return "Published"

    tools.register(
        "python_repl",
        "Test REPL",
        {"type": "object", "properties": {"code": {"type": "string"}}},
        python_repl,
        accepts_context=True,
    )
    store = TaskStore(str(tmp_path / "flow.db"))
    try:
        with _evaluation_scope(
            {
                "evaluation_thread_id": actor_id,
                "_runtime_key": runtime_key,
                "_evaluation_key": "counterexample",
                "_context_limit": None,
            }
        ):
            result = __import__("asyncio").run(
                FlowExecutor(store).run(
                    Backtest(
                        timeline,
                        str(tmp_path),
                        actor_thread_id=actor_id,
                        round_number=1,
                        tools=tools,
                    )
                )
            )
    finally:
        store.close()
        db.close()

    assert result.decision == "revise"
    assert marker not in result.feedback
    assert (
        namespace["new_evidence"]["counterexamples"][0]["actual"]["position"] == marker
    )


def test_world_model_file_backtest_and_plan(tmp_path):
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
    assert report == {"matches": 1, "counterexamples": []}

    plan = run_bfs(source, timeline, (1, 2), tmp_path / "bfs")
    assert plan == (
        {
            "action": 1,
            "prediction": {"position": 2, "legal_actions": (1, 2)},
        },
    )


def test_deterministic_commitment_prefers_goal_plan(tmp_path):
    timeline = ({"position": 0, "legal_actions": (1, 2)},)
    assert deterministic_commitment(WORLD_MODEL, timeline, tmp_path)[0]["action"] == 2


def test_world_model_skeleton_documents_exact_single_model_api(tmp_path):
    path = ensure_world_model(tmp_path)
    source = path.read_text()

    assert source == WORLD_MODEL_TEMPLATE
    assert "def ground(history):" in source
    assert "def step(state, action):" in source
    assert "def render(state):" in source
    assert "def is_goal(state):" in source
    assert "Public observation shape" in source
    assert '"grid"' in source
    assert '"legal_actions"' in source
    assert '"state"' in source
    assert '"levels_completed"' in source
    assert '"win_levels"' in source
    assert "action['action']" in source
    assert "HYPOTHESES" not in source


def test_backtest_missing_file_returns_actor_revision(tmp_path):
    critic = Backtest(({"position": 0},), str(tmp_path))
    store = TaskStore(str(tmp_path / "flow.db"))
    try:
        result = __import__("asyncio").run(FlowExecutor(store).run(critic))
    finally:
        store.close()
    assert result.decision == "revise"
    assert "world_model.py" in result.feedback


def test_hypothesize_returns_world_model_file_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    modeler = ScriptedLLM(["world_model.py saved"] * 10)

    class PrewrittenHypothesize(Hypothesize):
        def run(self):
            inner = Path(self.workspace, "innerContext")
            inner.mkdir(parents=True, exist_ok=True)
            Path(inner, "world_model.py").write_text(WORLD_MODEL)
            result = yield ActorCritic(
                actor=self.agent,
                critic=Backtest(self.timeline, str(inner)),
                actor_prompt=self._prompt,
                max_rounds=self.max_rounds,
                names=("Modeler", "Backtest"),
            )
            return result.value

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
            )
        ),
        test=lambda **_: Value(None),
        deliberate=lambda **_: Value(None),
        execute=lambda **_: Value(None),
        identity={"test": "world-model-file"},
    ).run(run_dir=tmp_path / "run", max_cycles=1)

    assert result.hypotheses == WORLD_MODEL
    assert modeler.calls == 1
    assert "position" not in str(modeler.messages[0])
    assert "`timeline`" in str(modeler.messages[0])
    db = ThreadsDB(tmp_path / "run" / ".egg" / "threads.sqlite")
    try:
        actor_id = next(
            thread.thread_id for thread in list_threads(db) if thread.name == "Modeler"
        )
        projection = load_thread_projection(db, actor_id)
        prompt = next(
            message.payload["content"]
            for message in projection.messages
            if message.payload.get("role") == "user"
            and message.payload.get("eggopt_actor_critic_key")
        )
        repl_calls = [
            message.payload
            for message in projection.messages
            if message.payload.get("role") == "user"
            and message.payload.get("synthetic_user_tool_request")
        ]
        assert "position" not in prompt
        assert "`timeline`" in prompt
        assert len(repl_calls) == 1
        assert all(
            call["tool_calls"][0]["function"]["name"] == "python_repl"
            for call in repl_calls
        )
        publish_code = repl_calls[0]["tool_calls"][0]["function"]["arguments"]
        assert "position" in publish_code
        assert "timeline" in publish_code
    finally:
        db.close()


def test_deliberate_publishes_planner_data_without_prompt_dump(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    marker = "PLANNER_PAYLOAD_MUST_NOT_ENTER_PROMPT"
    timeline = ({"position": marker, "legal_actions": (1,)},)
    planner = ScriptedLLM(['{"intents":[{"action":1,"prediction":{"position":1}}]}'])

    class Value(Task):
        def __init__(self, value):
            self.value = value

        def run(self):
            return self.value

    result = PhysicsStrategy(
        observe=lambda **_: Value(timeline[0]),
        hypothesize=lambda **_: Value(WORLD_MODEL),
        test=lambda **_: Value(None),
        deliberate=lambda timeline, hypotheses, evidence, workspace, **_: Deliberate(
            Agent(planner, {"role": "planner"}),
            timeline,
            hypotheses,
            evidence,
            workspace,
        ),
        execute=lambda **_: Value(None),
        identity={"test": "planner-repl-data"},
    ).run(run_dir=tmp_path / "run", max_actions=1, max_cycles=1)

    assert result.actions == 1
    assert marker not in str(planner.messages[0])
    assert "`world_model_source`" in str(planner.messages[0])
    db = ThreadsDB(tmp_path / "run" / ".egg" / "threads.sqlite")
    try:
        actor_id = next(
            thread.thread_id for thread in list_threads(db) if thread.name == "Planner"
        )
        projection = load_thread_projection(db, actor_id)
        prompt = next(
            message.payload["content"]
            for message in projection.messages
            if message.payload.get("role") == "user"
            and message.payload.get("eggopt_actor_critic_key")
        )
        repl_calls = [
            message.payload
            for message in projection.messages
            if message.payload.get("role") == "user"
            and message.payload.get("synthetic_user_tool_request")
        ]
        assert marker not in prompt
        assert len(repl_calls) == 1
        assert marker in repl_calls[0]["tool_calls"][0]["function"]["arguments"]
        assert (
            "world_model_source"
            in repl_calls[0]["tool_calls"][0]["function"]["arguments"]
        )
        assert "def ground" in repl_calls[0]["tool_calls"][0]["function"]["arguments"]
    finally:
        db.close()


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


def test_arc_physics_requires_repl_for_each_reasoning_role(tmp_path):
    modeler = Agent(
        ScriptedLLM([]),
        {"role": "modeler"},
        auto_approve_tools=True,
        allowed_tools=frozenset({"bash"}),
    )
    planner = Agent(
        ScriptedLLM([]),
        {"role": "planner"},
        auto_approve_tools=True,
        allowed_tools=frozenset({"python_repl"}),
    )
    with pytest.raises(ValueError, match="modeler needs python_repl"):
        arc_physics(
            game="fake",
            seed=0,
            environments_dir=tmp_path,
            modeler=modeler,
            planner=planner,
        )


def test_offline_runner_defaults_to_ls20_seed_zero():
    arguments = build_parser().parse_args([])

    assert arguments.game == "ls20"
    assert arguments.seed == 0
    assert arguments.run_dir == Path("runs/physics-ls20-seed0")
    assert arguments.modeler_model == "Pro: GPT-5.6 Sol max"
    assert arguments.planner_model == "Pro: GPT-5.6 Sol max"
    source = Path("arcagi3_physics/run.py").read_text()
    assert 'allowed_tools=frozenset({"bash", "python_exec", "python_repl"})' in source
    assert 'allowed_tools=frozenset({"python_repl"})' in source


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

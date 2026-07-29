from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eggflow import Task
from eggopt import ActorCritic, Agent, ThreadTool
from eggopt.identity import canonical_json, digest_payload

from .world import ensure_world_model, run_backtest, run_bfs, snapshot_world_model


@dataclass
class PublishData(Task):
    """Publish authoritative domain values into one assigned role's REPL."""

    thread_id: str
    values: dict[str, Any]
    occurrence: int
    role: str
    tools: Any = field(repr=False, compare=False)

    def get_cache_key(self):
        return digest_payload(
            "arcagi3.physics.publish-data.v1",
            {
                "thread": self.thread_id,
                "values": self.values,
                "occurrence": self.occurrence,
                "role": self.role,
            },
        )

    def run(self):
        source = canonical_json(self.values, what="ARC role data")
        output = yield ThreadTool(
            self.tools,
            self.thread_id,
            "python_repl",
            {
                "code": (
                    "import json as _json\n"
                    f"_arc_role_data = _json.loads({source!r})\n"
                    "globals().update(_arc_role_data)\n"
                    f"_arc_expected = {tuple(sorted(self.values))!r}\n"
                    "assert all(_name in globals() for _name in _arc_expected)\n"
                    "del _arc_role_data\n"
                    "del _arc_expected\n"
                    f"print('Published ARC data: {', '.join(sorted(self.values))}')"
                )
            },
            occurrence=self.occurrence,
            origin=f"arcagi3.physics.{self.role}",
        )
        _require_repl_success(output, "publish ARC data")
        return output


@dataclass
class PublishPrompt(Task):
    thread_id: str
    values: dict[str, Any]
    prompt: str
    occurrence: int
    role: str
    tools: Any = field(repr=False, compare=False)

    def get_cache_key(self):
        return digest_payload(
            "arcagi3.physics.publish-prompt.v1",
            {
                "thread": self.thread_id,
                "values": self.values,
                "prompt": self.prompt,
                "occurrence": self.occurrence,
                "role": self.role,
            },
        )

    def run(self):
        yield PublishData(
            self.thread_id,
            self.values,
            self.occurrence,
            self.role,
            self.tools,
        )
        return self.prompt


@dataclass
class Hypothesize(Task):
    """Continue one file-editing Modeler and snapshot its accepted program."""

    agent: Agent
    timeline: tuple[Any, ...]
    previous: Any
    evidence: Any
    workspace: str
    max_rounds: int = 4

    def get_cache_key(self):
        return digest_payload(
            "arcagi3.physics.hypothesize.v2",
            {
                "agent": self.agent.task_identity,
                "timeline": self.timeline,
                "previous": self.previous,
                "evidence": self.evidence,
                "data_transport": "python-repl-v1",
            },
        )

    def run(self):
        ensure_world_model(Path(self.workspace) / "innerContext")
        result = yield ActorCritic(
            actor=self.agent,
            critic=Backtest(self.timeline, self.workspace, tools=self.agent.tools),
            actor_prompt=self._prompt,
            max_rounds=self.max_rounds,
            names=("Modeler", "Backtest"),
        )
        if not result.accepted:
            raise RuntimeError("Modeler exhausted its correction rounds")
        return snapshot_world_model(result.workspace)

    def _prompt(self, round_number, state):
        if round_number > 1:
            return (
                state["feedback"]
                + " Inspect the referenced variables with `python_repl` before revising."
            )
        return PublishPrompt(
            state["actor_thread_id"],
            {
                "timeline": self.timeline,
                "latest_observation": _observation(self.timeline),
                "new_evidence": self.evidence,
            },
            self._instructions(),
            round_number,
            "modeler",
            self.agent.tools,
        )

    def _instructions(self):
        previous = "exists" if self.previous is not None else "does not exist"
        return (
            "Act as a physicist studying one hidden world. ./world_model.py is one "
            "provisional hypothesis, not a hypothesis collection. The file has already "
            "been created with complete documentation and required function stubs. Read "
            "it first. Authoritative game data is in your persistent Python REPL: "
            "`timeline`, `latest_observation`, and `new_evidence`. Inspect or slice these "
            "variables with `python_repl`; do not look for their contents in this prompt. "
            f"A previous accepted world-model snapshot {previous}. Then use bash or "
            "Python tools to implement or revise the file. The file—"
            "not your chat response—is the result. Preserve exactly this public API:\n"
            "  ground(history) -> current latent state\n"
            "  step(state, action) -> predicted next latent state\n"
            "  render(state) -> complete predicted public observation\n"
            "  is_goal(state) -> bool\n"
            "Grounding, mechanism, rendering, and goal may all be revised. Do not add a "
            "HYPOTHESES collection or hypothesis arguments. Probabilistic uncertainty, "
            "if needed, belongs inside this one model's state. Do not access the real "
            "environment or hidden game implementation. When world_model.py is saved, "
            "answer briefly."
        )


@dataclass
class Backtest(Task):
    timeline: tuple[Any, ...]
    workspace: str
    answer: Any = None
    actor_thread_id: str | None = None
    round_number: int | None = None
    tools: Any = field(default=None, repr=False, compare=False)

    def get_cache_key(self):
        return digest_payload(
            "arcagi3.physics.backtest-review.v2",
            {"timeline": self.timeline},
        )

    def run(self):
        try:
            source = snapshot_world_model(self.workspace)
            report = yield RunBacktest(source, self.timeline, Path(self.workspace))
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
            return {
                "decision": "revise",
                "feedback": (
                    f"world_model.py is missing or invalid: {exc}. Use your tools to "
                    "create or repair ./world_model.py, save it to disk, then answer briefly."
                ),
            }
        if report["counterexamples"]:
            if self.actor_thread_id is None or self.tools is None:
                raise RuntimeError("Backtest requires its assigned Modeler thread")
            yield PublishData(
                self.actor_thread_id,
                {"new_evidence": {"counterexamples": report["counterexamples"]}},
                self.round_number or 1,
                "modeler-counterexample",
                self.tools,
            )
            return {
                "decision": "revise",
                "feedback": (
                    "Reality contradicts world_model.py. The authoritative counterexamples "
                    "are in Python variable `new_evidence`. Revise the file; the mismatch "
                    "may indict grounding, mechanism, rendering, or goal."
                ),
            }
        return {
            "decision": "accept",
            "feedback": "world_model.py replays the complete Timeline.",
        }


@dataclass
class RunBacktest(Task):
    source: str
    timeline: tuple[Any, ...]
    workspace: Path

    def get_cache_key(self):
        return digest_payload(
            "arcagi3.physics.backtest.v2",
            {"source": self.source, "timeline": self.timeline},
        )

    def run(self):
        return run_backtest(self.source, self.timeline, self.workspace)


@dataclass
class Test(Task):
    world_model: str
    timeline: tuple[Any, ...]
    commitment: Any
    workspace: str

    def get_cache_key(self):
        return digest_payload(
            "arcagi3.physics.test.v2",
            {
                "world_model": self.world_model,
                "timeline": self.timeline,
                "commitment": self.commitment,
            },
        )

    def run(self):
        report = yield RunBacktest(
            self.world_model, self.timeline, Path(self.workspace)
        )
        if report["counterexamples"]:
            return {
                "counterexamples": report["counterexamples"],
                "commitment": self.commitment,
            }
        if self.commitment is not None:
            actual = self.timeline[-1]["observation"]
            predictions = self.commitment.get("predictions")
            prediction = self.commitment.get("prediction")
            expected = tuple(predictions) if predictions is not None else (prediction,)
            if actual not in expected:
                return {"prediction_mismatch": {"expected": expected, "actual": actual}}
        return None


@dataclass
class Deliberate(Task):
    agent: Agent
    timeline: tuple[Any, ...]
    world_model: str
    evidence: Any
    workspace: str
    max_rounds: int = 3

    def get_cache_key(self):
        return digest_payload(
            "arcagi3.physics.deliberate.v2",
            {
                "agent": self.agent.task_identity,
                "timeline": self.timeline,
                "world_model": self.world_model,
                "evidence": self.evidence,
                "data_transport": "python-repl-v1",
            },
        )

    def run(self):
        result = yield ActorCritic(
            actor=self.agent,
            critic=ValidateCommitment(self.timeline),
            actor_prompt=self._prompt,
            max_rounds=self.max_rounds,
            names=("Planner", "Plan Review"),
        )
        if not result.accepted:
            raise RuntimeError("Planner exhausted its correction rounds")
        return _commitment(result.answer)

    def _prompt(self, round_number, state):
        if round_number > 1:
            return (
                state["feedback"]
                + " Reinspect the named persistent `python_repl` variables if needed."
            )
        legal = _observation(self.timeline).get("legal_actions", ())
        snapshot = hashlib.sha256(self.world_model.encode()).hexdigest()
        return PublishPrompt(
            state["actor_thread_id"],
            {
                "timeline": self.timeline,
                "latest_observation": _observation(self.timeline),
                "legal_actions": legal,
                "world_model_source": self.world_model,
                "new_evidence": self.evidence,
            },
            (
                "Authoritative game data is in your persistent Python REPL variables "
                "`timeline`, `latest_observation`, `legal_actions`, "
                "`world_model_source`, and `new_evidence`. Inspect or slice them with "
                "`python_repl`; do not look for their contents in this prompt. "
                f"The world-model snapshot SHA-256 is `{snapshot}`. Choose a "
                "goal-reaching plan supported by that model when credible. If there is "
                "no credible plan, choose one cheap legal discovery action whose outcome "
                "would most improve or falsify the current model. Every intent must freeze "
                "the model's predicted public observation before execution.\n\n"
                'Return only strict JSON: {"intents":[...]}, or {"intents":null}.'
            ),
            round_number,
            "planner",
            self.agent.tools,
        )


@dataclass
class ValidateCommitment(Task):
    timeline: tuple[Any, ...]
    answer: Any = None

    def run(self):
        try:
            intents = _commitment(self.answer)
            if intents is None:
                return {"decision": "accept", "feedback": "No responsible action."}
            legal = set(_observation(self.timeline).get("legal_actions", ()))
            if any(int(intent["action"]) not in legal for intent in intents):
                raise ValueError("commitment contains an illegal action")
            if any(
                "prediction" not in intent and "predictions" not in intent
                for intent in intents
            ):
                raise ValueError("every intent must freeze prediction or predictions")
        except (TypeError, ValueError) as exc:
            return {"decision": "revise", "feedback": f"Commitment is invalid: {exc}"}
        return {"decision": "accept", "feedback": "Commitment is legal and testable."}


def deterministic_commitment(world_model, timeline, workspace, *, max_depth=12):
    legal = tuple(_observation(timeline).get("legal_actions", ()))
    plan = run_bfs(
        world_model,
        timeline,
        legal,
        Path(workspace) / "planning",
        max_depth=max_depth,
    )
    return plan


def _commitment(answer: Any):
    payload = json.loads(answer) if isinstance(answer, str) else answer
    if not isinstance(payload, dict) or set(payload) != {"intents"}:
        raise ValueError("response must contain only intents")
    intents = payload["intents"]
    if intents is None:
        return None
    if not isinstance(intents, list) or not intents:
        raise ValueError("intents must be a non-empty list or null")
    if any(not isinstance(intent, dict) for intent in intents):
        raise ValueError("each intent must be an object")
    return tuple(intents)


def _observation(timeline: tuple[Any, ...]) -> dict[str, Any]:
    latest = timeline[-1]
    return latest.get("observation", latest)


def _require_repl_success(output: Any, operation: str) -> None:
    text = str(output).strip()
    if text.startswith(("Error:", "--- INTERRUPTED ---")):
        raise RuntimeError(f"could not {operation}: {text}")

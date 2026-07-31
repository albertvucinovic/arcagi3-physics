from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .world import canonical_plans, run_backtest, run_planner


def write_actor_files(workspace: str | Path, timeline) -> None:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    _write_if_missing(workspace / "INSTRUCTIONS.md", ACTOR_INSTRUCTIONS)
    _write_json(workspace / "canonical-input.json", {"timeline": timeline})
    _write_if_missing(workspace / ".gitignore", "scratch/\n__pycache__/\n*.pyc\n")
    _write_if_missing(workspace / "backtest.py", BACKTEST_WRAPPER)
    _write_if_missing(workspace / "plan.py", PLAN_WRAPPER)
    _write_if_missing(workspace / "commit.py", COMMIT_WRAPPER)


ACTOR_INSTRUCTIONS = """# ARC Physics Actor

`world_model.py` is your current theory. Improve it. Every hypothesis is a
matching `step_<suffix>(state, action)` and `reward_<suffix>(state)` pair.
`step_*` predicts the next complete public observation; `reward_*` assigns a
finite utility and therefore defines that hypothesis's goal.

Canonical evidence is copied into `canonical-input.json` as an append-only
Timeline. The first item is the initial state; each later item is
`{"state": ..., "action": intent, "next_state": ...}`.

Use the supplied instruments:

- `python backtest.py`: report every model's Timeline mismatches.
- `python plan.py`: report goal plans for every valid model and shortest
  discrimination plans for model subsets, whether or not the models survive
  the current Timeline.
- `python commit.py PLAN_ID`: write the selected canonical non-empty plan to
  `committed-plan.json` and make the Git commit. This must be your final mutation
  before answering; the Critic rejects a dirty repository.

An intent is an action plus predictions keyed by model suffix. An experiment
may contain a shared setup prefix; execution stops after the first actual
prediction branch or any earlier mismatch. You cannot execute real actions.
The trusted Critic pulls Git HEAD and independently repeats the full pipeline.
Deleting `.git` requests restoration from the Critic's history copy; it never
rewinds the real game's canonical state.
"""

BACKTEST_WRAPPER = """from arcagi3_physics.instruments import actor_backtest

if __name__ == "__main__":
    actor_backtest()
"""

PLAN_WRAPPER = """from arcagi3_physics.instruments import actor_plan

if __name__ == "__main__":
    actor_plan()
"""

COMMIT_WRAPPER = """import sys
from arcagi3_physics.instruments import actor_commit

if __name__ == "__main__":
    actor_commit(sys.argv[1] if len(sys.argv) > 1 else "")
"""


def actor_backtest() -> None:
    workspace = Path.cwd()
    timeline = _input(workspace)
    source = (workspace / "world_model.py").read_text()
    report = run_backtest(source, timeline, workspace / "scratch" / "backtest")
    _write_json(workspace / "backtest-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def actor_plan() -> None:
    workspace = Path.cwd()
    timeline = _input(workspace)
    source = (workspace / "world_model.py").read_text()
    report = run_planner(source, timeline, workspace / "scratch" / "planner")
    plans = [
        {"plan_id": f"plan-{index}", "plan": plan}
        for index, plan in enumerate(canonical_plans(report), start=1)
    ]
    document = {**report, "canonical_plans": plans}
    _write_json(workspace / "plan-report.json", document)
    print(json.dumps(document, indent=2, sort_keys=True))


def actor_commit(plan_id: str) -> None:
    workspace = Path.cwd()
    report_path = workspace / "plan-report.json"
    if not report_path.is_file():
        raise SystemExit("Run python plan.py before commit.py")
    report = json.loads(report_path.read_text())
    selected = next(
        (
            item["plan"]
            for item in report.get("canonical_plans", ())
            if item["plan_id"] == plan_id
        ),
        None,
    )
    if selected is None:
        raise SystemExit(f"Unknown plan_id: {plan_id!r}")
    if not selected.get("intents"):
        raise SystemExit("Committed plan must contain at least one intent")
    _write_json(workspace / "committed-plan.json", selected)
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Actor commits {plan_id}"],
        cwd=workspace,
        check=True,
    )


def _input(workspace):
    return tuple(
        json.loads((workspace / "canonical-input.json").read_text())["timeline"]
    )


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_if_missing(path, content):
    if not path.exists():
        path.write_text(content)

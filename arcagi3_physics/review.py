from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import termios
import tty
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from eggthreads import ThreadsDB, list_threads, load_thread_projection

_CLEAR = "\033[2J\033[H"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_RESET = "\033[0m"


@dataclass(frozen=True)
class Review:
    repository: Path
    timeline: tuple[Any, ...]
    report: dict[str, Any]
    head: str
    actor_turns: int
    critic_turns: int
    actor_commits: int
    critic_commits: int
    evaluation_reports: int
    evaluated_head: str | None
    model_count: int
    surviving_models: tuple[str, ...]
    generated_plans: int

    @property
    def transitions(self) -> int:
        return max(0, len(self.timeline) - 1)


def load_review(run_dir: str | Path) -> Review:
    """Load review state from the Critic repository and public thread APIs."""

    run = Path(run_dir).expanduser().resolve()
    repository = run / "workspace" / "critic-repository"
    if not (repository / ".git").is_dir():
        raise FileNotFoundError(f"Critic Git repository not found: {repository}")
    state_path = repository / ".trusted" / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Critic canonical state not found: {state_path}")
    state = json.loads(state_path.read_text())
    timeline = tuple(state.get("timeline") or ())
    if not timeline:
        raise ValueError("Critic canonical Timeline is empty")
    report = state.get("last_report") or {}
    if not isinstance(report, dict):
        report = {}
    heads = _commit_subjects(repository)
    actor_commits = sum(not subject.startswith("[physics]") for subject in heads)
    critic_commits = len(heads) - actor_commits
    actor_turns, critic_turns = _thread_turns(run / ".egg" / "threads.sqlite")
    if actor_turns == 0:
        actor_turns = actor_commits
    if critic_turns == 0:
        critic_turns = critic_commits
    evaluations = repository / ".trusted" / "evaluations"
    evaluation_files = (
        tuple(path for path in evaluations.glob("*.json") if path.is_file())
        if evaluations.is_dir()
        else ()
    )
    evaluated_head, evaluation = _latest_evaluation(evaluation_files, report)
    backtest = evaluation.get("backtest", {}) if isinstance(evaluation, dict) else {}
    planning = evaluation.get("planning", {}) if isinstance(evaluation, dict) else {}
    models = backtest.get("models", {}) if isinstance(backtest, dict) else {}
    surviving = (
        tuple(str(item) for item in backtest.get("surviving_models", ()))
        if isinstance(backtest, dict)
        else ()
    )
    plans = planning.get("plans", ()) if isinstance(planning, dict) else ()
    return Review(
        repository=repository,
        timeline=timeline,
        report=report,
        head=_git(repository, "rev-parse", "HEAD"),
        actor_turns=actor_turns,
        critic_turns=critic_turns,
        actor_commits=actor_commits,
        critic_commits=critic_commits,
        evaluation_reports=len(evaluation_files),
        evaluated_head=evaluated_head,
        model_count=len(models) if isinstance(models, dict) else 0,
        surviving_models=surviving,
        generated_plans=len(plans) if isinstance(plans, list) else 0,
    )


def frame(review: Review, index: int) -> dict[str, Any]:
    if index < 0 or index > review.transitions:
        raise IndexError("Timeline frame is out of range")
    if index == 0:
        state = review.timeline[0]
        action = None
        transition = None
    else:
        transition = review.timeline[index]
        state = transition["next_state"]
        action = transition.get("action")
    if not isinstance(state, dict):
        raise TypeError("ARC Timeline state must be a mapping")
    return {"index": index, "state": state, "action": action, "transition": transition}


def render(review: Review, index: int, *, color: bool = True) -> str:
    item = frame(review, index)
    state = item["state"]
    grid = _visible_grid(state.get("grid"))
    report = review.report
    plan = report.get("committed_plan") if isinstance(report, dict) else None
    models = report.get("compatible_models", ()) if isinstance(report, dict) else ()
    lines = [
        (
            f"ARC Physics review  frame {index}/{review.transitions}  "
            f"HEAD {review.head[:12]}"
        ),
        (
            f"Actor turns: {review.actor_turns}  Critic turns: {review.critic_turns}  "
            f"Actor commits: {review.actor_commits}  Critic commits: {review.critic_commits}"
        ),
        (
            f"Real actions: {review.transitions}  Evaluation reports: "
            f"{review.evaluation_reports}  Report stage: {report.get('stage', '-')}  "
            f"resolution: {report.get('resolution', '-')}"
        ),
        (
            f"Evaluated Actor HEAD: {(review.evaluated_head or '-')[:12]}  "
            f"models: {review.model_count}  surviving: {review.surviving_models}  "
            f"generated plans: {review.generated_plans}"
        ),
        (
            f"Game state: {state.get('state', '-')}  levels: "
            f"{state.get('levels_completed', '-')}/{state.get('win_levels', '-')}  "
            f"legal: {state.get('legal_actions', ())}"
        ),
        (
            "Arriving action: "
            + (json.dumps(item["action"], sort_keys=True) if item["action"] else "initial observation")
        ),
        (
            f"Plan: {(plan or {}).get('purpose', '-')}  models: "
            f"{(plan or {}).get('models', ())}  compatible: {models}"
        ),
        "",
    ]
    lines.extend(_render_grid(grid, color=color))
    lines.extend(
        [
            "",
            "←/→ or h/l: previous/next   Home/End: first/latest   r: reload   q: quit",
            f"Critic repository: {review.repository}",
        ]
    )
    return "\n".join(lines)


def review(run_dir: str | Path, *, stream: TextIO = sys.stdout) -> None:
    current = load_review(run_dir)
    index = current.transitions
    if not sys.stdin.isatty() or not stream.isatty():
        stream.write(render(current, index, color=False) + "\n")
        return
    with _raw_input(sys.stdin):
        stream.write(_HIDE_CURSOR)
        try:
            while True:
                stream.write(_CLEAR + render(current, index) + _RESET)
                stream.flush()
                key = _read_key(sys.stdin)
                if key in {"q", "escape"}:
                    return
                if key in {"left", "h"}:
                    index = max(0, index - 1)
                elif key in {"right", "l"}:
                    index = min(current.transitions, index + 1)
                elif key == "home":
                    index = 0
                elif key == "end":
                    index = current.transitions
                elif key == "r":
                    at_latest = index == current.transitions
                    current = load_review(run_dir)
                    index = current.transitions if at_latest else min(index, current.transitions)
        finally:
            stream.write(_SHOW_CURSOR + _RESET + "\n")
            stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review a live or completed ARC Physics run from its Critic repository."
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/physics-ls20-seed0")
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Print one frame and exit instead of opening the arrow-key viewer.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.frame is None:
            review(arguments.run_dir)
        else:
            state = load_review(arguments.run_dir)
            print(render(state, arguments.frame, color=sys.stdout.isatty()))
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Review unavailable: {exc}", file=sys.stderr)
        return 1
    return 0


def _thread_turns(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    db = ThreadsDB(path)
    try:
        actor = critic = 0
        for thread in list_threads(db):
            if thread.name not in {"Actor", "Critic"}:
                continue
            messages = load_thread_projection(db, thread.thread_id).messages
            turns = sum(message.payload.get("role") == "assistant" for message in messages)
            if thread.name == "Actor":
                actor += turns
            else:
                critic += turns
        return actor, critic
    finally:
        db.close()


def _commit_subjects(repository: Path) -> tuple[str, ...]:
    output = _git(repository, "log", "--format=%s")
    return tuple(line for line in output.splitlines() if line)


def _latest_evaluation(
    paths: tuple[Path, ...], report: dict[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    preferred = str(report.get("head") or "")
    selected = next((path for path in paths if path.stem == preferred), None)
    if selected is None and paths:
        selected = max(paths, key=lambda path: path.stat().st_mtime_ns)
    if selected is None:
        return None, {}
    try:
        value = json.loads(selected.read_text())
    except (OSError, json.JSONDecodeError):
        return selected.stem, {}
    return selected.stem, value if isinstance(value, dict) else {}


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    return completed.stdout.strip()


def _visible_grid(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        return ()
    layer = value[-1]
    if not isinstance(layer, (list, tuple)):
        return ()
    return tuple(tuple(int(cell) for cell in row) for row in layer)


def _render_grid(grid: tuple[tuple[int, ...], ...], *, color: bool) -> list[str]:
    if not grid:
        return ["(no grid in this frame)"]
    if not color:
        symbols = " .:-=+*#%@ABCDEF"
        return ["".join(symbols[cell % len(symbols)] for cell in row) for row in grid]
    palette = (16, 21, 196, 46, 226, 201, 208, 51, 244, 15, 160, 34, 27, 129, 220, 231)
    return [
        "".join(f"\033[48;5;{palette[cell % 16]}m  " for cell in row) + _RESET
        for row in grid
    ]


@contextlib.contextmanager
def _raw_input(stream: TextIO) -> Iterator[None]:
    fd = stream.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def _read_key(stream: TextIO) -> str:
    char = stream.read(1)
    if char != "\x1b":
        return char
    second = stream.read(1)
    if second not in {"[", "O"}:
        return "escape"
    third = stream.read(1)
    return _decode_key("\x1b" + second + third)


def _decode_key(sequence: str) -> str:
    return {
        "\x1b[D": "left",
        "\x1b[C": "right",
        "\x1b[H": "home",
        "\x1b[F": "end",
        "\x1bOH": "home",
        "\x1bOF": "end",
    }.get(sequence, "escape")


if __name__ == "__main__":
    raise SystemExit(main())

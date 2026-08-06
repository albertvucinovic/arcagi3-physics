from __future__ import annotations

import argparse
import bisect
import contextlib
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import termios
import tty
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from arc_agi.rendering import COLOR_MAP, hex_to_rgb
from eggthreads import (
    ThreadsDB,
    get_thread_working_directory,
    list_threads,
    load_thread_projection,
)

_CLEAR = "\033[2J\033[H"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_RESET = "\033[0m"
_DEFAULT_COLUMNS = 80
_DEFAULT_LINES = 24
_MIN_COLUMNS = 20
_PHYSICS_PREFIX = "[physics]"
_ACTOR_CRITIC_TURN_KEY = "eggopt.actor-critic.turn."

# ARC-AGI-3's canonical palette.  Foreground and background colors let one
# terminal cell represent two vertical game pixels without distorting the
# square 64x64 image.
_PALETTE = tuple(hex_to_rgb(COLOR_MAP[index]) for index in range(16))


@dataclass(frozen=True)
class TurnStats:
    actor_turn: int
    actor_head: str | None
    critic_head: str
    action_start: int
    action_end: int
    proposed_actions: int
    executed_actions: int
    stage: str
    resolution: str
    evaluation_report: bool
    evaluated_head: str | None
    model_count: int
    surviving_models: tuple[str, ...]
    matching_models: tuple[str, ...]
    generated_plans: int
    actor_commits: int
    critic_commits: int
    prediction_matches: tuple[bool | None, ...] = ()
    action_matching_models: tuple[tuple[str, ...] | None, ...] = ()

    @property
    def action_range(self) -> str:
        if self.executed_actions == 0:
            return "none"
        if self.action_start == self.action_end:
            return str(self.action_start)
        return f"{self.action_start}-{self.action_end}"


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
    turns: tuple[TurnStats, ...] = ()
    initial_head: str | None = None
    initial_actor_commits: int = 0
    initial_critic_commits: int = 0

    @property
    def transitions(self) -> int:
        return max(0, len(self.timeline) - 1)


@dataclass(frozen=True)
class _Commit:
    head: str
    subject: str
    timestamp: int


@dataclass(frozen=True)
class _ThreadActivity:
    actor_turns: int = 0
    critic_turns: int = 0
    actor_turn_started_at: tuple[float, ...] = ()


def load_review(run_dir: str | Path) -> Review:
    """Load review state from the Critic repository and public thread APIs."""

    run = Path(run_dir).expanduser().resolve()
    repository = run / "workspace" / "critic-repository"
    if not (repository / ".git").is_dir():
        raise FileNotFoundError(f"Critic Git repository not found: {repository}")
    head = _git(repository, "rev-parse", "HEAD")
    state = _git_json(repository, head, ".trusted/state.json")
    timeline = tuple(state.get("timeline") or ())
    if not timeline:
        raise ValueError("Critic canonical Timeline is empty")
    report = state.get("last_report") or {}
    if not isinstance(report, dict):
        report = {}
    commits = _commits(repository, head)
    heads = tuple(commit.subject for commit in commits)
    actor_commits = sum(not subject.startswith(_PHYSICS_PREFIX) for subject in heads)
    critic_commits = len(heads) - actor_commits
    activity = _thread_activity(
        run / ".egg" / "threads.sqlite", run_dir=run
    )
    turns = _turn_stats(repository, commits, activity.actor_turn_started_at)
    recorded_actor_turns = max((turn.actor_turn for turn in turns), default=0)
    actor_turns = max(
        activity.actor_turns,
        recorded_actor_turns,
        actor_commits if not turns else 0,
    )
    critic_turns = max(activity.critic_turns, recorded_actor_turns, len(turns))
    initial_actor_commits, initial_critic_commits = _initial_commit_counts(commits)
    evaluation_heads = _evaluation_heads(repository, head)
    evaluated_head, evaluation = _latest_evaluation(
        repository, head, evaluation_heads, report
    )
    backtest = evaluation.get("backtest", {}) if isinstance(evaluation, dict) else {}
    planning = evaluation.get("planning", {}) if isinstance(evaluation, dict) else {}
    models = backtest.get("models", {}) if isinstance(backtest, dict) else {}
    surviving = (
        tuple(str(item) for item in backtest.get("surviving_models", ()))
        if isinstance(backtest, dict)
        else ()
    )
    suggestions = planning.get("suggestions", ()) if isinstance(planning, dict) else ()
    return Review(
        repository=repository,
        timeline=timeline,
        report=report,
        head=head,
        actor_turns=actor_turns,
        critic_turns=critic_turns,
        actor_commits=actor_commits,
        critic_commits=critic_commits,
        evaluation_reports=len(evaluation_heads),
        evaluated_head=evaluated_head,
        model_count=len(models) if isinstance(models, dict) else 0,
        surviving_models=surviving,
        generated_plans=len(suggestions) if isinstance(suggestions, list) else 0,
        turns=turns,
        initial_head=commits[0].head if commits else None,
        initial_actor_commits=initial_actor_commits,
        initial_critic_commits=initial_critic_commits,
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
    turn = _turn_for_action(review.turns, index) if index else None
    return {
        "index": index,
        "state": state,
        "action": action,
        "transition": transition,
        "turn": turn,
    }


def render(
    review: Review,
    index: int,
    *,
    color: bool = True,
    columns: int | None = None,
    lines: int | None = None,
) -> str:
    item = frame(review, index)
    state = item["state"]
    grid = _visible_grid(state.get("grid"))
    turn = item["turn"]
    historical = turn is not None
    actor_turn = turn.actor_turn if historical else 0
    critic_turn = min(actor_turn, review.critic_turns) if historical else 0
    actor_commits = turn.actor_commits if historical else review.initial_actor_commits
    critic_commits = (
        turn.critic_commits if historical else review.initial_critic_commits
    )
    evaluation_reports = sum(
        candidate.evaluation_report
        for candidate in review.turns
        if turn is not None and candidate.actor_turn <= turn.actor_turn
    )
    evaluated_head = turn.evaluated_head if historical else None
    model_count = turn.model_count if historical else 0
    surviving_models = turn.surviving_models if historical else ()
    generated_plans = turn.generated_plans if historical else 0
    stage = turn.stage if historical else "-"
    resolution = turn.resolution if historical else "-"
    selected_head = turn.critic_head if historical else review.initial_head
    terminal_columns, terminal_lines = _terminal_viewport()
    width = max(_MIN_COLUMNS, columns or terminal_columns)
    height = lines or terminal_lines
    metadata = [
        (
            f"ARC Physics review  action {index}/{review.transitions}  "
            f"Critic HEAD {(selected_head or '-')[:12]}  latest {review.head[:12]}"
        ),
        (
            f"Through action: Actor turns: {actor_turn}/{review.actor_turns}  "
            f"Critic turns: {critic_turn}/{review.critic_turns}  "
            f"Actor commits {actor_commits}/{review.actor_commits}  "
            f"Critic commits {critic_commits}/{review.critic_commits}"
        ),
        (
            f"Real actions: {index}/{review.transitions}  Evaluation reports: "
            f"{evaluation_reports}/{review.evaluation_reports}  Report stage: {stage}  "
            f"resolution: {resolution}"
        ),
        (
            f"Evaluated Actor HEAD: {(evaluated_head or '-')[:12]}  "
            f"models: {model_count}  surviving: {surviving_models}  "
            f"planner suggestions: {generated_plans}"
        ),
        (
            f"Game state: {state.get('state', '-')}  levels: "
            f"{state.get('levels_completed', '-')}/{state.get('win_levels', '-')}  "
            f"legal: {state.get('legal_actions', ())}"
        ),
        (
            f"Action {index}/{review.transitions}: "
            + (
                json.dumps(item["action"], sort_keys=True)
                if item["action"]
                else "initial observation"
            )
        ),
        _turn_line(turn, index),
    ]
    footer = [
        "←/→ or h/l: previous/next   Home/End: first/latest   r: reload   q: quit",
        f"Critic repository: {review.repository}",
    ]
    if color:
        metadata = [_clip(line, width) for line in metadata]
        footer = [_clip(line, width) for line in footer]
        grid_lines = max(1, height - len(metadata) - len(footer) - 2)
    else:
        grid_lines = None
    output = [*metadata, ""]
    output.extend(_render_grid(grid, color=color, columns=width, lines=grid_lines))
    output.extend(["", *footer])
    return "\n".join(output)


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
                columns, lines = _terminal_viewport(stream)
                stream.write(
                    _CLEAR
                    + render(current, index, columns=columns, lines=lines)
                    + _RESET
                )
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
                    index = (
                        current.transitions
                        if at_latest
                        else min(index, current.transitions)
                    )
        finally:
            stream.write(_SHOW_CURSOR + _RESET + "\n")
            stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review a live or completed ARC Physics run from its Critic repository."
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/physics-ls20-seed0"))
    parser.add_argument(
        "--frame",
        type=int,
        help="Print one frame and exit instead of opening the arrow-key viewer.",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        help="Export the authoritative Timeline to an animated GIF and exit.",
    )
    parser.add_argument("--gif-scale", type=_positive, default=8)
    parser.add_argument("--gif-duration-ms", type=_positive, default=200)
    parser.add_argument("--gif-level-pause-ms", type=_positive, default=800)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        selected = sum(
            (
                arguments.frame is not None,
                arguments.gif is not None,
            )
        )
        if selected > 1:
            raise ValueError("--frame and --gif are mutually exclusive")
        if arguments.gif is not None:
            from .gif import export_gif

            state = load_review(arguments.run_dir)
            destination = export_gif(
                state,
                arguments.gif,
                scale=arguments.gif_scale,
                duration_ms=arguments.gif_duration_ms,
                level_pause_ms=arguments.gif_level_pause_ms,
            )
            print(f"GIF: {destination}")
        elif arguments.frame is None:
            review(arguments.run_dir)
        else:
            state = load_review(arguments.run_dir)
            print(render(state, arguments.frame, color=sys.stdout.isatty()))
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Review unavailable: {exc}", file=sys.stderr)
        return 1
    return 0


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _thread_turns(path: Path, *, run_dir: Path | None = None) -> tuple[int, int]:
    activity = _thread_activity(path, run_dir=run_dir)
    return activity.actor_turns, activity.critic_turns


def _thread_activity(
    path: Path, *, run_dir: Path | None = None
) -> _ThreadActivity:
    run_dir = run_dir or path.parent.parent
    shared = False
    if not path.is_file():
        benchmark_path = next(
            (
                parent / ".egg" / "threads.sqlite"
                for parent in path.parents
                if (parent / "benchmark.json").is_file()
            ),
            path,
        )
        path = benchmark_path
        shared = path.is_file()
    if not path.is_file():
        return _ThreadActivity()
    db = ThreadsDB(path)
    try:
        actor = critic = 0
        actor_turn_started_at = []
        for thread in list_threads(db):
            if thread.name not in {"Actor", "Critic"}:
                continue
            working_directory = _thread_working_directory(db, thread.thread_id)
            if shared and (
                working_directory is None
                or not working_directory.is_relative_to(run_dir)
            ):
                continue
            messages = load_thread_projection(db, thread.thread_id).messages
            turn_messages = [
                message
                for message in messages
                if _is_actor_critic_turn(message.payload)
            ]
            turns = len(turn_messages) or sum(
                message.payload.get("role") == "assistant" for message in messages
            )
            if thread.name == "Actor":
                actor += turns
                actor_turn_started_at.extend(
                    timestamp
                    for message in turn_messages
                    if (timestamp := _timestamp(message.created_at)) is not None
                )
            else:
                critic += turns
        return _ThreadActivity(
            actor_turns=actor,
            critic_turns=critic,
            actor_turn_started_at=tuple(sorted(actor_turn_started_at)),
        )
    finally:
        db.close()


def _thread_working_directory(db: ThreadsDB, thread_id: str) -> Path | None:
    try:
        return get_thread_working_directory(db, thread_id)
    except (OSError, TypeError, ValueError):
        return None


def _commits(repository: Path, head: str) -> tuple[_Commit, ...]:
    output = _git(
        repository,
        "log",
        "--reverse",
        "--format=%H%x00%P%x00%s%x00%ct",
        head,
    )
    commits = []
    for line in output.splitlines():
        commit, _parents, subject, timestamp = line.split("\0", 3)
        commits.append(
            _Commit(
                head=commit,
                subject=subject,
                timestamp=int(timestamp),
            )
        )
    return tuple(commits)


def _initial_commit_counts(commits: tuple[_Commit, ...]) -> tuple[int, int]:
    actor = critic = 0
    for commit in commits:
        if not commit.subject.startswith(_PHYSICS_PREFIX):
            break
        critic += 1
    return actor, critic


def _turn_stats(
    repository: Path,
    commits: tuple[_Commit, ...],
    actor_turn_started_at: tuple[float, ...],
) -> tuple[TurnStats, ...]:
    turns = []
    previous_actions = 0
    actor_head = None
    actor_commits = 0
    critic_commits = 0
    previous_report = None
    has_previous_report = False
    for commit in commits:
        if not commit.subject.startswith(_PHYSICS_PREFIX):
            actor_commits += 1
            actor_head = commit.head
            continue
        critic_commits += 1
        state = _optional_git_json(repository, commit.head, ".trusted/state.json")
        if state is None:
            continue
        report = state.get("last_report")
        if not isinstance(report, dict):
            continue
        current_actions = _nonnegative_int(state.get("actions"), previous_actions)
        if (
            has_previous_report
            and current_actions == previous_actions
            and report == previous_report
        ):
            continue
        executed = report.get("executed")
        executed_actions = len(executed) if isinstance(executed, list) else 0
        if executed_actions == 0:
            executed_actions = max(0, current_actions - previous_actions)
        action_end = current_actions
        action_start = (
            action_end - executed_actions + 1 if executed_actions else action_end + 1
        )
        actor_turn = max(
            bisect.bisect_right(actor_turn_started_at, float(commit.timestamp)),
            len(turns) + 1,
        )
        reported_head = _full_head(report.get("head"))
        evaluated_head = None
        evaluation = None
        for candidate in (reported_head, actor_head):
            if candidate is None:
                continue
            candidate_evaluation = _optional_git_json(
                repository,
                commit.head,
                f".trusted/evaluations/{candidate}.json",
            )
            if candidate_evaluation is not None:
                evaluated_head = candidate
                evaluation = candidate_evaluation
                break
        if evaluation is None:
            evaluation_heads = _evaluation_heads(repository, commit.head)
            if evaluation_heads:
                evaluated_head = evaluation_heads[-1]
                evaluation = _optional_git_json(
                    repository,
                    commit.head,
                    f".trusted/evaluations/{evaluated_head}.json",
                )
        if evaluated_head is None:
            evaluated_head = reported_head
        evaluation_report = evaluation is not None
        evaluation_data = evaluation or {}
        backtest = evaluation_data.get("backtest", {})
        planning = evaluation_data.get("planning", {})
        models = backtest.get("models", {}) if isinstance(backtest, dict) else {}
        surviving = (
            backtest.get("surviving_models", ())
            if isinstance(backtest, dict)
            else ()
        )
        suggestions = (
            planning.get("suggestions", ()) if isinstance(planning, dict) else ()
        )
        plan = report.get("plan")
        matching = report.get("matching_models", ())
        turns.append(
            TurnStats(
                actor_turn=max(1, actor_turn),
                actor_head=actor_head,
                critic_head=commit.head,
                action_start=action_start,
                action_end=action_end,
                proposed_actions=len(plan) if isinstance(plan, list) else 0,
                executed_actions=executed_actions,
                stage=str(report.get("stage") or "-"),
                resolution=str(report.get("resolution") or "-"),
                evaluation_report=evaluation_report,
                evaluated_head=evaluated_head,
                model_count=len(models) if isinstance(models, dict) else 0,
                surviving_models=_strings(surviving),
                matching_models=_strings(matching),
                generated_plans=len(suggestions) if isinstance(suggestions, list) else 0,
                actor_commits=actor_commits,
                critic_commits=critic_commits,
                prediction_matches=_prediction_matches(report, executed_actions),
                action_matching_models=_action_matching_models(
                    report, executed_actions
                ),
            )
        )
        previous_actions = max(previous_actions, current_actions)
        previous_report = report
        has_previous_report = True
    return tuple(turns)


def _turn_for_action(turns: tuple[TurnStats, ...], action: int) -> TurnStats | None:
    return next(
        (
            turn
            for turn in turns
            if turn.executed_actions and turn.action_start <= action <= turn.action_end
        ),
        None,
    )


def _turn_line(turn: TurnStats | None, action: int) -> str:
    if turn is None:
        return "Actor turn: -  selected frame has no executed action"
    within_turn = action - turn.action_start + 1
    prediction = (
        turn.prediction_matches[within_turn - 1]
        if within_turn <= len(turn.prediction_matches)
        else None
    )
    prediction_text = {True: "matched", False: "mismatched", None: "unavailable"}[
        prediction
    ]
    matching_models = (
        turn.action_matching_models[within_turn - 1]
        if within_turn <= len(turn.action_matching_models)
        else None
    )
    if matching_models is None:
        matching_models = (
            turn.matching_models
            if within_turn == turn.executed_actions
            else "unavailable"
        )
    return (
        f"Actor turn: {turn.actor_turn}  plan proposed: {turn.proposed_actions}  "
        f"executed: {turn.executed_actions} (actions {turn.action_range})  "
        f"selected: {within_turn}/{turn.executed_actions}  "
        f"prediction: {prediction_text}  matching models: {matching_models}"
    )


def _prediction_matches(
    report: Mapping[str, Any], executed_actions: int
) -> tuple[bool | None, ...]:
    plan = report.get("plan")
    executed = report.get("executed")
    if not isinstance(plan, list) or not isinstance(executed, list):
        return (None,) * executed_actions
    return tuple(
        _predicted_transition_matches(plan, executed, index)
        for index in range(executed_actions)
    )


def _predicted_transition_matches(
    plan: list[Any], executed: list[Any], index: int
) -> bool | None:
    if index >= len(plan) or index >= len(executed):
        return None
    prediction = plan[index]
    actual = executed[index]
    if not isinstance(prediction, dict) or not isinstance(actual, dict):
        return None
    return prediction.get("next_state") == actual.get("next_state")


def _action_matching_models(
    report: Mapping[str, Any], executed_actions: int
) -> tuple[tuple[str, ...] | None, ...]:
    validation = report.get("plan_validation")
    backtest = report.get("backtest")
    executed = report.get("executed")
    if not all(isinstance(value, dict) for value in (validation, backtest)):
        return (None,) * executed_actions
    predictions = validation.get("predictions")
    surviving = backtest.get("surviving_models")
    if not isinstance(predictions, list) or not isinstance(surviving, list):
        return (None,) * executed_actions
    if not isinstance(executed, list):
        return (None,) * executed_actions
    matches = []
    for index in range(executed_actions):
        if index >= len(predictions) or index >= len(executed):
            matches.append(None)
            continue
        by_model = predictions[index]
        actual = executed[index]
        if not isinstance(by_model, dict) or not isinstance(actual, dict):
            matches.append(None)
            continue
        next_state = actual.get("next_state")
        matches.append(
            tuple(
                str(model)
                for model in surviving
                if by_model.get(str(model)) == next_state
            )
        )
    return tuple(matches)


def _is_actor_critic_turn(payload: Mapping[str, Any]) -> bool:
    key = payload.get("eggopt_actor_critic_key")
    return (
        payload.get("role") == "user"
        and isinstance(key, str)
        and key.startswith(_ACTOR_CRITIC_TURN_KEY)
    )


def _timestamp(value: str) -> float | None:
    try:
        return dt.datetime.fromisoformat(value).timestamp()
    except (AttributeError, TypeError, ValueError):
        return None


def _optional_git_json(
    repository: Path, commit: str, path: str
) -> dict[str, Any] | None:
    try:
        return _git_json(repository, commit, path)
    except (RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _nonnegative_int(value: Any, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return default


def _full_head(value: Any) -> str | None:
    text = str(value or "").lower()
    if len(text) == 40 and all(
        character in "0123456789abcdef" for character in text
    ):
        return text
    return None


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _latest_evaluation(
    repository: Path,
    head: str,
    evaluated_heads: tuple[str, ...],
    report: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    preferred = str(report.get("head") or "")
    selected = preferred if preferred in evaluated_heads else None
    if selected is None and evaluated_heads:
        selected = evaluated_heads[-1]
    if selected is None:
        return None, {}
    try:
        value = _git_json(repository, head, f".trusted/evaluations/{selected}.json")
    except (RuntimeError, TypeError, ValueError):
        return selected, {}
    return selected, value


def _evaluation_heads(repository: Path, head: str) -> tuple[str, ...]:
    output = _git(repository, "ls-tree", "-r", "--name-only", head)
    prefix = ".trusted/evaluations/"
    return tuple(
        Path(path).stem
        for path in output.splitlines()
        if path.startswith(prefix) and path.endswith(".json")
    )


def _git_json(repository: Path, head: str, path: str) -> dict[str, Any]:
    value = json.loads(_git(repository, "show", f"{head}:{path}"))
    if not isinstance(value, dict):
        raise TypeError(f"Critic Git JSON must be an object: {path}")
    return value


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


def _render_grid(
    grid: tuple[tuple[int, ...], ...],
    *,
    color: bool,
    columns: int | None = None,
    lines: int | None = None,
) -> list[str]:
    if not grid:
        return ["(no grid in this frame)"]
    if not color:
        symbols = " .:-=+*#%@ABCDEF"
        return ["".join(symbols[cell % len(symbols)] for cell in row) for row in grid]

    available_columns, available_lines = _terminal_viewport()
    width = columns or available_columns
    height = lines or available_lines
    fitted = _fit_grid(grid, width, height)
    return [_half_block_row(top, bottom) for top, bottom in _row_pairs(fitted)]


def _half_block_row(top: tuple[int, ...], bottom: tuple[int, ...]) -> str:
    width = max(len(top), len(bottom))
    parts: list[str] = []
    previous: tuple[int, int] | None = None
    for column in range(width):
        colors = _cell(top, column), _cell(bottom, column)
        if colors != previous:
            parts.extend((_foreground(colors[0]), _background(colors[1])))
            previous = colors
        parts.append("▀")
    return "".join(parts) + _RESET


def _row_pairs(
    grid: tuple[tuple[int, ...], ...],
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    for index in range(0, len(grid), 2):
        top = grid[index]
        bottom = grid[index + 1] if index + 1 < len(grid) else top
        yield top, bottom


def _cell(row: tuple[int, ...], column: int) -> int:
    return row[column] if column < len(row) else 0


def _foreground(cell: int) -> str:
    red, green, blue = _PALETTE[cell % len(_PALETTE)]
    return f"\033[38;2;{red};{green};{blue}m"


def _background(cell: int) -> str:
    red, green, blue = _PALETTE[cell % len(_PALETTE)]
    return f"\033[48;2;{red};{green};{blue}m"


def _fit_grid(
    grid: tuple[tuple[int, ...], ...], columns: int, lines: int
) -> tuple[tuple[int, ...], ...]:
    source_height = len(grid)
    source_width = max(len(row) for row in grid)
    scale = min(1.0, columns / source_width, (lines * 2) / source_height)
    target_width = max(1, int(source_width * scale))
    target_height = max(1, int(source_height * scale))
    if target_width == source_width and target_height == source_height:
        return grid
    return tuple(
        tuple(
            _cell(
                grid[_sample(source_height, target_height, row)],
                _sample(source_width, target_width, column),
            )
            for column in range(target_width)
        )
        for row in range(target_height)
    )


def _sample(source: int, target: int, index: int) -> int:
    return min(source - 1, ((index * 2 + 1) * source) // (target * 2))


def _terminal_viewport(stream: TextIO = sys.stdout) -> tuple[int, int]:
    try:
        size = os.get_terminal_size(stream.fileno())
    except (AttributeError, OSError):
        size = shutil.get_terminal_size(fallback=(_DEFAULT_COLUMNS, _DEFAULT_LINES))
    # OSC 8 links and wide-glyph handling differ among terminals. Reserving a
    # small right margin prevents escape sequences from triggering auto-wrap.
    columns = size.columns - max(4, size.columns // 20)
    return max(_MIN_COLUMNS, columns), max(1, size.lines)


def _clip(line: str, columns: int) -> str:
    if len(line) <= columns:
        return line
    return line[: columns - 1] + "…"


@contextlib.contextmanager
def _raw_input(stream: TextIO) -> Iterator[None]:
    fd = stream.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # setraw() changes both input and output flags. Keeping the original
        # output flags lets the terminal continue translating LF to CRLF.
        raw = termios.tcgetattr(fd)
        raw[1] = previous[1]
        termios.tcsetattr(fd, termios.TCSANOW, raw)
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

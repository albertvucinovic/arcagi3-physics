from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import termios
import tty
from collections.abc import Iterator
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

# ARC-AGI-3's canonical palette.  Foreground and background colors let one
# terminal cell represent two vertical game pixels without distorting the
# square 64x64 image.
_PALETTE = tuple(hex_to_rgb(COLOR_MAP[index]) for index in range(16))


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
    head = _git(repository, "rev-parse", "HEAD")
    state = _git_json(repository, head, ".trusted/state.json")
    timeline = tuple(state.get("timeline") or ())
    if not timeline:
        raise ValueError("Critic canonical Timeline is empty")
    report = state.get("last_report") or {}
    if not isinstance(report, dict):
        report = {}
    heads = _commit_subjects(repository, head)
    actor_commits = sum(not subject.startswith("[physics]") for subject in heads)
    critic_commits = len(heads) - actor_commits
    actor_turns, critic_turns = _thread_turns(
        run / ".egg" / "threads.sqlite", run_dir=run
    )
    if actor_turns == 0:
        actor_turns = actor_commits
    if critic_turns == 0:
        critic_turns = critic_commits
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
    plans = planning.get("plans", ()) if isinstance(planning, dict) else ()
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
    report = review.report
    plan = report.get("committed_plan") if isinstance(report, dict) else None
    models = report.get("compatible_models", ()) if isinstance(report, dict) else ()
    terminal_columns, terminal_lines = _terminal_viewport()
    width = max(_MIN_COLUMNS, columns or terminal_columns)
    height = lines or terminal_lines
    metadata = [
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
            + (
                json.dumps(item["action"], sort_keys=True)
                if item["action"]
                else "initial observation"
            )
        ),
        (
            f"Plan: {(plan or {}).get('purpose', '-')}  models: "
            f"{(plan or {}).get('models', ())}  compatible: {models}"
        ),
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


def _thread_turns(path: Path, *, run_dir: Path | None = None) -> tuple[int, int]:
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
        return 0, 0
    db = ThreadsDB(path)
    try:
        actor = critic = 0
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
            turns = sum(
                message.payload.get("role") == "assistant" for message in messages
            )
            if thread.name == "Actor":
                actor += turns
            else:
                critic += turns
        return actor, critic
    finally:
        db.close()


def _thread_working_directory(db: ThreadsDB, thread_id: str) -> Path | None:
    try:
        return get_thread_working_directory(db, thread_id)
    except (OSError, TypeError, ValueError):
        return None


def _commit_subjects(repository: Path, head: str) -> tuple[str, ...]:
    output = _git(repository, "log", "--format=%s", head)
    return tuple(line for line in output.splitlines() if line)


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

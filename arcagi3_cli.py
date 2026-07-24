#!/usr/bin/env python3
"""Tiny offline terminal CLI for ARC-AGI-3 games.

Commands:
  python arcagi3_cli.py list
  python arcagi3_cli.py play <game_handle>

Requires:
  pip install arc-agi

Offline mode loads only local games from the ARC-AGI toolkit's
`environment_files` directory, or from --environments-dir / ENVIRONMENTS_DIR.
"""

from __future__ import annotations

import argparse
import sys
from importlib import metadata
from typing import Any, Iterable


def installed_version(distribution_name: str) -> str:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "not installed"


try:
    from arc_agi import Arcade, OperationMode
except ModuleNotFoundError as exc:  # pragma: no cover - helpful CLI error
    if exc.name != "arc_agi":
        raise
    print(
        "Missing dependency: arc-agi\n"
        "Install the ARC-AGI-3 toolkit in the Python environment used to run this script:\n\n"
        "  python -m pip install 'arc-agi>=0.9.1'\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc
except ImportError as exc:  # pragma: no cover - helpful CLI error
    print(
        "Installed `arc-agi` does not expose the ARC-AGI-3 Toolkit API "
        "(`Arcade`, `OperationMode`).\n"
        f"Python: {sys.version.split()[0]}\n"
        f"Installed arc-agi: {installed_version('arc-agi')}\n\n"
        "The ARC-AGI-3 toolkit releases require Python 3.12+. If you created the venv\n"
        "with Python 3.11, pip installs an old 0.0.x package instead. Recreate the venv\n"
        "with Python 3.12 or newer, then install:\n\n"
        "  python -m pip install --upgrade pip\n"
        "  python -m pip install 'arc-agi>=0.9.1'\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

try:
    from arcengine import GameAction, GameState
except ModuleNotFoundError as exc:  # pragma: no cover - helpful CLI error
    if exc.name != "arcengine":
        raise
    print(
        "Missing dependency: arcengine\n"
        f"Python: {sys.version.split()[0]}\n"
        f"Installed arc-agi: {installed_version('arc-agi')}\n\n"
        "This usually means the wrong/old `arc-agi` package was installed.\n"
        "Use Python 3.12+ and reinstall the ARC-AGI-3 toolkit:\n\n"
        "  python -m pip install --upgrade --force-reinstall 'arc-agi>=0.9.1'\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


SIMPLE_ACTION_ALIASES: dict[str, GameAction] = {
    "1": GameAction.ACTION1,
    "w": GameAction.ACTION1,
    "up": GameAction.ACTION1,
    "action1": GameAction.ACTION1,
    "2": GameAction.ACTION2,
    "s": GameAction.ACTION2,
    "down": GameAction.ACTION2,
    "action2": GameAction.ACTION2,
    "3": GameAction.ACTION3,
    "a": GameAction.ACTION3,
    "left": GameAction.ACTION3,
    "action3": GameAction.ACTION3,
    "4": GameAction.ACTION4,
    "d": GameAction.ACTION4,
    "right": GameAction.ACTION4,
    "action4": GameAction.ACTION4,
    "5": GameAction.ACTION5,
    "space": GameAction.ACTION5,
    "f": GameAction.ACTION5,
    "enter": GameAction.ACTION5,
    "action5": GameAction.ACTION5,
    "7": GameAction.ACTION7,
    "u": GameAction.ACTION7,
    "z": GameAction.ACTION7,
    "undo": GameAction.ACTION7,
    "action7": GameAction.ACTION7,
}

COMPLEX_ACTION_ALIASES = {"6", "click", "tap", "xy", "action6"}
RESET_ALIASES = {"r", "reset"}
QUIT_ALIASES = {"q", "quit", "exit"}
HELP_ALIASES = {"h", "help", "?"}
ACTIONS_ALIASES = {"actions", "list-actions", "available"}


def make_arcade(environments_dir: str | None = None) -> Arcade:
    """Create an ARC-AGI client that never contacts the online API."""
    kwargs: dict[str, Any] = {"operation_mode": OperationMode.OFFLINE}
    if environments_dir:
        kwargs["environments_dir"] = environments_dir
    return Arcade(**kwargs)


def action_names(actions: Iterable[GameAction]) -> list[str]:
    return [action.name for action in actions]


def print_game(game: Any) -> None:
    """Print EnvironmentInfo without depending on every optional field existing."""
    game_id = getattr(game, "game_id", "<unknown>")
    title = getattr(game, "title", "") or ""
    tags = getattr(game, "tags", None)
    suffix = f" - {title}" if title and title != game_id else ""
    tag_text = f" [{', '.join(tags)}]" if tags else ""
    print(f"{game_id}{suffix}{tag_text}")


def list_games(args: argparse.Namespace) -> int:
    arc = make_arcade(args.environments_dir)
    games = arc.get_environments()

    if not games:
        print(
            "No offline ARC-AGI-3 games were found.\n"
            "Offline mode only lists local games; installing `arc-agi` installs\n"
            "the toolkit/engine, not the game files. Put downloaded games in the\n"
            "default `environment_files` directory, or pass:\n\n"
            "  --environments-dir /path/to/environment_files\n\n"
            "Expected layout example:\n"
            "  environment_files/ls20/9607627b/metadata.json\n"
            "  environment_files/ls20/9607627b/ls20.py\n"
        )
        return 1

    for game in games:
        print_game(game)
    return 0


def print_help() -> None:
    print(
        "\nControls / commands:\n"
        "  w/up/1        ACTION1\n"
        "  s/down/2      ACTION2\n"
        "  a/left/3      ACTION3\n"
        "  d/right/4     ACTION4\n"
        "  space/f/5     ACTION5\n"
        "  click x y/6 x y  ACTION6 with coordinates in the 0-63 range\n"
        "  u/undo/7      ACTION7\n"
        "  actions       show currently available actions\n"
        "  reset         reset this game\n"
        "  help          show this help\n"
        "  quit          exit\n"
    )


def parse_player_command(command: str) -> tuple[GameAction, dict[str, int]] | str | None:
    """Return (action, data), a control string, or None for invalid input."""
    parts = command.strip().lower().split()
    if not parts:
        return None

    head = parts[0]
    if head in QUIT_ALIASES:
        return "quit"
    if head in HELP_ALIASES:
        return "help"
    if head in RESET_ALIASES:
        return "reset"
    if head in ACTIONS_ALIASES:
        return "actions"

    if head in SIMPLE_ACTION_ALIASES:
        return SIMPLE_ACTION_ALIASES[head], {}

    if head in COMPLEX_ACTION_ALIASES:
        if len(parts) != 3:
            print("ACTION6 needs coordinates: `click <x> <y>` or `6 <x> <y>`")
            return None
        try:
            x = int(parts[1])
            y = int(parts[2])
        except ValueError:
            print("Coordinates must be integers in the 0-63 range.")
            return None
        if not (0 <= x <= 63 and 0 <= y <= 63):
            print("Coordinates must be in the 0-63 range.")
            return None
        return GameAction.ACTION6, {"x": x, "y": y}

    print(f"Unknown command: {command!r}. Type `help` for controls.")
    return None


def is_valid_action(env: Any, action: GameAction) -> bool:
    return action in env.action_space


def print_state(obs: Any) -> None:
    if obs is None:
        print("No observation returned.")
        return

    state = getattr(obs, "state", None)
    levels_completed = getattr(obs, "levels_completed", None)
    state_name = getattr(state, "name", state)

    bits = []
    if state_name is not None:
        bits.append(f"state={state_name}")
    if levels_completed is not None:
        bits.append(f"levels_completed={levels_completed}")
    if bits:
        print(" | ".join(bits))


def play_game(args: argparse.Namespace) -> int:
    arc = make_arcade(args.environments_dir)
    render_mode = "terminal-fast" if args.fast else "terminal"
    env = arc.make(args.game_handle, seed=args.seed, render_mode=render_mode)

    if env is None:
        print(f"Could not create offline environment for game handle: {args.game_handle}", file=sys.stderr)
        return 1

    print(f"Playing {args.game_handle!r} in offline mode. Type `help` for controls.")
    print(f"Available actions: {action_names(env.action_space)}")

    while True:
        try:
            command = input("arc> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0

        parsed = parse_player_command(command)
        if parsed is None:
            continue

        if parsed == "quit":
            print("bye")
            return 0
        if parsed == "help":
            print_help()
            continue
        if parsed == "actions":
            print(f"Available actions: {action_names(env.action_space)}")
            continue
        if parsed == "reset":
            obs = env.reset()
            print_state(obs)
            print(f"Available actions: {action_names(env.action_space)}")
            continue

        action, data = parsed
        if not is_valid_action(env, action):
            print(f"{action.name} is not currently available.")
            print(f"Available actions: {action_names(env.action_space)}")
            continue

        obs = env.step(action, data=data)
        print_state(obs)
        print(f"Available actions: {action_names(env.action_space)}")

        state = getattr(obs, "state", None) if obs is not None else None
        if state == GameState.WIN:
            print("Game won! You can `reset` or `quit`.")
        elif state == GameState.GAME_OVER:
            print("Game over. Type `reset` to restart or `quit` to exit.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline ARC-AGI-3 terminal CLI",
    )
    parser.add_argument(
        "--environments-dir",
        help="Directory containing local ARC-AGI-3 environment files "
        "(defaults to toolkit/ENVIRONMENTS_DIR default).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List local/offline games")
    list_parser.set_defaults(func=list_games)

    play_parser = subparsers.add_parser("play", help="Play a local/offline game in the terminal")
    play_parser.add_argument("game_handle", help="Game handle/game_id, for example: ls20")
    play_parser.add_argument("--seed", type=int, default=0, help="Game seed (default: 0)")
    play_parser.add_argument(
        "--fast",
        action="store_true",
        help="Use terminal-fast rendering instead of rate-limited terminal rendering",
    )
    play_parser.set_defaults(func=play_game)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

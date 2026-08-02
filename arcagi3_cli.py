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
import contextlib
import os
import select
import shutil
import sys
import termios
import tty
from dataclasses import dataclass
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
    "w": GameAction.ACTION1,
    "up": GameAction.ACTION1,
    "action1": GameAction.ACTION1,
    "s": GameAction.ACTION2,
    "down": GameAction.ACTION2,
    "action2": GameAction.ACTION2,
    "a": GameAction.ACTION3,
    "left": GameAction.ACTION3,
    "action3": GameAction.ACTION3,
    "d": GameAction.ACTION4,
    "right": GameAction.ACTION4,
    "action4": GameAction.ACTION4,
    "space": GameAction.ACTION5,
    "f": GameAction.ACTION5,
    "enter": GameAction.ACTION5,
    "action5": GameAction.ACTION5,
    "u": GameAction.ACTION7,
    "z": GameAction.ACTION7,
    "undo": GameAction.ACTION7,
    "action7": GameAction.ACTION7,
}

COMPLEX_ACTION_ALIASES = {"click", "tap", "xy", "action6"}
RESET_ALIASES = {"r", "reset"}
QUIT_ALIASES = {"q", "quit", "exit"}
HELP_ALIASES = {"h", "help", "?"}
ACTIONS_ALIASES = {"actions", "list-actions", "available"}

# Immediate keys use a numpad layout. These are deliberately separate from
# written command aliases: a bare key acts immediately, while a command typed
# at the `arc>` prompt is parsed only after Enter.
IMMEDIATE_ACTION_KEYS: dict[str, GameAction] = {
    "8": GameAction.ACTION1,
    "2": GameAction.ACTION2,
    "4": GameAction.ACTION3,
    "6": GameAction.ACTION4,
    "5": GameAction.ACTION5,
    "7": GameAction.ACTION7,
}
ARROW_ACTIONS: dict[str, GameAction] = {
    "A": GameAction.ACTION1,
    "B": GameAction.ACTION2,
    "D": GameAction.ACTION3,
    "C": GameAction.ACTION4,
}
KEYPAD_ESCAPE_ACTIONS: dict[str, GameAction] = {
    "Ox": GameAction.ACTION1,  # keypad 8 in application keypad mode
    "Or": GameAction.ACTION2,  # keypad 2
    "Ot": GameAction.ACTION3,  # keypad 4
    "Ov": GameAction.ACTION4,  # keypad 6
    "Ou": GameAction.ACTION5,  # keypad 5
    "Ow": GameAction.ACTION7,  # keypad 7
}
KEYPAD_RESET_ESCAPE_SEQUENCES = {"Op"}  # keypad 0

ANSI_CLEAR_LINE = "\033[2K"
ANSI_CURSOR_COLUMN_ONE = "\r"
ANSI_ENABLE_MOUSE = "\033[?1000h\033[?1006h"
ANSI_DISABLE_MOUSE = "\033[?1006l\033[?1000l"
ANSI_SHOW_CURSOR = "\033[?25h"
ANSI_ENABLE_APPLICATION_KEYPAD = "\033="
ANSI_DISABLE_APPLICATION_KEYPAD = "\033>"
ANSI_RESET_SCROLL_REGION = "\033[r"

BOARD_TERMINAL_COLUMNS = 128
BOARD_BOTTOM_ROW = 66
STATUS_START_ROW = BOARD_BOTTOM_ROW + 1
MIN_MOUSE_TERMINAL_ROWS = 70

ActionRequest = tuple[GameAction, dict[str, int]]
ParsedPlayerInput = ActionRequest | str | None


@dataclass(frozen=True)
class SubmittedCommand:
    text: str


def terminal_can_map_mouse() -> bool:
    """Whether the toolkit's 128x66 terminal board fits without wrapping."""
    terminal = shutil.get_terminal_size()
    return (
        terminal.columns >= BOARD_TERMINAL_COLUMNS
        and terminal.lines >= MIN_MOUSE_TERMINAL_ROWS
    )


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
        "\nImmediate controls (no Enter):\n"
        "  Up / numpad 8       ACTION1\n"
        "  Down / numpad 2     ACTION2\n"
        "  Left / numpad 4     ACTION3\n"
        "  Right / numpad 6    ACTION4\n"
        "  Numpad 5            ACTION5\n"
        "  Mouse click         ACTION6 (needs a 128x70+ terminal)\n"
        "  Numpad 7            ACTION7 / undo\n"
        "  Numpad 0            reset\n"
        "  Esc                  quit\n"
        "\nWritten commands (type at `arc>` and press Enter):\n"
        "  up/down/left/right  ACTION1..ACTION4\n"
        "  space/f             ACTION5\n"
        "  click x y           ACTION6 with coordinates in the 0-63 range\n"
        "  undo                ACTION7\n"
        "  actions             show currently available actions\n"
        "  reset, help, quit   control the player\n"
    )


def parse_player_command(command: str) -> ParsedPlayerInput:
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
            print("ACTION6 needs coordinates: `click <x> <y>`")
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


def redraw_prompt(buffer: str) -> None:
    """Draw the editable command prompt without disturbing immediate controls."""
    print(
        f"{ANSI_CURSOR_COLUMN_ONE}{ANSI_CLEAR_LINE}arc> {buffer}",
        end="",
        flush=True,
    )


def read_escape_sequence(fd: int, timeout: float = 0.01) -> str:
    """Read bytes following ESC that are already arriving from the terminal."""
    sequence = ""
    # A short initial wait distinguishes Esc from a key sequence. Use a longer
    # grace period than subsequent bytes so arrows remain reliable over SSH.
    next_timeout = 0.1
    while len(sequence) < 31:
        ready, _, _ = select.select([fd], [], [], next_timeout)
        if not ready:
            break
        char = os.read(fd, 1).decode("utf-8", errors="ignore")
        sequence += char
        # CSI key sequences normally end in a letter or `~`; SS3 sequences
        # (arrows/application keypad) begin with O and need one more byte.
        # SGR mouse sequences begin with [< and end in M/m.
        if sequence.startswith("[<") and char in {"M", "m"}:
            break
        if sequence.startswith("[") and not sequence.startswith("[<") and (
            char.isalpha() or char == "~"
        ):
            break
        if sequence.startswith("O") and len(sequence) >= 2:
            break
        if sequence[0] not in {"[", "O"}:
            break
        next_timeout = timeout
    return sequence


def mouse_action(sequence: str) -> ActionRequest | None:
    """Translate an SGR terminal click over the rendered board to ACTION6."""
    if not sequence.startswith("[<") or not sequence.endswith("M"):
        return None

    try:
        button_text, column_text, row_text = sequence[2:-1].split(";")
        button = int(button_text)
        column = int(column_text)
        row = int(row_text)
    except (ValueError, TypeError):
        return None

    # SGR encodes modifiers in bits 2-4, motion in bit 5, and wheel events in
    # bits 6-7. Accept only an unmodified primary-button press.
    if button != 0:
        return None

    if not terminal_can_map_mouse():
        return None

    # arc_agi.rendering.render_frames_terminal starts the 64x64 board after:
    #   row 1: "Step: ..."
    #   row 2: blank
    # and renders each game pixel as two terminal columns ("██").
    x = (column - 1) // 2
    y = row - 3
    if 0 <= x <= 63 and 0 <= y <= 63:
        return GameAction.ACTION6, {"x": x, "y": y}
    return None


@contextlib.contextmanager
def interactive_terminal() -> Iterable[int]:
    """Enter character-at-a-time mode and restore the terminal on every exit."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("Interactive play requires a terminal (TTY).")

    fd = sys.stdin.fileno()
    original_attributes = termios.tcgetattr(fd)
    mouse_enabled = terminal_can_map_mouse()
    terminal_lines = shutil.get_terminal_size().lines
    try:
        tty.setcbreak(fd)
        # cbreak normally keeps ISIG enabled, which would suspend the process
        # for Ctrl+Z. Deliver control keys as bytes and handle them above.
        interactive_attributes = termios.tcgetattr(fd)
        interactive_attributes[3] &= ~termios.ISIG
        termios.tcsetattr(fd, termios.TCSANOW, interactive_attributes)
        setup = ANSI_ENABLE_APPLICATION_KEYPAD
        if mouse_enabled:
            # Keep rendering fixed in rows 1-66 while help, errors, and the
            # command prompt scroll only in the status area below the board.
            setup += f"\033[{STATUS_START_ROW};{terminal_lines}r{ANSI_ENABLE_MOUSE}"
        print(setup, end="", flush=True)
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_attributes)
        print(
            f"{ANSI_DISABLE_MOUSE}{ANSI_RESET_SCROLL_REGION}"
            f"{ANSI_DISABLE_APPLICATION_KEYPAD}{ANSI_SHOW_CURSOR}"
            f"{ANSI_CURSOR_COLUMN_ONE}{ANSI_CLEAR_LINE}",
            end="",
            flush=True,
        )


def rerender_current_observation(env: Any) -> None:
    """Re-anchor the current board after terminal interaction is configured."""
    renderer = getattr(env, "renderer", None)
    observation = env.observation_space
    if renderer is not None and observation is not None and observation.frame:
        renderer(getattr(env, "_steps", 0), observation)


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


def submit_action(env: Any, action: GameAction, data: dict[str, int]) -> Any:
    """Validate and submit one action, reporting the resulting state."""
    if not is_valid_action(env, action):
        print(f"{action.name} is not currently available.")
        print(f"Available actions: {action_names(env.action_space)}")
        return None

    obs = env.step(action, data=data)
    print_state(obs)
    print(f"Available actions: {action_names(env.action_space)}")

    state = getattr(obs, "state", None) if obs is not None else None
    if state == GameState.WIN:
        print("Game won! Press numpad 0 to reset or Esc to quit.")
    elif state == GameState.GAME_OVER:
        print("Game over. Press numpad 0 to reset or Esc to quit.")
    return obs


def execute_player_input(env: Any, parsed: ParsedPlayerInput) -> bool:
    """Execute parsed player input; return False when the player wants to quit."""
    if parsed is None:
        return True
    if parsed == "quit":
        return False
    if parsed == "help":
        print_help()
        return True
    if parsed == "actions":
        print(f"Available actions: {action_names(env.action_space)}")
        return True
    if parsed == "reset":
        obs = env.reset()
        print_state(obs)
        print(f"Available actions: {action_names(env.action_space)}")
        return True

    action, data = parsed
    submit_action(env, action, data)
    return True


def read_player_input(
    fd: int, buffer: str
) -> tuple[ParsedPlayerInput | SubmittedCommand, str, bool]:
    """Read one terminal event.

    Returns (parsed_input, command_buffer, buffer_changed). parsed_input is None
    when the event only edits the written command buffer.
    """
    char = os.read(fd, 1).decode("utf-8", errors="ignore")

    if char == "\x03":
        raise KeyboardInterrupt
    if char == "\x04" and not buffer:
        return "quit", buffer, False
    if char == "\x1a":
        return (GameAction.ACTION7, {}), buffer, False
    if char == "\x1b":
        sequence = read_escape_sequence(fd)
        if not sequence:
            return "quit", buffer, False
        if sequence[:1] in {"[", "O"} and sequence[-1:] in ARROW_ACTIONS:
            return (ARROW_ACTIONS[sequence[-1]], {}), buffer, False
        keypad_action = KEYPAD_ESCAPE_ACTIONS.get(sequence)
        if keypad_action is not None:
            return (keypad_action, {}), buffer, False
        if sequence in KEYPAD_RESET_ESCAPE_SEQUENCES:
            return "reset", buffer, False
        click = mouse_action(sequence)
        if click is not None:
            return click, buffer, False
        return None, buffer, False
    if char in {"\r", "\n"}:
        return SubmittedCommand(buffer.strip()), "", True
    if char in {"\x7f", "\b"}:
        return None, buffer[:-1], True

    # A lone numpad-style key is immediate only when no written command is in
    # progress. Once text exists, digits remain available to future commands.
    if not buffer:
        if char == "0":
            return "reset", buffer, False
        action = IMMEDIATE_ACTION_KEYS.get(char)
        if action is not None:
            return (action, {}), buffer, False

    if char.isprintable():
        return None, buffer + char, True
    return None, buffer, False


def play_game(args: argparse.Namespace) -> int:
    arc = make_arcade(args.environments_dir)
    render_mode = "terminal-fast" if args.fast else "terminal"
    env = arc.make(args.game_handle, seed=args.seed, render_mode=render_mode)

    if env is None:
        print(
            f"Could not create offline environment for game handle: {args.game_handle}",
            file=sys.stderr,
        )
        return 1

    try:
        with interactive_terminal() as fd:
            rerender_current_observation(env)
            print(
                f"Playing {args.game_handle!r} offline. "
                "Keys: arrows/8/2/4/6, 5, 7, 0; Esc quits."
            )
            if not terminal_can_map_mouse():
                print(
                    "Mouse ACTION6 needs a 128x70+ terminal; "
                    "use the written command `click x y`."
                )
            print(f"Available actions: {action_names(env.action_space)}")
            buffer = ""
            redraw_prompt(buffer)
            while True:
                event, buffer, buffer_changed = read_player_input(fd, buffer)
                if isinstance(event, SubmittedCommand):
                    # Finish the visible command line before parsing. Parser
                    # errors can then be printed cleanly beneath the prompt.
                    print()
                    parsed = (
                        parse_player_command(event.text) if event.text else None
                    )
                    if parsed is None:
                        redraw_prompt(buffer)
                        continue
                else:
                    parsed = event

                if parsed is None:
                    if buffer_changed:
                        redraw_prompt(buffer)
                    continue

                # Immediate keys leave the cursor on the `arc>` line. Written
                # commands have already completed that line above.
                if not isinstance(event, SubmittedCommand):
                    print()
                if not execute_player_input(env, parsed):
                    break
                redraw_prompt(buffer)
    except (EOFError, KeyboardInterrupt):
        pass
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("bye")
    return 0


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

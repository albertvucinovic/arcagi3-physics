"""Resolve the API-advertised ARC environment used by the shell launcher."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

OFFICIAL_BASE_URL = "https://three.arcprize.org"


def official_game_id(
    game: str,
    environments_dir: str | Path,
    *,
    api_key: str = "",
    base_url: str = OFFICIAL_BASE_URL,
    arcade_factory: Callable[..., Any] | None = None,
) -> str:
    """Return the sole API version for ``game`` after proving it exists locally."""

    if "-" in game:
        raise ValueError(
            "the default ARC game must be an unversioned base ID; set ARC_GAME "
            "to play an explicit version"
        )
    if arcade_factory is None:
        arcade_factory = _online_arcade
    arcade = arcade_factory(api_key=api_key, base_url=base_url)
    matches = sorted(
        environment.game_id
        for environment in arcade.get_environments()
        if isinstance(getattr(environment, "game_id", None), str)
        and environment.game_id.split("-", 1)[0] == game
    )
    if not matches:
        raise RuntimeError(f"official ARC API does not advertise base game {game!r}")
    if len(matches) != 1:
        raise RuntimeError(
            f"official ARC API advertises multiple versions for {game!r}: "
            + ", ".join(matches)
        )
    game_id = matches[0]
    base, version = game_id.split("-", 1)
    metadata = (
        Path(environments_dir).expanduser().resolve() / base / version / "metadata.json"
    )
    if not metadata.is_file():
        raise FileNotFoundError(
            f"current official ARC environment is not installed: {game_id}; run "
            "./leaderboard-submission/leaderboard.sh environments "
            "--environments-dir environment_files --sync"
        )
    value = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("game_id") != game_id:
        raise ValueError(
            f"local ARC metadata does not match the current API version: {metadata}"
        )
    return game_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve one base game to the exact version advertised by the ARC API."
    )
    parser.add_argument("game")
    parser.add_argument("--environments-dir", type=Path, required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default=OFFICIAL_BASE_URL)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    print(
        official_game_id(
            arguments.game,
            arguments.environments_dir,
            api_key=arguments.api_key,
            base_url=arguments.base_url,
        )
    )
    return 0


def _online_arcade(*, api_key: str = "", base_url: str = OFFICIAL_BASE_URL):
    from arc_agi import Arcade, OperationMode

    logging.getLogger("arc_agi").setLevel(logging.CRITICAL)
    logging.getLogger("arc_agi.scorecard").setLevel(logging.CRITICAL)
    return Arcade(
        arc_api_key=api_key,
        arc_base_url=base_url,
        operation_mode=OperationMode.ONLINE,
        environments_dir=tempfile.mkdtemp(prefix="arcagi3-launch-"),
        logger=logging.getLogger("arcagi3_physics.launch"),
    )


if __name__ == "__main__":
    raise SystemExit(main())

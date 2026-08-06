"""Official ARC API environment and Competition Mode scorecard workflows."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from arcagi3_physics.environment import observation, validate_action
from arcagi3_physics.review import load_review

from .scoring import official_scorecard_url, score_timeline

OFFICIAL_BASE_URL = "https://three.arcprize.org"


@dataclass(frozen=True)
class Trajectory:
    """One locally solved, versioned ARC trajectory ready for strict replay."""

    game: str
    game_id: str
    seed: int
    run_dir: Path
    timeline: tuple[Any, ...]

    @property
    def actions(self) -> int:
        return len(self.timeline) - 1


def create_competition_arcade(
    *,
    api_key: str = "",
    base_url: str = OFFICIAL_BASE_URL,
):
    """Create an API-only Competition Mode client without scanning local games."""

    from arc_agi import Arcade, OperationMode

    logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
    empty_environments = tempfile.mkdtemp(prefix="arcagi3-official-")
    arcade = Arcade(
        arc_api_key=api_key,
        arc_base_url=base_url,
        operation_mode=OperationMode.COMPETITION,
        environments_dir=empty_environments,
        logger=_quiet_logger(),
    )
    arcade._arcagi3_empty_environments_dir = empty_environments
    return arcade


def create_online_arcade(
    *,
    api_key: str = "",
    base_url: str = OFFICIAL_BASE_URL,
):
    """Create an API-only non-Competition client for read-only suite discovery."""

    from arc_agi import Arcade, OperationMode

    logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
    empty_environments = tempfile.mkdtemp(prefix="arcagi3-official-")
    arcade = Arcade(
        arc_api_key=api_key,
        arc_base_url=base_url,
        operation_mode=OperationMode.ONLINE,
        environments_dir=empty_environments,
        logger=_quiet_logger(),
    )
    arcade._arcagi3_empty_environments_dir = empty_environments
    return arcade


def open_official_scorecard(
    arcade: Any,
    *,
    source_url: str,
    tags: list[str] | None = None,
    opaque: Any = None,
) -> dict[str, str]:
    """Open the sole Competition Mode scorecard and return its public URL."""

    _require_competition(arcade)
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("official ARC scorecard source_url must be non-empty")
    scorecard_id = arcade.create_scorecard(
        source_url=source_url.strip(),
        tags=tags,
        opaque=opaque,
    )
    return {
        "scorecard_id": scorecard_id,
        "scorecard_url": official_scorecard_url(scorecard_id),
    }


def official_game_ids(arcade: Any) -> tuple[str, ...]:
    """Return exact API-advertised versioned game IDs in stable order."""

    environments = arcade.get_environments()
    games = []
    for environment in environments:
        game_id = getattr(environment, "game_id", None)
        if not isinstance(game_id, str) or not game_id:
            raise ValueError("official ARC environment has no game_id")
        games.append(game_id)
    duplicates = sorted(game for game in set(games) if games.count(game) > 1)
    if duplicates:
        raise ValueError(
            "official ARC API returned duplicate environments: " + ", ".join(duplicates)
        )
    return tuple(sorted(games))


def official_games(arcade: Any) -> tuple[str, ...]:
    """Return every API-advertised base game exactly once in stable order."""

    bases = tuple(game_id.split("-", 1)[0] for game_id in official_game_ids(arcade))
    duplicates = sorted(game for game in set(bases) if bases.count(game) > 1)
    if duplicates:
        raise ValueError(
            "official ARC API advertises multiple versions for: "
            + ", ".join(duplicates)
        )
    return bases


def resolve_current_game(
    game: str,
    environments_dir: str | Path,
    *,
    api_key: str = "",
    base_url: str = OFFICIAL_BASE_URL,
    arcade_factory=create_online_arcade,
) -> str:
    """Resolve one base game to its sole current API version installed locally."""

    if "-" in game:
        raise ValueError(
            "current-game resolution requires an unversioned base ID; set "
            "ARC_GAME to play an explicit version"
        )
    matches = tuple(
        game_id
        for game_id in official_game_ids(
            arcade_factory(api_key=api_key, base_url=base_url)
        )
        if game_id.split("-", 1)[0] == game
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


def close_official_scorecard(
    arcade: Any,
    scorecard_id: str,
    destination: str | Path,
) -> dict[str, Any]:
    """Close an official scorecard and durably save its result and public URL."""

    _require_competition(arcade)
    return _close_and_save_scorecard(arcade, scorecard_id, destination)


def _close_and_save_scorecard(
    arcade: Any,
    scorecard_id: str,
    destination: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scorecard = arcade.close_scorecard(scorecard_id)
    if scorecard is None:
        raise RuntimeError("official ARC scorecard did not close successfully")
    value = scorecard.model_dump(mode="json", exclude_none=True)
    if not isinstance(value, dict):
        raise TypeError("official ARC scorecard result must be an object")
    value["scorecard_url"] = official_scorecard_url(scorecard_id)
    if extra:
        value.update(extra)
    _write_json(Path(destination), value)
    return value


def load_trajectory(run_dir: str | Path) -> Trajectory:
    """Load and validate one authoritative Timeline from a single run directory."""

    review = load_review(run_dir)
    run = Path(run_dir).expanduser().resolve()
    configuration = _run_configuration(run)
    environments_dir = _environments_dir(run, configuration)
    game = _game(run, configuration, environments_dir)
    configured_game_id = configuration.get("game_id")
    score_game = (
        configured_game_id
        if isinstance(configured_game_id, str)
        else _infer_timeline_game_id(
            game, review.timeline, environments_dir, seed=configuration.get("seed", 0)
        )
    )
    score = score_timeline(
        score_game,
        review.timeline,
        environments_dir,
    )
    game_id = score["game_id"]
    if isinstance(configured_game_id, str) and configured_game_id != game_id:
        raise ValueError(
            f"ARC run {run_dir} records {configured_game_id}, but its selected local "
            f"metadata resolves to {game_id}"
        )
    seed = configuration.get("seed", 0)
    if type(seed) is not int:
        raise ValueError(f"ARC run seed must be an integer: {run_dir}")
    return Trajectory(
        game=game.split("-", 1)[0],
        game_id=game_id,
        seed=seed,
        run_dir=Path(run_dir).expanduser().resolve(),
        timeline=review.timeline,
    )


def collect_trajectories(run_dirs: Sequence[str | Path]) -> tuple[Trajectory, ...]:
    """Load one complete trajectory per base game from arbitrary single-run paths."""

    if not run_dirs:
        raise ValueError("at least one --run-dir is required")
    trajectories = tuple(load_trajectory(run_dir) for run_dir in run_dirs)
    by_game: dict[str, list[Path]] = {}
    for trajectory in trajectories:
        by_game.setdefault(trajectory.game, []).append(trajectory.run_dir)
    duplicates = {game: paths for game, paths in by_game.items() if len(paths) > 1}
    if duplicates:
        details = "; ".join(
            f"{game}: {', '.join(map(str, paths))}"
            for game, paths in sorted(duplicates.items())
        )
        raise ValueError(f"multiple trajectories supplied for one game: {details}")
    return tuple(sorted(trajectories, key=lambda item: item.game))


def validate_trajectory_set(
    trajectories: Sequence[Trajectory],
    official_ids: Sequence[str],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Require exact current-version coverage before any scorecard is opened."""

    expected = tuple(sorted(official_ids))
    expected_by_game = _unique_versions(expected, "official API")
    actual_by_game = _unique_versions(
        (trajectory.game_id for trajectory in trajectories), "provided trajectories"
    )
    missing = sorted(set(expected_by_game) - set(actual_by_game))
    extra = sorted(set(actual_by_game) - set(expected_by_game))
    mismatches = [
        {
            "game": game,
            "expected": expected_by_game[game],
            "actual": actual_by_game[game],
        }
        for game in sorted(set(expected_by_game) & set(actual_by_game))
        if expected_by_game[game] != actual_by_game[game]
    ]
    report = {
        "ready": not missing and not extra and not mismatches,
        "expected_environment_count": len(expected),
        "trajectory_count": len(trajectories),
        "expected_game_ids": list(expected),
        "provided": [
            {
                "game": trajectory.game,
                "game_id": trajectory.game_id,
                "actions": trajectory.actions,
                "run_dir": str(trajectory.run_dir),
            }
            for trajectory in trajectories
        ],
        "missing": missing,
        "extra": extra,
        "version_mismatches": mismatches,
    }
    if strict and not report["ready"]:
        problems = []
        if missing:
            problems.append("missing: " + ", ".join(missing))
        if extra:
            problems.append("extra: " + ", ".join(extra))
        if mismatches:
            problems.append(
                "version mismatches: "
                + ", ".join(
                    f"{item['game']} ({item['actual']} != {item['expected']})"
                    for item in mismatches
                )
            )
        raise ValueError(
            "trajectory set is not scorecard-ready; " + "; ".join(problems)
        )
    return report


def replay_trajectory(arcade: Any, scorecard_id: str, trajectory: Trajectory) -> None:
    """Submit one trajectory and reject the first server/local state divergence."""

    from arcengine import GameAction

    environment = arcade.make(
        trajectory.game_id,
        scorecard_id=scorecard_id,
        save_recording=True,
        include_frame_data=True,
        render_mode=None,
    )
    if environment is None:
        raise RuntimeError(
            f"official ARC environment did not open: {trajectory.game_id}"
        )
    current = observation(environment.observation_space)
    if current != trajectory.timeline[0]:
        raise RuntimeError(
            f"official ARC initial observation diverged for {trajectory.game_id}"
        )
    for index, transition in enumerate(trajectory.timeline[1:], 1):
        if not isinstance(transition, dict):
            raise TypeError(
                f"ARC Timeline transition {index} is not an object: {trajectory.game_id}"
            )
        action = transition.get("action")
        validate_action(current, action)
        expected = transition.get("next_state")
        if not isinstance(expected, dict):
            raise TypeError(
                f"ARC Timeline transition {index} has no next_state: {trajectory.game_id}"
            )
        frame = environment.step(
            GameAction.from_id(action["action"]),
            data=action.get("data", {}),
            reasoning={
                "source": "authoritative local Physics Timeline replay",
                "game_id": trajectory.game_id,
                "action_index": index,
            },
        )
        current = observation(frame)
        if current != expected:
            raise RuntimeError(
                f"official ARC replay diverged for {trajectory.game_id} at action {index}"
            )


def gather_scorecard(
    run_dirs: Sequence[str | Path],
    *,
    source_url: str,
    destination: str | Path,
    api_key: str = "",
    base_url: str = OFFICIAL_BASE_URL,
    tags: list[str] | None = None,
    recordings_dir: str | Path = "recordings/official-scorecard",
    confirm: bool = False,
    arcade_factory=create_competition_arcade,
) -> dict[str, Any]:
    """Preflight trajectories, then replay all of them into one hosted scorecard."""

    trajectories = collect_trajectories(run_dirs)
    arcade = arcade_factory(api_key=api_key, base_url=base_url)
    arcade.recordings_dir = str(Path(recordings_dir).expanduser().resolve())
    official_ids = official_game_ids(arcade)
    preflight = validate_trajectory_set(trajectories, official_ids, strict=confirm)
    if not confirm:
        return preflight

    opened = open_official_scorecard(
        arcade,
        source_url=source_url,
        tags=tags,
        opaque={
            "workflow": "authoritative-local-trajectory-replay",
            "game_ids": list(official_ids),
        },
    )
    scorecard_id = opened["scorecard_id"]
    progress_path = Path(destination).expanduser().resolve().with_suffix(
        ".progress.json"
    )
    progress = {
        **opened,
        "status": "replaying",
        "game_ids": list(official_ids),
        "completed_game_ids": [],
    }
    _write_json(progress_path, progress)
    try:
        for trajectory in trajectories:
            replay_trajectory(arcade, scorecard_id, trajectory)
            progress["completed_game_ids"].append(trajectory.game_id)
            _write_json(progress_path, progress)
    except BaseException:
        _save_replay_failure(destination, opened, sys.exc_info()[1])
        try:
            close_official_scorecard(arcade, scorecard_id, destination)
        except (RuntimeError, TypeError, ValueError) as close_error:
            _save_close_failure(destination, opened, close_error)
        raise
    result = close_official_scorecard(arcade, scorecard_id, destination)
    progress["status"] = "closed"
    _write_json(progress_path, progress)
    return result


def rehearse_scorecard(
    run_dirs: Sequence[str | Path],
    *,
    source_url: str,
    destination: str | Path,
    api_key: str = "",
    base_url: str = OFFICIAL_BASE_URL,
    tags: list[str] | None = None,
    recordings_dir: str | Path = "recordings/scorecard-rehearsal",
    arcade_factory=create_online_arcade,
) -> dict[str, Any]:
    """Replay the complete suite online without consuming Competition Mode."""

    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("ARC scorecard rehearsal source_url must be non-empty")
    trajectories = collect_trajectories(run_dirs)
    arcade = arcade_factory(api_key=api_key, base_url=base_url)
    arcade.recordings_dir = str(Path(recordings_dir).expanduser().resolve())
    official_ids = official_game_ids(arcade)
    validate_trajectory_set(trajectories, official_ids, strict=True)
    scorecard_id = arcade.create_scorecard(
        source_url=source_url.strip(),
        tags=[*(tags or []), "trajectory-rehearsal"],
        opaque={
            "workflow": "non-competition-trajectory-rehearsal",
            "game_ids": list(official_ids),
        },
    )
    opened = {
        "scorecard_id": scorecard_id,
        "scorecard_url": official_scorecard_url(scorecard_id),
    }
    progress_path = Path(destination).expanduser().resolve().with_suffix(
        ".progress.json"
    )
    progress = {
        **opened,
        "status": "rehearsing",
        "competition_mode": False,
        "completed_game_ids": [],
    }
    _write_json(progress_path, progress)
    try:
        for trajectory in trajectories:
            replay_trajectory(arcade, scorecard_id, trajectory)
            progress["completed_game_ids"].append(trajectory.game_id)
            _write_json(progress_path, progress)
    except BaseException:
        _save_replay_failure(destination, opened, sys.exc_info()[1])
        try:
            _close_and_save_scorecard(
                arcade,
                scorecard_id,
                destination,
                extra={"competition_mode": False, "rehearsal": True},
            )
        except (RuntimeError, TypeError, ValueError) as close_error:
            _save_close_failure(destination, opened, close_error)
        raise
    result = _close_and_save_scorecard(
        arcade,
        scorecard_id,
        destination,
        extra={"competition_mode": False, "rehearsal": True},
    )
    progress["status"] = "closed"
    _write_json(progress_path, progress)
    return result


def sync_official_environments(
    environments_dir: str | Path,
    *,
    api_key: str = "",
    base_url: str = OFFICIAL_BASE_URL,
    dry_run: bool = True,
    arcade_factory=create_online_arcade,
) -> dict[str, Any]:
    """Download the current official environment suite without deleting old versions."""

    destination = Path(environments_dir).expanduser().resolve()
    discovery = arcade_factory(api_key=api_key, base_url=base_url)
    expected = official_game_ids(discovery)
    current = _local_game_ids(destination)
    missing = tuple(game_id for game_id in expected if game_id not in current)
    existing = tuple(game_id for game_id in expected if game_id in current)
    expected_bases = {item.split("-", 1)[0] for item in expected}
    stale = tuple(
        game_id
        for game_id in current
        if game_id.split("-", 1)[0] in {item.split("-", 1)[0] for item in expected}
        and game_id not in expected
    )
    extra = tuple(
        game_id for game_id in current if game_id.split("-", 1)[0] not in expected_bases
    )
    report: dict[str, Any] = {
        "dry_run": dry_run,
        "official_environment_count": len(expected),
        "official_game_ids": list(expected),
        "local_game_ids": list(current),
        "missing_current_versions": list(missing),
        "refreshed_current_versions": [],
        "retained_old_versions": list(stale),
        "extra_local_environments": list(extra),
        "downloaded": [],
    }
    if dry_run:
        return report
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arcagi3-sync-") as staging:
        from arc_agi import Arcade, OperationMode

        downloader = Arcade(
            arc_api_key=api_key or discovery.arc_api_key,
            arc_base_url=base_url,
            operation_mode=OperationMode.NORMAL,
            environments_dir=staging,
            logger=_quiet_logger(),
        )
        for game_id in expected:
            environment = downloader.make(game_id, seed=0, render_mode=None)
            if environment is None:
                raise RuntimeError(
                    f"failed to download official ARC environment: {game_id}"
                )
            downloaded = (
                destination / game_id.split("-", 1)[0] / game_id.split("-", 1)[1]
            )
            source = Path(environment.environment_info.local_dir)
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            if downloaded.exists():
                shutil.rmtree(downloaded)
            shutil.move(str(source), str(downloaded))
            metadata = downloaded / "metadata.json"
            value = json.loads(metadata.read_text(encoding="utf-8"))
            value["local_dir"] = str(downloaded)
            _write_json(metadata, value)
            target = (
                report["refreshed_current_versions"]
                if game_id in existing
                else report["downloaded"]
            )
            target.append(game_id)
    report["local_game_ids"] = list(_local_game_ids(destination))
    return report


def validate_local_trajectories(
    run_dirs: Sequence[str | Path], environments_dir: str | Path
) -> dict[str, Any]:
    """Replay every Timeline locally against its exact refreshed game version."""

    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
    trajectories = collect_trajectories(run_dirs)
    results = []
    for trajectory in trajectories:
        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(Path(environments_dir).expanduser().resolve()),
            logger=_quiet_logger(),
        )
        environment = arcade.make(
            trajectory.game_id, seed=trajectory.seed, render_mode=None
        )
        if environment is None:
            raise RuntimeError(
                f"local ARC environment did not open: {trajectory.game_id}"
            )
        current = observation(environment.observation_space)
        if current != trajectory.timeline[0]:
            raise RuntimeError(
                f"local ARC initial observation diverged for {trajectory.game_id}"
            )
        for index, transition in enumerate(trajectory.timeline[1:], 1):
            action = transition["action"]
            validate_action(current, action)
            current = observation(
                environment.step(
                    GameAction.from_id(action["action"]),
                    data=action.get("data", {}),
                )
            )
            if current != transition["next_state"]:
                raise RuntimeError(
                    f"local ARC replay diverged for {trajectory.game_id} "
                    f"at action {index}"
                )
        results.append(
            {
                "game_id": trajectory.game_id,
                "actions": trajectory.actions,
                "run_dir": str(trajectory.run_dir),
            }
        )
    return {"valid": True, "trajectories": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover/sync official ARC environments or gather a hosted scorecard."
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default=OFFICIAL_BASE_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    environments = subparsers.add_parser(
        "environments", help="List exact current official environment versions."
    )
    environments.add_argument(
        "--environments-dir", type=Path, default=Path("environment_files")
    )
    environments.add_argument(
        "--sync",
        action="store_true",
        help="Download missing current versions; retain old versions.",
    )

    current_game = subparsers.add_parser(
        "current-game",
        help="Resolve one base game to the exact current API version installed locally.",
    )
    current_game.add_argument("game")
    current_game.add_argument(
        "--environments-dir", type=Path, default=Path("environment_files")
    )

    score = subparsers.add_parser(
        "score",
        help="Compute local official-toolkit RHAE reports from run directories.",
    )
    score.add_argument("--run-dir", type=Path, action="append", required=True)
    score.add_argument("--output", type=Path)
    score.add_argument(
        "--current-metadata",
        action="store_true",
        help="Use current API baselines, rejecting version mismatches.",
    )

    validate = subparsers.add_parser(
        "validate", help="Strictly replay Timelines against exact local versions."
    )
    validate.add_argument("--run-dir", type=Path, action="append", required=True)
    validate.add_argument(
        "--environments-dir", type=Path, default=Path("environment_files")
    )

    rehearse = subparsers.add_parser(
        "rehearse",
        help="Replay the complete suite online without Competition Mode.",
    )
    rehearse.add_argument("--run-dir", type=Path, action="append", required=True)
    rehearse.add_argument("--source-url", required=True)
    rehearse.add_argument("--tag", action="append")
    rehearse.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("recordings/scorecard-rehearsal"),
    )
    rehearse.add_argument(
        "--output", type=Path, default=Path("scorecard-rehearsal.json")
    )

    gather = subparsers.add_parser(
        "gather",
        help="Validate or replay completed single-run Timelines into one scorecard.",
    )
    gather.add_argument("--run-dir", type=Path, action="append", required=True)
    gather.add_argument("--source-url", required=True)
    gather.add_argument("--tag", action="append")
    gather.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("recordings/official-scorecard"),
    )
    gather.add_argument("--output", type=Path, default=Path("official-scorecard.json"))
    gather.add_argument(
        "--confirm-replay",
        action="store_true",
        help="Open the one-shot scorecard and submit actions; omit for safe preflight only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "environments":
            value = sync_official_environments(
                arguments.environments_dir,
                api_key=arguments.api_key,
                base_url=arguments.base_url,
                dry_run=not arguments.sync,
            )
        elif arguments.command == "current-game":
            print(
                resolve_current_game(
                    arguments.game,
                    arguments.environments_dir,
                    api_key=arguments.api_key,
                    base_url=arguments.base_url,
                )
            )
            return 0
        elif arguments.command == "score":
            value = score_run_directories(
                arguments.run_dir,
                output=arguments.output,
                current_metadata=arguments.current_metadata,
                api_key=arguments.api_key,
                base_url=arguments.base_url,
            )
        elif arguments.command == "validate":
            value = validate_local_trajectories(
                arguments.run_dir, arguments.environments_dir
            )
        elif arguments.command == "rehearse":
            value = rehearse_scorecard(
                arguments.run_dir,
                source_url=arguments.source_url,
                destination=arguments.output,
                api_key=arguments.api_key,
                base_url=arguments.base_url,
                tags=arguments.tag,
                recordings_dir=arguments.recordings_dir,
            )
        else:
            value = gather_scorecard(
                arguments.run_dir,
                source_url=arguments.source_url,
                destination=arguments.output,
                api_key=arguments.api_key,
                base_url=arguments.base_url,
                tags=arguments.tag,
                recordings_dir=arguments.recordings_dir,
                confirm=arguments.confirm_replay,
            )
        print(json.dumps(value, indent=2, sort_keys=True))
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Official ARC workflow unavailable: {exc}", file=sys.stderr)
        return 1
    return 0


def score_run_directories(
    run_dirs: Sequence[str | Path],
    *,
    output: str | Path | None = None,
    current_metadata: bool = False,
    api_key: str = "",
    base_url: str = OFFICIAL_BASE_URL,
) -> dict[str, Any]:
    """Score independent single runs and optionally save one aggregate report."""

    trajectories = collect_trajectories(run_dirs)
    scores = []
    for trajectory in trajectories:
        configuration = _run_configuration(trajectory.run_dir)
        environments_dir = _environments_dir(trajectory.run_dir, configuration)
        metadata_source = metadata = None
        if current_metadata:
            from .scoring import fetch_official_metadata

            metadata_source, metadata = fetch_official_metadata(
                trajectory.game, api_key=api_key, base_url=base_url
            )
            if metadata["game_id"] != trajectory.game_id:
                raise ValueError(
                    "current official ARC game version differs from the played local "
                    f"version: {metadata['game_id']} != {trajectory.game_id}"
                )
        score = score_timeline(
            trajectory.game_id,
            trajectory.timeline,
            environments_dir,
            metadata_override=metadata,
            metadata_source=metadata_source,
        )
        score["run_dir"] = str(trajectory.run_dir)
        _write_json(trajectory.run_dir / "score.json", score)
        scores.append(score)
    aggregate = {
        "metric": "RHAE",
        "environment_count": len(scores),
        "score": sum(item["score"] for item in scores) / len(scores),
        "environments": scores,
    }
    if output is not None:
        _write_json(Path(output), aggregate)
    return aggregate


def _run_configuration(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _infer_timeline_game_id(
    game: str, timeline: Sequence[Any], environments_dir: Path, *, seed: Any
) -> str:
    """Find the one local version whose seed-0 initial observation matches an old run."""

    from arc_agi import Arcade, OperationMode

    logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
    initial = timeline[0] if timeline else None
    if not isinstance(initial, dict):
        raise TypeError("ARC Timeline has no initial observation")
    if type(seed) is not int:
        raise ValueError("ARC run seed must be an integer")
    base = game.split("-", 1)[0]
    candidates = []
    for metadata in sorted((environments_dir / base).glob("*/metadata.json")):
        value = json.loads(metadata.read_text(encoding="utf-8"))
        game_id = value.get("game_id") if isinstance(value, dict) else None
        if not isinstance(game_id, str):
            continue
        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(environments_dir),
            logger=_quiet_logger(),
        )
        environment = arcade.make(game_id, seed=seed, render_mode=None)
        if environment is not None and observation(environment.observation_space) == initial:
            candidates.append(game_id)
    if len(candidates) > 1:
        matches = [
            game_id
            for game_id in candidates
            if _timeline_matches_local_version(
                game_id, timeline, environments_dir, seed=seed
            )
        ]
        if len(matches) == 1:
            return matches[0]
    if len(candidates) != 1:
        raise ValueError(
            f"old ARC run has no recorded game_id and its initial observation "
            f"matches {len(candidates)} local versions of {base}; add a trusted "
            "run.json game_id or rerun on an explicit version"
        )
    return candidates[0]


def _timeline_matches_local_version(
    game_id: str,
    timeline: Sequence[Any],
    environments_dir: Path,
    *,
    seed: int,
) -> bool:
    """Return whether a complete old Timeline exactly replays on one version."""

    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(environments_dir),
        logger=_quiet_logger(),
    )
    environment = arcade.make(game_id, seed=seed, render_mode=None)
    if environment is None:
        return False
    current = observation(environment.observation_space)
    if current != timeline[0]:
        return False
    for transition in timeline[1:]:
        if not isinstance(transition, dict):
            return False
        action = transition.get("action")
        try:
            validate_action(current, action)
            current = observation(
                environment.step(
                    GameAction.from_id(action["action"]),
                    data=action.get("data", {}),
                )
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return False
        if current != transition.get("next_state"):
            return False
    return True


def _environments_dir(run_dir: Path, configuration: dict[str, Any]) -> Path:
    configured = configuration.get("environments_dir")
    if isinstance(configured, str) and configured:
        return Path(configured).expanduser().resolve()
    candidates = (run_dir / "benchmark.json",) + tuple(
        parent / "benchmark.json" for parent in run_dir.parents
    )
    for path in candidates:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            configured = value.get("environments_dir") if isinstance(value, dict) else None
            if isinstance(configured, str) and configured:
                return Path(configured).expanduser().resolve()
    default = Path("environment_files").resolve()
    if default.is_dir():
        return default
    raise FileNotFoundError(
        f"ARC environments directory could not be inferred for {run_dir}"
    )


def _game(
    run_dir: Path, configuration: dict[str, Any], environments_dir: Path
) -> str:
    for filename in ("result.json", "status.json"):
        path = run_dir / filename
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            game = value.get("game") if isinstance(value, dict) else None
            if isinstance(game, str) and game:
                return game
    game = configuration.get("game")
    if isinstance(game, str) and game:
        return game
    if run_dir.name.startswith("Physics "):
        return run_dir.name.removeprefix("Physics ")
    candidates = set()
    for path in environments_dir.rglob("metadata.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        game_id = value.get("game_id") if isinstance(value, dict) else None
        if isinstance(game_id, str):
            game = game_id.split("-", 1)[0]
            if game in run_dir.name:
                candidates.add(game)
    if len(candidates) == 1:
        return candidates.pop()
    raise ValueError(f"ARC game could not be inferred from run: {run_dir}")


def _unique_versions(game_ids: Sequence[str], source: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for game_id in game_ids:
        if not isinstance(game_id, str) or "-" not in game_id:
            raise ValueError(f"{source} contains an unversioned game ID: {game_id!r}")
        game = game_id.split("-", 1)[0]
        if game in versions:
            raise ValueError(f"{source} contains multiple versions of {game}")
        versions[game] = game_id
    return versions


def _local_game_ids(environments_dir: Path) -> tuple[str, ...]:
    if not environments_dir.is_dir():
        return ()
    values = []
    for metadata in environments_dir.rglob("metadata.json"):
        value = json.loads(metadata.read_text(encoding="utf-8"))
        game_id = value.get("game_id") if isinstance(value, dict) else None
        if not isinstance(game_id, str) or "-" not in game_id:
            raise ValueError(f"ARC metadata has no versioned game_id: {metadata}")
        values.append(game_id)
    return tuple(sorted(values))


def _save_replay_failure(
    destination: str | Path, opened: dict[str, str], error: Any
) -> None:
    path = Path(destination).expanduser().resolve().with_suffix(".failure.json")
    _write_json(path, {**opened, "error": f"{type(error).__name__}: {error}"})


def _save_close_failure(
    destination: str | Path, opened: dict[str, str], error: Any
) -> None:
    path = Path(destination).expanduser().resolve().with_suffix(".close-failure.json")
    _write_json(path, {**opened, "error": f"{type(error).__name__}: {error}"})


def _quiet_logger() -> logging.Logger:
    logger = logging.getLogger("arcagi3_physics.official.arcade")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _write_json(path: Path, value: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as temporary:
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _require_competition(arcade: Any) -> None:
    from arc_agi import OperationMode

    if getattr(arcade, "operation_mode", None) != OperationMode.COMPETITION:
        raise ValueError("official ARC scorecards require OperationMode.COMPETITION")


if __name__ == "__main__":
    raise SystemExit(main())

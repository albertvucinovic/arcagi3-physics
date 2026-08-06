"""Reporting-only ARC-AGI-3 Relative Human Action Efficiency (RHAE)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from arc_agi import EnvironmentScoreCalculator


def score_timeline(
    game: str,
    timeline: Sequence[Any],
    environments_dir: str | Path,
    *,
    metadata_override: dict[str, Any] | None = None,
    metadata_source: str | None = None,
) -> dict[str, Any]:
    """Score one public ARC Timeline with the official toolkit calculator.

    The human baselines come from the downloaded environment's ``metadata.json``.
    This function is for reporting only and is not used by PhysicsStrategy or the
    Actor. It deliberately delegates the formula and weighting to the installed
    official ``arc_agi.EnvironmentScoreCalculator``.
    """

    base_game = game.split("-", 1)[0]
    states = _states(timeline)
    if metadata_override is None:
        metadata_path, metadata = _metadata(game, environments_dir)
        source = str(metadata_path)
    else:
        metadata_path = None
        metadata = _validate_metadata(
            base_game, metadata_override, metadata_source or "override"
        )
        source = metadata_source or "override"
    baselines = _baselines(metadata, source)
    completed = _level_completion_actions(states, len(baselines))
    initial_levels = _level_count(states[0])
    final = states[-1]
    total_actions = len(states) - 1
    levels_completed = _level_count(final)
    level_delta = levels_completed - initial_levels
    if level_delta < 0:
        raise ValueError("ARC Timeline final levels_completed precedes its initial value")
    expected_completed = min(level_delta, len(baselines))
    if len(completed) != expected_completed:
        raise ValueError(
            "ARC Timeline level progression is inconsistent: "
            f"initial levels_completed={initial_levels}, "
            f"final levels_completed={levels_completed}, observed completions={len(completed)}"
        )
    if initial_levels:
        raise ValueError(
            "official RHAE requires a complete Timeline starting at levels_completed=0"
        )

    calculator = EnvironmentScoreCalculator(id=str(metadata["game_id"]), resets=0)
    previous_actions = 0
    levels = []
    for offset, baseline in enumerate(baselines):
        level = offset + 1
        completion_offset = level - 1
        is_complete = completion_offset < len(completed)
        if is_complete:
            cumulative_actions = completed[completion_offset]
        elif level == len(completed) + 1:
            cumulative_actions = total_actions
        else:
            cumulative_actions = previous_actions
        actions = cumulative_actions - previous_actions
        if actions < 0:
            raise ValueError("ARC Timeline level action counts are not monotonic")
        calculator.add_level(
            level_index=level,
            completed=is_complete,
            actions_taken=actions,
            baseline_actions=baseline,
        )
        level_score = calculator.level_scores[-1]
        levels.append(
            {
                "level": level,
                "completed": is_complete,
                "actions": actions,
                "human_baseline_actions": baseline,
                "score": level_score,
            }
        )
        previous_actions = cumulative_actions

    official = calculator.to_score()
    return {
        "metric": "RHAE",
        "game": base_game,
        "game_id": str(metadata["game_id"]),
        "score": official.score,
        "complete_history": True,
        "levels_completed": levels_completed,
        "initial_levels_completed": initial_levels,
        "total_levels": len(baselines),
        "actions": total_actions,
        "human_baseline_actions": baselines,
        "levels": levels,
        "metadata_source": source,
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "methodology": "https://docs.arcprize.org/methodology",
    }


def score_reviews(
    reviews: Sequence[tuple[str, Sequence[Any]]],
    environments_dir: str | Path,
    *,
    expected_games: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score several game Timelines and average their official per-game scores."""

    environments = []
    unavailable = []
    for game, timeline in reviews:
        try:
            environments.append(score_timeline(game, timeline, environments_dir))
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            unavailable.append({"game": game, "reason": str(exc)})
    expected = tuple(dict.fromkeys(expected_games or (game for game, _ in reviews)))
    scores = {environment["game"]: environment["score"] for environment in environments}
    score = sum(scores.get(game, 0.0) for game in expected) / len(expected) if expected else 0.0
    return {
        "metric": "RHAE",
        "score": score,
        "environment_count": len(environments),
        "scored_environment_count": len(environments),
        "expected_environment_count": len(expected),
        "environments": environments,
        "unavailable": unavailable,
        "methodology": "https://docs.arcprize.org/methodology",
    }


def _states(timeline: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    if not timeline:
        raise ValueError("ARC Timeline is empty")
    states = []
    for index, item in enumerate(timeline):
        state = item if index == 0 else item.get("next_state") if isinstance(item, dict) else None
        if not isinstance(state, dict):
            raise TypeError(f"ARC Timeline state {index} must be an object")
        states.append(state)
    return tuple(states)


def fetch_official_metadata(
    game: str,
    *,
    api_key: str = "",
    base_url: str = "https://three.arcprize.org",
) -> tuple[str, dict[str, Any]]:
    """Fetch current ARC game metadata and human baselines from the official API."""

    import requests

    base_url = base_url.rstrip("/")
    if not api_key:
        response = requests.get(f"{base_url}/api/games/anonkey", timeout=10)
        response.raise_for_status()
        anonymous = response.json()
        api_key = anonymous.get("api_key") if isinstance(anonymous, dict) else None
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("official ARC API returned no usable API key")
    url = f"{base_url}/api/games/{game}"
    response = requests.get(
        url,
        headers={"X-API-Key": api_key, "Accept": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    value = response.json()
    return url, _validate_metadata(game, value, url)


def _metadata(game: str, environments_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(environments_dir).expanduser().resolve()
    base, separator, _ = game.partition("-")
    matches = sorted((root / base).glob("*/metadata.json"))
    if not matches:
        matches = sorted(
            path
            for path in root.rglob("metadata.json")
            if _metadata_game_id(path).split("-", 1)[0] == base
        )
    if separator:
        matches = [path for path in matches if _metadata_game_id(path) == game]
    if not matches:
        raise FileNotFoundError(f"ARC metadata not found for {game!r} under {root}")
    if len(matches) > 1:
        configured = [
            path
            for path in matches
            if str(json.loads(path.read_text()).get("date_downloaded") or "")
        ]
        if configured:
            matches = [
                max(
                    configured,
                    key=lambda path: str(
                        json.loads(path.read_text()).get("date_downloaded") or ""
                    ),
                )
            ]
        else:
            raise ValueError(
                f"multiple ARC metadata versions found for {game!r} under {root}; "
                "use the exact versioned game ID"
            )
    path = matches[0]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"ARC metadata must be an object: {path}")
    return path, _validate_metadata(base, value, str(path))


def _validate_metadata(
    game: str, value: Any, source: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"ARC metadata must be an object: {source}")
    game_id = value.get("game_id")
    if not isinstance(game_id, str) or game_id.split("-", 1)[0] != game:
        raise ValueError(f"ARC metadata game_id does not match {game!r}: {source}")
    return value


def _metadata_game_id(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(value.get("game_id", "")) if isinstance(value, dict) else ""


def _baselines(metadata: dict[str, Any], source: str | Path) -> list[int]:
    value = metadata.get("baseline_actions")
    if not isinstance(value, list) or not value:
        raise ValueError(f"ARC human baseline actions are unavailable: {source}")
    if any(type(item) is not int or item < 1 for item in value):
        raise ValueError(f"ARC human baseline actions are invalid: {source}")
    return list(value)


def _level_count(state: dict[str, Any]) -> int:
    value = state.get("levels_completed")
    if type(value) is not int or value < 0:
        raise ValueError("ARC state levels_completed must be a non-negative integer")
    return value


def _level_completion_actions(
    states: Sequence[dict[str, Any]], total_levels: int
) -> tuple[int, ...]:
    previous = _level_count(states[0])
    completed: list[int] = []
    for action_index, state in enumerate(states[1:], 1):
        current = _level_count(state)
        if current < previous or current - previous > 1:
            raise ValueError("ARC Timeline levels_completed must increase one level at a time")
        if current > total_levels:
            raise ValueError("ARC Timeline completes more levels than metadata defines")
        if current > previous:
            completed.append(action_index)
        previous = current
    return tuple(completed)


def official_scorecard_url(scorecard_id: str) -> str:
    """Return the hosted scorecard URL required by the Community Leaderboard."""

    if not isinstance(scorecard_id, str) or not scorecard_id.strip():
        raise ValueError("official ARC scorecard ID must be a non-empty string")
    return f"https://arcprize.org/scorecards/{scorecard_id.strip()}"

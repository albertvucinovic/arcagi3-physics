from __future__ import annotations

from pathlib import Path

import pytest


def test_official_rhae_scoring_uses_downloaded_human_baselines(tmp_path):
    import json

    from leaderboard_submission.scoring import score_timeline

    metadata = tmp_path / "environment_files" / "toy" / "version" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"game_id": "toy-version", "baseline_actions": [10, 20, 30]})
    )
    initial = {
        "grid": [[[0]]],
        "legal_actions": [1],
        "state": "NOT_FINISHED",
        "levels_completed": 0,
        "win_levels": 3,
    }

    def transition(state, levels):
        next_state = {**state, "levels_completed": levels}
        return {"state": state, "action": {"action": 1}, "next_state": next_state}

    timeline = [initial]
    state = initial
    for action_index in range(1, 7):
        levels = 1 if action_index >= 2 else 0
        levels = 2 if action_index >= 6 else levels
        item = transition(state, levels)
        timeline.append(item)
        state = item["next_state"]

    score = score_timeline("toy", timeline, tmp_path / "environment_files")

    assert score["human_baseline_actions"] == [10, 20, 30]
    assert score["levels_completed"] == 2
    assert score["actions"] == 6
    assert [level["actions"] for level in score["levels"]] == [2, 4, 0]
    assert [level["completed"] for level in score["levels"]] == [True, True, False]
    assert [level["score"] for level in score["levels"]] == [115.0, 115.0, 0.0]
    # Official game-level completion cap: weights 1+2 completed out of 1+2+3.
    assert score["score"] == 50.0


def test_official_rhae_assigns_remaining_actions_only_to_attempted_level(tmp_path):
    import json

    from leaderboard_submission.scoring import score_timeline

    metadata = tmp_path / "toy" / "version" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"game_id": "toy-v", "baseline_actions": [10, 20, 30]})
    )
    initial = {"grid": [[[0]]], "levels_completed": 0}
    final = {"grid": [[[1]]], "levels_completed": 0}
    score = score_timeline(
        "toy",
        [initial, {"state": initial, "action": {"action": 1}, "next_state": final}],
        tmp_path,
    )

    assert [level["actions"] for level in score["levels"]] == [1, 0, 0]


def test_official_rhae_scoring_rejects_skipped_level_progress(tmp_path):
    import json

    from leaderboard_submission.scoring import score_timeline

    metadata = tmp_path / "toy" / "version" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"game_id": "toy-v", "baseline_actions": [1, 1]}))
    initial = {"grid": [[[0]]], "levels_completed": 0}
    timeline = [
        initial,
        {
            "state": initial,
            "action": {"action": 1},
            "next_state": {"grid": [[[1]]], "levels_completed": 2},
        },
    ]

    with pytest.raises(ValueError, match="one level at a time"):
        score_timeline("toy", timeline, tmp_path)


def test_official_rhae_rejects_incomplete_timeline_history(tmp_path):
    import json

    from leaderboard_submission.scoring import score_timeline

    metadata = tmp_path / "toy" / "version" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"game_id": "toy-v", "baseline_actions": [1, 1]}))

    with pytest.raises(ValueError, match="complete Timeline"):
        score_timeline(
            "toy",
            [{"grid": [[[0]]], "levels_completed": 1}],
            tmp_path,
        )


def test_official_scorecard_url_matches_community_leaderboard_contract():
    from leaderboard_submission.scoring import official_scorecard_url

    assert official_scorecard_url("card-123") == (
        "https://arcprize.org/scorecards/card-123"
    )
    with pytest.raises(ValueError, match="non-empty"):
        official_scorecard_url("")


def test_timeline_score_can_use_explicit_current_official_metadata(tmp_path):
    from leaderboard_submission.scoring import score_timeline

    timeline = ({"grid": [[[0]]], "levels_completed": 0},)
    score = score_timeline(
        "toy",
        timeline,
        tmp_path,
        metadata_override={"game_id": "toy-current", "baseline_actions": [11]},
        metadata_source="https://three.arcprize.org/api/games/toy",
    )

    assert score["game_id"] == "toy-current"
    assert score["human_baseline_actions"] == [11]
    assert score["metadata_path"] is None
    assert score["metadata_source"] == "https://three.arcprize.org/api/games/toy"


def test_fetch_official_metadata_uses_official_api(monkeypatch):
    from leaderboard_submission.scoring import fetch_official_metadata

    calls = []

    class Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            return None

        def json(self):
            return self.value

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/anonkey"):
            return Response({"api_key": "anonymous"})
        return Response({"game_id": "toy-current", "baseline_actions": [7]})

    monkeypatch.setattr("requests.get", get)

    source, metadata = fetch_official_metadata(
        "toy", base_url="https://three.arcprize.org/"
    )

    assert source == "https://three.arcprize.org/api/games/toy"
    assert metadata["baseline_actions"] == [7]
    assert calls[1][1]["headers"]["X-API-Key"] == "anonymous"


def test_multi_game_rhae_counts_missing_expected_games_as_zero(monkeypatch):
    from leaderboard_submission import scoring

    monkeypatch.setattr(
        scoring,
        "score_timeline",
        lambda game, timeline, environments_dir: {
            "game": game,
            "score": 100.0,
            "complete_history": True,
        },
    )

    score = scoring.score_reviews(
        [("played", ({},))],
        ".",
        expected_games=("played", "missing"),
    )

    assert score["score"] == 50.0
    assert score["environment_count"] == 1
    assert score["expected_environment_count"] == 2


def test_official_scorecard_close_is_durable_and_preserves_submission_url(tmp_path):
    import json

    from arc_agi import OperationMode
    from leaderboard_submission.workflow import close_official_scorecard

    class Scorecard:
        def model_dump(self, **kwargs):
            assert kwargs == {"mode": "json", "exclude_none": True}
            return {"score": 42.0, "card_id": "card-123"}

    class Arcade:
        operation_mode = OperationMode.COMPETITION

        def close_scorecard(self, scorecard_id):
            assert scorecard_id == "card-123"
            return Scorecard()

    destination = tmp_path / "official-scorecard.json"
    result = close_official_scorecard(Arcade(), "card-123", destination)

    assert result["scorecard_url"] == "https://arcprize.org/scorecards/card-123"
    assert json.loads(destination.read_text()) == result


def test_competition_arcade_uses_official_competition_mode(monkeypatch):
    from arc_agi import OperationMode
    from leaderboard_submission.workflow import create_competition_arcade

    captured = {}

    class Arcade:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("arc_agi.Arcade", Arcade)

    create_competition_arcade(api_key="key", base_url="https://example.test")

    assert captured["arc_api_key"] == "key"
    assert captured["arc_base_url"] == "https://example.test"
    assert captured["operation_mode"] == OperationMode.COMPETITION
    assert Path(captured["environments_dir"]).is_dir()


def test_official_scorecard_open_returns_submission_url():
    from arc_agi import OperationMode
    from leaderboard_submission.workflow import open_official_scorecard

    class Arcade:
        operation_mode = OperationMode.COMPETITION

        def create_scorecard(self, **kwargs):
            assert kwargs == {
                "source_url": "https://github.com/example/project",
                "tags": ["physics"],
                "opaque": {"commit": "abc"},
            }
            return "card-456"

    result = open_official_scorecard(
        Arcade(),
        source_url="https://github.com/example/project",
        tags=["physics"],
        opaque={"commit": "abc"},
    )

    assert result == {
        "scorecard_id": "card-456",
        "scorecard_url": "https://arcprize.org/scorecards/card-456",
    }


def test_timeline_rhae_matches_official_scorecard_calculation(tmp_path):
    import json

    from arc_agi import EnvironmentInfo, EnvironmentScorecard
    from arc_agi.scorecard import Scorecard
    from leaderboard_submission.scoring import score_timeline

    metadata = tmp_path / "toy" / "version" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"game_id": "toy-v", "baseline_actions": [10, 20, 30]})
    )
    initial = {"grid": [[[0]]], "levels_completed": 0}
    timeline = [initial]
    state = initial
    official = Scorecard(games=["toy-v"])
    official.new_play("toy-v", "guid")
    for action_index in range(1, 7):
        levels = 1 if action_index >= 2 else 0
        levels = 2 if action_index >= 6 else levels
        next_state = {"grid": [[[action_index % 16]]], "levels_completed": levels}
        timeline.append(
            {"state": state, "action": {"action": 1}, "next_state": next_state}
        )
        state = next_state
        official.take_action("toy-v", "guid")
        official.set_levels_completed("toy-v", "guid", levels)

    expected = EnvironmentScorecard.from_scorecard(
        official,
        [EnvironmentInfo(game_id="toy-v", baseline_actions=[10, 20, 30])],
    ).environments[0].runs[0]
    actual = score_timeline("toy", timeline, tmp_path)

    assert actual["score"] == expected.score
    assert [level["actions"] for level in actual["levels"]] == expected.level_actions
    assert [level["score"] for level in actual["levels"]] == expected.level_scores


def test_official_games_uses_every_advertised_base_game_once():
    from types import SimpleNamespace

    from arc_agi import OperationMode
    from leaderboard_submission.workflow import official_games

    class Arcade:
        operation_mode = OperationMode.COMPETITION

        def get_environments(self):
            return [
                SimpleNamespace(game_id="ls20-v1"),
                SimpleNamespace(game_id="re86-v2"),
                SimpleNamespace(game_id="ls20-v2"),
            ]

    with pytest.raises(ValueError, match="multiple versions"):
        official_games(Arcade())


def test_official_scorecard_helpers_reject_non_competition_clients():
    from types import SimpleNamespace

    from arc_agi import OperationMode
    from leaderboard_submission.workflow import open_official_scorecard

    with pytest.raises(ValueError, match="COMPETITION"):
        open_official_scorecard(
            SimpleNamespace(operation_mode=OperationMode.ONLINE),
            source_url="https://github.com/example/project",
        )


def test_competition_arcade_isolated_from_stale_local_environment_metadata(
    monkeypatch,
):
    from arc_agi import OperationMode
    from leaderboard_submission.workflow import create_competition_arcade

    captured = {}

    class Arcade:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("arc_agi.Arcade", Arcade)
    arcade = create_competition_arcade(api_key="key", base_url="https://example.test")

    assert captured["operation_mode"] == OperationMode.COMPETITION
    assert captured["arc_api_key"] == "key"
    assert captured["arc_base_url"] == "https://example.test"
    assert Path(captured["environments_dir"]).is_dir()
    assert Path(arcade._arcagi3_empty_environments_dir) == Path(
        captured["environments_dir"]
    )


def test_official_game_ids_require_one_exact_version_per_base_game():
    from types import SimpleNamespace

    from leaderboard_submission.workflow import official_game_ids, official_games

    class Arcade:
        def get_environments(self):
            return [
                SimpleNamespace(game_id="re86-v2"),
                SimpleNamespace(game_id="ls20-v1"),
            ]

    assert official_game_ids(Arcade()) == ("ls20-v1", "re86-v2")
    assert official_games(Arcade()) == ("ls20", "re86")


def test_current_game_resolves_api_version_and_requires_matching_local_metadata(
    tmp_path,
):
    from types import SimpleNamespace

    from leaderboard_submission.workflow import resolve_current_game

    class Arcade:
        def get_environments(self):
            return [SimpleNamespace(game_id="ls20-current")]

    def factory(**_):
        return Arcade()

    with pytest.raises(FileNotFoundError, match="ls20-current"):
        resolve_current_game("ls20", tmp_path, arcade_factory=factory)

    metadata = tmp_path / "ls20" / "current" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"game_id": "wrong-version"}')
    with pytest.raises(ValueError, match="metadata does not match"):
        resolve_current_game("ls20", tmp_path, arcade_factory=factory)

    metadata.write_text('{"game_id": "ls20-current"}')
    assert (
        resolve_current_game("ls20", tmp_path, arcade_factory=factory) == "ls20-current"
    )


def test_scorecard_preflight_requires_exact_complete_current_suite(tmp_path):
    from leaderboard_submission.workflow import Trajectory, validate_trajectory_set

    state = {"grid": [[[0]]], "levels_completed": 0}
    trajectories = (
        Trajectory("ls20", "ls20-current", 0, tmp_path / "ls20", (state,)),
        Trajectory("re86", "re86-old", 0, tmp_path / "re86", (state,)),
    )

    with pytest.raises(ValueError, match="version mismatches"):
        validate_trajectory_set(
            trajectories,
            ("ls20-current", "re86-current", "ft09-current"),
            strict=True,
        )


def test_scorecard_preflight_accepts_independent_single_run_directories(tmp_path):
    from leaderboard_submission.workflow import Trajectory, validate_trajectory_set

    state = {"grid": [[[0]]], "levels_completed": 0}
    trajectories = (
        Trajectory("re86", "re86-current", 0, tmp_path / "one", (state,)),
        Trajectory("ls20", "ls20-current", 0, tmp_path / "elsewhere", (state,)),
    )

    report = validate_trajectory_set(
        trajectories,
        ("ls20-current", "re86-current"),
    )

    assert report["ready"] is True
    assert report["trajectory_count"] == 2
    assert [item["run_dir"] for item in report["provided"]] == [
        str(tmp_path / "one"),
        str(tmp_path / "elsewhere"),
    ]


def test_scorecard_gather_defaults_to_safe_preflight_without_opening_card(tmp_path):
    from types import SimpleNamespace

    from arc_agi import OperationMode
    from leaderboard_submission import workflow as official

    state = {"grid": [[[0]]], "levels_completed": 0}
    trajectory = official.Trajectory(
        "ls20", "ls20-current", 0, tmp_path / "run", (state,)
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(official, "collect_trajectories", lambda _: (trajectory,))

    class Arcade:
        operation_mode = OperationMode.COMPETITION

        def get_environments(self):
            return [SimpleNamespace(game_id="ls20-current")]

        def create_scorecard(self, **_):
            raise AssertionError("preflight must not open a scorecard")

    try:
        report = official.gather_scorecard(
            [tmp_path / "run"],
            source_url="https://github.com/example/project",
            destination=tmp_path / "scorecard.json",
            arcade_factory=lambda **_: Arcade(),
        )
    finally:
        monkeypatch.undo()

    assert report["ready"] is True


def test_strict_replay_verifies_initial_and_each_action_observation(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    from arcengine import GameAction, GameState
    from leaderboard_submission.workflow import Trajectory, replay_trajectory

    def frame(cell):
        return SimpleNamespace(
            frame=[np.array([[cell]])],
            available_actions=[1],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=1,
        )

    class Environment:
        def __init__(self):
            self.observation_space = frame(0)
            self.action_space = [GameAction.ACTION1]

        def step(self, action, data=None, reasoning=None):
            assert action == GameAction.ACTION1
            assert data == {}
            assert reasoning["action_index"] == 1
            return frame(1)

    class Arcade:
        def make(self, game_id, **kwargs):
            assert game_id == "toy-v"
            assert kwargs["scorecard_id"] == "card"
            return Environment()

    initial = {
        "grid": [[[0]]],
        "legal_actions": [1],
        "state": "NOT_FINISHED",
        "levels_completed": 0,
        "win_levels": 1,
    }
    final = {**initial, "grid": [[[1]]]}
    trajectory = Trajectory(
        "toy",
        "toy-v",
        0,
        tmp_path,
        (initial, {"state": initial, "action": {"action": 1}, "next_state": final}),
    )

    assert replay_trajectory(Arcade(), "card", trajectory) is None


def test_non_strict_scorecard_preflight_reports_missing_games(tmp_path):
    from leaderboard_submission.workflow import Trajectory, validate_trajectory_set

    state = {"grid": [[[0]]], "levels_completed": 0}
    report = validate_trajectory_set(
        (Trajectory("ls20", "ls20-current", 0, tmp_path, (state,)),),
        ("ls20-current", "re86-current"),
        strict=False,
    )

    assert report["ready"] is False
    assert report["missing"] == ["re86"]


def test_environment_sync_dry_run_reports_versions_without_downloading(tmp_path):
    from types import SimpleNamespace

    from leaderboard_submission.workflow import sync_official_environments

    old = tmp_path / "ls20" / "old" / "metadata.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"game_id": "ls20-old"}')

    class Arcade:
        def get_environments(self):
            return [SimpleNamespace(game_id="ls20-current")]

    report = sync_official_environments(
        tmp_path,
        arcade_factory=lambda **_: Arcade(),
    )

    assert report["dry_run"] is True
    assert report["missing_current_versions"] == ["ls20-current"]
    assert report["retained_old_versions"] == ["ls20-old"]
    assert report["downloaded"] == []


def test_versioned_timeline_score_accepts_matching_current_metadata_override(tmp_path):
    from leaderboard_submission.scoring import score_timeline

    timeline = ({"grid": [[[0]]], "levels_completed": 0},)
    score = score_timeline(
        "toy-current",
        timeline,
        tmp_path,
        metadata_override={"game_id": "toy-current", "baseline_actions": [7]},
        metadata_source="official-api",
    )

    assert score["game"] == "toy"
    assert score["game_id"] == "toy-current"


def test_confirmed_gather_persists_progress_and_final_scorecard(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from arc_agi import OperationMode
    from leaderboard_submission import workflow

    state = {"grid": [[[0]]], "levels_completed": 0}
    trajectory = workflow.Trajectory(
        "ls20", "ls20-current", 0, tmp_path / "run", (state,)
    )
    monkeypatch.setattr(workflow, "collect_trajectories", lambda _: (trajectory,))
    monkeypatch.setattr(workflow, "replay_trajectory", lambda *args: None)

    class Scorecard:
        def model_dump(self, **_):
            return {"card_id": "card", "score": 0.0}

    class Arcade:
        operation_mode = OperationMode.COMPETITION
        recordings_dir = ""

        def get_environments(self):
            return [SimpleNamespace(game_id="ls20-current")]

        def create_scorecard(self, **_):
            return "card"

        def close_scorecard(self, _):
            return Scorecard()

    destination = tmp_path / "official.json"
    result = workflow.gather_scorecard(
        [tmp_path / "run"],
        source_url="https://github.com/example/project",
        destination=destination,
        recordings_dir=tmp_path / "recordings",
        confirm=True,
        arcade_factory=lambda **_: Arcade(),
    )

    import json

    progress = json.loads((tmp_path / "official.progress.json").read_text())
    assert result["scorecard_url"].endswith("/card")
    assert progress["status"] == "closed"
    assert progress["completed_game_ids"] == ["ls20-current"]
    assert destination.is_file()


def test_online_rehearsal_is_marked_non_competition(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from arc_agi import OperationMode
    from leaderboard_submission import workflow

    state = {"grid": [[[0]]], "levels_completed": 0}
    trajectory = workflow.Trajectory(
        "ls20", "ls20-current", 0, tmp_path / "run", (state,)
    )
    monkeypatch.setattr(workflow, "collect_trajectories", lambda _: (trajectory,))
    monkeypatch.setattr(workflow, "replay_trajectory", lambda *args: None)

    class Scorecard:
        def model_dump(self, **_):
            return {"card_id": "rehearsal", "score": 0.0}

    class Arcade:
        operation_mode = OperationMode.ONLINE
        recordings_dir = ""

        def get_environments(self):
            return [SimpleNamespace(game_id="ls20-current")]

        def create_scorecard(self, **kwargs):
            assert "trajectory-rehearsal" in kwargs["tags"]
            return "rehearsal"

        def close_scorecard(self, _):
            return Scorecard()

    destination = tmp_path / "rehearsal.json"
    result = workflow.rehearse_scorecard(
        [tmp_path / "run"],
        source_url="https://github.com/example/project",
        destination=destination,
        arcade_factory=lambda **_: Arcade(),
    )

    assert result["competition_mode"] is False
    assert result["rehearsal"] is True

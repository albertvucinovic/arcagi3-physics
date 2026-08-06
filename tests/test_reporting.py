from __future__ import annotations


def test_review_exports_all_timeline_frames_to_animated_gif(tmp_path):
    from PIL import Image

    from arcagi3_physics.gif import export_gif
    from arcagi3_physics.review import Review

    def state(cell, levels):
        return {
            "grid": [[[cell, 0], [0, cell]]],
            "legal_actions": [1],
            "state": "NOT_FINISHED",
            "levels_completed": levels,
            "win_levels": 2,
        }

    initial = state(1, 0)
    first = state(2, 0)
    second = state(3, 1)
    review = Review(
        repository=tmp_path,
        timeline=(
            initial,
            {"state": initial, "action": {"action": 1}, "next_state": first},
            {"state": first, "action": {"action": 1}, "next_state": second},
        ),
        report={},
        head="a" * 40,
        actor_turns=0,
        critic_turns=0,
        actor_commits=0,
        critic_commits=0,
        evaluation_reports=0,
        evaluated_head=None,
        model_count=0,
        surviving_models=(),
        generated_plans=0,
    )
    destination = export_gif(
        review,
        tmp_path / "played.gif",
        scale=3,
        duration_ms=120,
        level_pause_ms=700,
    )

    with Image.open(destination) as image:
        assert image.format == "GIF"
        assert image.size == (6, 20)
        assert image.n_frames == 3
        durations = []
        for index in range(image.n_frames):
            image.seek(index)
            durations.append(image.info["duration"])
        assert durations == [120, 120, 700]


def test_gif_preserves_repeated_observations_as_action_frames(tmp_path):
    from PIL import Image

    from arcagi3_physics.gif import export_gif
    from arcagi3_physics.review import Review

    state = {"grid": [[[0]]], "levels_completed": 0}
    review = Review(
        repository=tmp_path,
        timeline=(
            state,
            {"state": state, "action": {"action": 1}, "next_state": state},
        ),
        report={},
        head="a" * 40,
        actor_turns=0,
        critic_turns=0,
        actor_commits=0,
        critic_commits=0,
        evaluation_reports=0,
        evaluated_head=None,
        model_count=0,
        surviving_models=(),
        generated_plans=0,
    )

    destination = export_gif(review, tmp_path / "repeated.gif", scale=2)

    with Image.open(destination) as image:
        assert image.n_frames == 2

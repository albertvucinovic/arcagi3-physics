from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "runPhysics.sh"


def test_launcher_uses_resolved_api_version_for_game_and_run_dir(tmp_path):
    fake_python = tmp_path / "python"
    fake_submission = tmp_path / "leaderboard.sh"
    log = tmp_path / "calls"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$CALLS"\n'
        "if [[ $* == *'import arc_agi, eggopt, eggthreads'* ]]; then exit 0; fi\n"
    )
    fake_python.chmod(0o755)
    fake_submission.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "submission:$*" >> "$CALLS"\n'
        "printf '%s\\n' ls20-current\n"
    )
    fake_submission.chmod(0o755)

    completed = subprocess.run(
        [str(LAUNCHER)],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHON": str(fake_python),
            "CALLS": str(log),
            "ARC_ENVIRONMENTS_DIR": str(tmp_path / "environments"),
            "ARC_LEADERBOARD": str(fake_submission),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    calls = log.read_text().splitlines()
    assert "ARC API current game: ls20-current" in completed.stdout
    assert any("submission:" in call and "current-game ls20" in call for call in calls)
    solver = next(call for call in calls if "-m arcagi3_physics.run" in call)
    assert "--game ls20-current" in solver
    assert "--run-dir " + str(ROOT / "runs/physics-ls20-current-astar") in solver


def test_launcher_arc_game_override_skips_api_resolution(tmp_path):
    fake_python = tmp_path / "python"
    log = tmp_path / "calls"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$CALLS"\n'
        "if [[ $* == *'import arc_agi, eggopt, eggthreads'* ]]; then exit 0; fi\n"
    )
    fake_python.chmod(0o755)

    subprocess.run(
        [str(LAUNCHER)],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHON": str(fake_python),
            "CALLS": str(log),
            "ARC_GAME": "ls20-old",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    calls = log.read_text().splitlines()
    assert not any("submission:" in call for call in calls)
    solver = next(call for call in calls if "-m arcagi3_physics.run" in call)
    assert "--game ls20-old" in solver
    assert "--run-dir " + str(ROOT / "runs/physics-ls20-old-astar") in solver


def test_launcher_command_line_game_takes_precedence(tmp_path):
    fake_python = tmp_path / "python"
    log = tmp_path / "calls"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$CALLS"\n'
        "if [[ $* == *'import arc_agi, eggopt, eggthreads'* ]]; then exit 0; fi\n"
    )
    fake_python.chmod(0o755)

    subprocess.run(
        [str(LAUNCHER), "--game", "re86-old"],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHON": str(fake_python),
            "CALLS": str(log),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    calls = log.read_text().splitlines()
    assert not any("submission:" in call for call in calls)
    solver = next(call for call in calls if "-m arcagi3_physics.run" in call)
    assert "--game ls20" in solver
    assert solver.endswith("--game re86-old")


def test_core_package_has_no_official_api_launch_module():
    assert not (ROOT / "arcagi3_physics/launch.py").exists()

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from eggopt import Agent, physics_actor_system_prompt
from eggthreads import create_llm_client

from .solver import ARC_DOMAIN_PROMPT, arc_physics


def run(arguments: argparse.Namespace) -> Path:
    models, all_models = _model_paths(arguments.models, arguments.all_models)
    llm = create_llm_client(models_path=models, all_models_path=all_models)
    actor = Agent(
        llm,
        {"role": "arc-physics-actor", "version": 2},
        model_key=arguments.actor_model,
        models_path=models,
        context_limit=_limit(arguments.actor_context_limit),
        auto_approve_tools=True,
        allowed_tools=frozenset({"bash", "python_exec"}),
        system_prompt=physics_actor_system_prompt(ARC_DOMAIN_PROMPT),
    )
    strategy = arc_physics(
        game=arguments.game,
        seed=arguments.seed,
        environments_dir=arguments.environments_dir,
        actor=actor,
        max_depth=arguments.max_plan_depth,
        max_nodes=arguments.max_plan_nodes,
        evaluator_timeout_sec=arguments.critic_timeout,
    )

    print(
        f"ARC Physics: game={arguments.game} seed={arguments.seed} "
        f"actor={arguments.actor_model!r}"
    )
    result = strategy.run(
        run_dir=arguments.run_dir,
        max_actions=arguments.max_actions,
        max_cycles=arguments.max_cycles,
    )
    destination = Path(arguments.run_dir) / "result.json"
    _write_json(destination, asdict(result))
    print(
        f"ARC Physics stopped: {result.stopping_reason}; "
        f"rounds={result.rounds}; head={result.head}"
    )
    print(f"Result: {destination.resolve()}")
    print(
        "Actor repository: "
        f"{(Path(arguments.run_dir) / 'workspace/innerContext').resolve()}"
    )
    print(f"Physics thread: {result.physics_thread_id}")
    print(f"Review: ./reviewPhysics.sh --run-dir {Path(arguments.run_dir).resolve()}")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Git-backed Eggopt PhysicsStrategy on one ARC-AGI-3 game."
    )
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--environments-dir", type=Path, default=Path("environment_files")
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/physics-ls20-seed0"))
    parser.add_argument("--actor-model", default="Pro: GPT-5.6 Sol max")
    parser.add_argument("--models", type=Path)
    parser.add_argument("--all-models", type=Path)
    parser.add_argument("--max-actions", type=_positive, default=50)
    parser.add_argument("--max-cycles", type=_positive, default=100)
    parser.add_argument("--max-plan-depth", type=_positive, default=8)
    parser.add_argument("--max-plan-nodes", type=_positive, default=10_000)
    parser.add_argument(
        "--critic-timeout",
        type=_positive,
        default=300,
        help="Maximum seconds for each trusted Critic evaluator subprocess.",
    )
    parser.add_argument(
        "--actor-context-limit",
        type=_non_negative,
        default=0,
        help="Full-history token limit; 0 means unlimited.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


def _model_paths(models: Path | None, all_models: Path | None) -> tuple[str, str]:
    if models is None:
        from eggconfig import get_all_models_path, get_models_path

        return str(get_models_path()), str(get_all_models_path())
    catalog = all_models or models.with_name("all-models.json")
    return str(models), str(catalog)


def _limit(value: int) -> int | None:
    return value or None


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as temporary:
        json.dump(value, temporary, indent=2, sort_keys=True, default=repr)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from typing import Any

from eggflow import ContextLimitExceededError, Task, TaskError, wrapped
from eggopt import Agent, PhysicsResult, physics_actor_system_prompt
from eggopt.identity import digest_payload
from eggopt.runtime import Runtime
from eggthreads import (
    RunnerConfig,
    SubtreeScheduler,
    collect_subtree,
    continue_child_thread_manually,
    create_child_thread,
    create_llm_client,
    create_root_thread,
    diagnose_thread,
    list_children_with_meta,
    list_root_threads,
    list_threads,
    load_thread_projection,
    thread_token_stats,
    validate_model_handle,
)

from .run import _limit, _model_paths, _write_json
from .solver import ARC_DOMAIN_PROMPT, arc_physics

ROOT_NAME = "arc-agi-3-public"
DEFAULT_MODEL = "Pro: GPT-5.6 Luna max"


@dataclass(frozen=True)
class EnvironmentResult:
    game: str
    status: str
    run_dir: str
    physics_thread_id: str
    result_path: str | None = None
    stopping_reason: str | None = None
    actions: int = 0
    rounds: int = 0
    head: str | None = None
    error: str | None = None
    traceback_path: str | None = None
    duration_sec: float = 0.0


@dataclass(frozen=True)
class PreparedBenchmark:
    run_dir: str
    root_thread_id: str
    games: tuple[str, ...]
    physics_thread_ids: tuple[str, ...]


@dataclass
class _EnsureBenchmarkRoot(Task):
    threads: Any = field(repr=False, compare=False)
    model: str
    models_path: str
    all_models_path: str

    def get_cache_key(self) -> str:
        return digest_payload(
            "arc-agi-3.public-benchmark.root.v1",
            {"name": ROOT_NAME, "model": self.model},
        )

    def run(self) -> str:
        roots = [
            thread_id
            for thread_id in list_root_threads(self.threads)
            if self.threads.get_thread(thread_id).name == ROOT_NAME
        ]
        if len(roots) > 1:
            raise RuntimeError(f"benchmark has multiple {ROOT_NAME!r} root threads")
        if roots:
            return roots[0]
        return create_root_thread(
            self.threads,
            name=ROOT_NAME,
            initial_model_key=self.model,
            models_path=self.models_path,
            all_models_path=self.all_models_path,
        )


@dataclass
class _EnsurePhysicsRun(Task):
    threads: Any = field(repr=False, compare=False)
    root_id: str
    game: str
    model: str
    models_path: str
    all_models_path: str

    def get_cache_key(self) -> str:
        return digest_payload(
            "arc-agi-3.public-benchmark.physics-run.v1",
            {"root": self.root_id, "game": self.game, "model": self.model},
        )

    def run(self) -> str:
        name = f"Physics {self.game}"
        matches = [
            thread_id
            for thread_id, child_name, *_ in list_children_with_meta(
                self.threads, self.root_id
            )
            if child_name == name
        ]
        if len(matches) > 1:
            raise RuntimeError(f"benchmark has duplicate {name!r} threads")
        if matches:
            return matches[0]
        return create_child_thread(
            self.threads,
            self.root_id,
            name=name,
            initial_model_key=self.model,
            models_path=self.models_path,
            all_models_path=self.all_models_path,
            inherit_tools_config=False,
        )


@dataclass
class _RunEnvironment(Task):
    # The wrapper records failures as values so one environment cannot abort the
    # batch. Keep it uncached: completed studies short-circuit via result.json,
    # while failed/interrupted studies must re-enter Physics recovery on rerun.
    cacheable = False

    strategy: Any = field(repr=False, compare=False)
    game: str
    run_dir: str
    runtime_key: str
    physics_thread_id: str
    max_actions: int
    max_cycles: int

    def get_cache_key(self) -> str:
        return digest_payload(
            "arc-agi-3.public-benchmark.environment.v1",
            {
                "game": self.game,
                "run_dir": self.run_dir,
                "physics_thread": self.physics_thread_id,
                "max_actions": self.max_actions,
                "max_cycles": self.max_cycles,
                "strategy": self.strategy.identity,
                "actor": self.strategy.actor.task_identity,
            },
        )

    def run(self):
        started = monotonic()
        run_dir = Path(self.run_dir)
        completed = run_dir / "result.json"
        if completed.is_file():
            try:
                return _completed_result(
                    self.game,
                    run_dir,
                    self.physics_thread_id,
                    _read_json(completed),
                )
            except (KeyError, TypeError, ValueError):
                pass
        _write_status(
            run_dir,
            {
                "game": self.game,
                "status": "running",
                "physics_thread_id": self.physics_thread_id,
            },
        )
        try:
            outcome = yield wrapped(
                self.strategy.task(
                    runtime_key=self.runtime_key,
                    run_dir=run_dir,
                    physics_thread_id=self.physics_thread_id,
                    max_actions=self.max_actions,
                    max_cycles=self.max_cycles,
                )
            )
        except (ContextLimitExceededError, TaskError) as exc:
            if isinstance(exc, ContextLimitExceededError) or exc.is_terminal:
                return _terminal_failure(
                    self.game,
                    run_dir,
                    self.physics_thread_id,
                    started,
                    exc,
                )
            return _failure(self.game, run_dir, self.physics_thread_id, started, exc)
        if not outcome.is_success:
            if outcome.is_terminal:
                return _terminal_failure(
                    self.game,
                    run_dir,
                    self.physics_thread_id,
                    started,
                    outcome.error or "Physics run reached a terminal limit",
                )
            return _failure(
                self.game,
                run_dir,
                self.physics_thread_id,
                started,
                RuntimeError(outcome.error or "Physics run failed"),
            )
        if not isinstance(outcome.value, PhysicsResult):
            return _failure(
                self.game,
                run_dir,
                self.physics_thread_id,
                started,
                TypeError(f"unexpected Physics result: {type(outcome.value).__name__}"),
            )
        result = outcome.value
        failure_path = run_dir / "failure.txt"
        if failure_path.exists():
            failure_path.unlink()
        destination = run_dir / "result.json"
        _write_json(destination, asdict(result))
        summary = EnvironmentResult(
            game=self.game,
            status="completed",
            run_dir=str(run_dir),
            physics_thread_id=self.physics_thread_id,
            result_path=str(destination),
            stopping_reason=result.stopping_reason,
            actions=result.actions,
            rounds=result.rounds,
            head=result.head,
            duration_sec=monotonic() - started,
        )
        _write_status(run_dir, asdict(summary))
        return summary


def discover_public_environments(environments_dir: str | Path) -> tuple[str, ...]:
    """Return deterministic base game IDs from locally downloaded public metadata."""

    root = Path(environments_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ARC environments directory not found: {root}")
    games = set()
    for metadata in root.rglob("metadata.json"):
        data = _read_json(metadata)
        game_id = data.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            raise ValueError(f"ARC metadata has no game_id: {metadata}")
        games.add(game_id.split("-", 1)[0])
    if not games:
        raise ValueError(f"no ARC environment metadata found under {root}")
    return tuple(sorted(games))


def run(arguments: argparse.Namespace) -> tuple[Path, int]:
    return asyncio.run(_run(arguments))


async def _run(arguments: argparse.Namespace) -> tuple[Path, int]:
    run_dir = Path(arguments.run_dir).expanduser().resolve()
    environments_dir = Path(arguments.environments_dir).expanduser().resolve()
    discovered_games = discover_public_environments(environments_dir)
    if not arguments.games:
        _require_complete_public_suite(discovered_games)
    games = _selected_games(discovered_games, arguments.games)
    models, all_models = _model_paths(arguments.models, arguments.all_models)
    if not validate_model_handle(
        arguments.actor_model, models, all_models_path=all_models
    ):
        raise ValueError(
            f"model {arguments.actor_model!r} is unavailable in {models} or {all_models}"
        )
    _validate_luna_max(arguments.actor_model, models, all_models)
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
        scheduler_managed=True,
    )

    print(
        f"ARC public benchmark: selected={len(games)}/{len(discovered_games)}; "
        f"actor={arguments.actor_model!r}; parallel={arguments.max_parallel}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _ensure_configuration(
        run_dir / "benchmark.json",
        _configuration(arguments, discovered_games, environments_dir, models),
    )

    with Runtime.open(run_dir) as runtime:
        root_id, physics_ids = await _ensure_thread_tree(
            runtime,
            games,
            arguments.actor_model,
            models,
            all_models,
        )
        repaired = _recover_threads_on_restart(runtime.threads, root_id)
        if repaired:
            print(f"Restart recovery continued {repaired} interrupted thread(s).")

        scheduler = SubtreeScheduler(
            runtime.threads,
            root_thread_id=root_id,
            llm=llm,
            config=RunnerConfig(
                max_concurrent_threads=arguments.max_parallel,
                max_concurrent_llm_threads=arguments.max_parallel,
                priority_mode="alphabetical",
                api_timeout_sec=0,
                tool_timeout_sec=0,
            ),
            models_path=models,
            all_models_path=all_models,
            tools=actor.tools,
        )
        tasks = [
            _RunEnvironment(
                strategy=arc_physics(
                    game=game,
                    seed=arguments.seed,
                    environments_dir=environments_dir,
                    actor=actor,
                    max_depth=arguments.max_plan_depth,
                    max_nodes=arguments.max_plan_nodes,
                ),
                game=game,
                run_dir=str(run_dir / f"Physics {game}"),
                runtime_key=runtime.runtime_key,
                physics_thread_id=physics_id,
                max_actions=arguments.max_actions,
                max_cycles=arguments.max_cycles,
            )
            for game, physics_id in zip(games, physics_ids, strict=True)
        ]
        scheduler_task = asyncio.create_task(scheduler.run_forever(poll_sec=0.05))
        batch_task = None
        results = None
        pending_error: BaseException | None = None
        try:
            batch_task = asyncio.create_task(runtime.flow.run(tasks))
            done, _ = await asyncio.wait(
                {batch_task, scheduler_task}, return_when=asyncio.FIRST_COMPLETED
            )
            error = scheduler_task.exception() if scheduler_task in done else None
            if scheduler_task in done and error is not None:
                batch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await batch_task
                raise RuntimeError("benchmark scheduler failed") from error
            if scheduler_task in done and not batch_task.done():
                batch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await batch_task
                raise RuntimeError("benchmark scheduler stopped unexpectedly")
            results = await batch_task
        except asyncio.CancelledError as exc:
            if batch_task is not None and not batch_task.done():
                batch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await batch_task
            pending_error = exc
        except (
            TaskError,
            RuntimeError,
            TypeError,
            ValueError,
            KeyboardInterrupt,
        ) as exc:
            if batch_task is not None and not batch_task.done():
                batch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await batch_task
            pending_error = exc
        finally:
            scheduler_task.cancel()
            cleanup = asyncio.create_task(
                _shutdown_scheduler(scheduler, scheduler_task)
            )
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as exc:
                if pending_error is None:
                    pending_error = exc
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                cleanup.result()

        if pending_error is not None:
            results = _status_results(
                games,
                dict(zip(games, physics_ids, strict=True)),
                run_dir,
            )
        if results is None:
            raise RuntimeError("benchmark produced no environment results")

        summary_path = _write_summary(
            arguments,
            root_id,
            games,
            games,
            results,
            dict(zip(games, physics_ids, strict=True)),
            runtime.threads,
            llm,
        )

    if pending_error is not None:
        raise pending_error
    summary = _read_json(summary_path)
    selected = {
        item["game"]: item
        for item in summary["environments"]
        if item["game"] in set(games)
    }
    completed = sum(item["status"] == "completed" for item in selected.values())
    failed = sum(item["status"] == "failed" for item in selected.values())
    terminal = sum(item["status"] == "terminal" for item in selected.values())
    print(
        f"Benchmark stopped: completed={completed}; failed={failed}; "
        f"terminal={terminal}"
    )
    print(f"Summary: {summary_path}")
    if failed:
        print(
            "Rerun the same command to retry failed environments and reuse completed work."
        )
    if terminal:
        print("Terminal environments require a new run directory or a higher limit.")
    return summary_path, 1 if failed or terminal else 0


async def prepare(arguments: argparse.Namespace) -> PreparedBenchmark:
    """Create only the durable benchmark root and Physics children."""

    run_dir = Path(arguments.run_dir).expanduser().resolve()
    environments_dir = Path(arguments.environments_dir).expanduser().resolve()
    discovered_games = discover_public_environments(environments_dir)
    if not arguments.games:
        _require_complete_public_suite(discovered_games)
    games = _selected_games(discovered_games, arguments.games)
    models, all_models = _model_paths(arguments.models, arguments.all_models)
    if not validate_model_handle(
        arguments.actor_model, models, all_models_path=all_models
    ):
        raise ValueError(f"model {arguments.actor_model!r} is unavailable")
    _validate_luna_max(arguments.actor_model, models, all_models)
    run_dir.mkdir(parents=True, exist_ok=True)
    _ensure_configuration(
        run_dir / "benchmark.json",
        _configuration(arguments, games, environments_dir, models),
    )
    with Runtime.open(run_dir) as runtime:
        root_id, physics_ids = await _ensure_thread_tree(
            runtime,
            games,
            arguments.actor_model,
            models,
            all_models,
        )
    return PreparedBenchmark(str(run_dir), root_id, games, tuple(physics_ids))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PhysicsStrategy on all downloaded public ARC-AGI-3 environments."
    )
    parser.add_argument(
        "--environments-dir", type=Path, default=Path("environment_files")
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/luna-public-benchmark")
    )
    parser.add_argument("--actor-model", default=DEFAULT_MODEL)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--all-models", type=Path)
    parser.add_argument("--games", nargs="*", help="Optional base game IDs to run.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-parallel", type=_positive, default=3)
    parser.add_argument("--max-actions", type=_positive, default=50)
    parser.add_argument("--max-cycles", type=_positive, default=100)
    parser.add_argument("--max-plan-depth", type=_positive, default=8)
    parser.add_argument("--max-plan-nodes", type=_positive, default=10_000)
    parser.add_argument(
        "--actor-context-limit",
        type=_non_negative,
        default=300_000,
        help="Full-history token limit per Actor; 0 means unlimited.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _summary_path, exit_code = run(build_parser().parse_args(argv))
    return exit_code


def _recover_threads_on_restart(threads, root_id: str) -> int:
    """Apply EvolveTropy's diagnosis/manual-continue restart policy."""

    recovered = 0
    for thread_id in collect_subtree(threads, root_id):
        if thread_id == root_id:
            continue
        diagnosis = diagnose_thread(threads, thread_id)
        if diagnosis.is_healthy or _context_limited(threads, thread_id):
            continue
        result = continue_child_thread_manually(
            threads,
            root_id,
            thread_id,
            msg_id=diagnosis.suggested_continue_point,
            source="ARC public benchmark restart recovery",
        )
        if not result.success:
            raise RuntimeError(
                f"failed to recover thread {thread_id}: {result.message}"
            )
        recovered += 1
    return recovered


async def _ensure_thread_tree(runtime, games, model, models, all_models):
    root_id = await runtime.flow.run(
        _EnsureBenchmarkRoot(runtime.threads, model, models, all_models)
    )
    physics_ids = []
    for game in games:
        physics_ids.append(
            await runtime.flow.run(
                _EnsurePhysicsRun(
                    runtime.threads,
                    root_id,
                    game,
                    model,
                    models,
                    all_models,
                )
            )
        )
    return root_id, physics_ids


async def _shutdown_scheduler(scheduler, scheduler_task):
    with contextlib.suppress(asyncio.CancelledError):
        await scheduler_task
    await scheduler.shutdown()


def _context_limited(threads, thread_id: str) -> bool:
    try:
        messages = [
            message.payload
            for message in load_thread_projection(threads, thread_id).messages[-8:]
        ]
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return any(
        message.get("role") == "system"
        and "context" in str(message.get("content", "")).lower()
        and any(
            word in str(message.get("content", "")).lower()
            for word in ("limit", "exceed", "too long")
        )
        for message in messages
    )


def _selected_games(available: Sequence[str], requested: Iterable[str] | None):
    if not requested:
        return tuple(available)
    selected = tuple(dict.fromkeys(str(game).strip() for game in requested if game))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown public ARC environments: {', '.join(unknown)}")
    return selected


def _require_complete_public_suite(games: Sequence[str]) -> None:
    if len(games) != 25:
        raise ValueError(
            "the default Luna benchmark requires all 25 public ARC-AGI-3 "
            f"environments, but discovered {len(games)}; use --games explicitly "
            "for a partial diagnostic run"
        )


def _failure(game, run_dir, thread_id, started, exc):
    error = f"{type(exc).__name__}: {exc}"
    trace = run_dir / "failure.txt"
    trace.parent.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        result_path.unlink()
    with NamedTemporaryFile("w", dir=trace.parent, delete=False) as temporary:
        temporary.write("".join(traceback.format_exception(exc)))
        temporary_path = Path(temporary.name)
    temporary_path.replace(trace)
    result = EnvironmentResult(
        game=game,
        status="failed",
        run_dir=str(run_dir),
        physics_thread_id=thread_id,
        error=error,
        traceback_path=str(trace),
        duration_sec=monotonic() - started,
    )
    _write_status(run_dir, asdict(result))
    return result


def _terminal_failure(game, run_dir, thread_id, started, error):
    result_path = run_dir / "result.json"
    if result_path.exists():
        result_path.unlink()
    failure_path = run_dir / "failure.txt"
    if failure_path.exists():
        failure_path.unlink()
    result = EnvironmentResult(
        game=game,
        status="terminal",
        run_dir=str(run_dir),
        physics_thread_id=thread_id,
        error=str(error),
        duration_sec=monotonic() - started,
    )
    _write_status(run_dir, asdict(result))
    return result


def _completed_result(game, run_dir, thread_id, value):
    if value.get("physics_thread_id") != thread_id:
        raise ValueError("persisted result belongs to another Physics thread")
    payload = value.get("value")
    if isinstance(payload, dict):
        actions = int(payload.get("actions", 0))
    else:
        timeline = getattr(payload, "timeline", ())
        actions = (
            max(0, len(timeline) - 1) if isinstance(timeline, (list, tuple)) else 0
        )
    result = EnvironmentResult(
        game=game,
        status="completed",
        run_dir=str(run_dir),
        physics_thread_id=thread_id,
        result_path=str(run_dir / "result.json"),
        stopping_reason=_required_string(value, "stopping_reason"),
        actions=actions,
        rounds=int(value["rounds"]),
        head=value.get("head"),
    )
    _write_status(run_dir, asdict(result))
    return result


def _required_string(value: dict[str, Any], name: str) -> str:
    item = value[name]
    if not isinstance(item, str) or not item:
        raise TypeError(f"persisted result {name} must be a non-empty string")
    return item


def _status_results(games, physics_ids, run_dir: Path):
    results = []
    for game in games:
        environment_dir = run_dir / f"Physics {game}"
        status_path = environment_dir / "status.json"
        if status_path.is_file():
            try:
                value = _read_json(status_path)
                results.append(_environment_result(value))
                continue
            except (TypeError, ValueError):
                pass
        results.append(
            EnvironmentResult(
                game,
                "interrupted",
                str(environment_dir),
                physics_ids[game],
            )
        )
    return results


def _environment_result(value: dict[str, Any]) -> EnvironmentResult:
    required = {"game", "status", "run_dir", "physics_thread_id"}
    if not required <= value.keys():
        raise ValueError("environment status is incomplete")
    game = value["game"]
    status = value["status"]
    run_dir = value["run_dir"]
    thread_id = value["physics_thread_id"]
    if any(
        not isinstance(item, str) or not item
        for item in (game, status, run_dir, thread_id)
    ):
        raise TypeError("environment status string fields must be non-empty")
    return EnvironmentResult(
        game=game,
        status=status,
        run_dir=run_dir,
        physics_thread_id=thread_id,
        result_path=value.get("result_path"),
        stopping_reason=value.get("stopping_reason"),
        actions=int(value.get("actions", 0)),
        rounds=int(value.get("rounds", 0)),
        head=value.get("head"),
        error=value.get("error"),
        traceback_path=value.get("traceback_path"),
        duration_sec=float(value.get("duration_sec", 0.0)),
    )


def _summary(
    arguments,
    root_id,
    public_games,
    selected_games,
    results,
    physics_ids,
    threads,
    llm,
):
    rows = {thread.thread_id: thread for thread in list_threads(threads)}
    selected_results = dict(zip(selected_games, results, strict=True))
    per_game = []
    total_cost = 0.0
    total_input = total_output = 0
    for game in public_games:
        run_dir = Path(arguments.run_dir).resolve() / f"Physics {game}"
        raw_result = selected_results.get(game)
        if isinstance(raw_result, EnvironmentResult):
            result = raw_result
        elif game in selected_results:
            result = EnvironmentResult(
                game=game,
                status="failed",
                run_dir=str(run_dir),
                physics_thread_id=physics_ids[game],
                error=f"unexpected benchmark result: {raw_result!r}",
            )
        elif (run_dir / "result.json").is_file():
            try:
                result = _completed_result(
                    game,
                    run_dir,
                    physics_ids[game],
                    _read_json(run_dir / "result.json"),
                )
            except (KeyError, TypeError, ValueError):
                result = EnvironmentResult(
                    game,
                    "pending",
                    str(run_dir),
                    physics_ids[game],
                )
        elif (run_dir / "status.json").is_file():
            try:
                result = _environment_result(_read_json(run_dir / "status.json"))
            except (TypeError, ValueError):
                result = EnvironmentResult(
                    game,
                    "pending",
                    str(run_dir),
                    physics_ids[game],
                )
        else:
            result = EnvironmentResult(
                game,
                "pending",
                str(run_dir),
                physics_ids[game],
            )
        subtree = (
            collect_subtree(threads, result.physics_thread_id)
            if threads.get_thread(result.physics_thread_id) is not None
            else []
        )
        cost = 0.0
        input_tokens = output_tokens = 0
        for thread_id in subtree:
            stats = thread_token_stats(threads, thread_id, llm=llm)
            usage = stats.get("api_usage", {})
            cost += float(usage.get("cost_usd", {}).get("total", 0.0) or 0.0)
            input_tokens += int(usage.get("total_input_tokens", 0) or 0)
            output_tokens += int(usage.get("total_output_tokens", 0) or 0)
        total_cost += cost
        total_input += input_tokens
        total_output += output_tokens
        per_game.append(
            {
                **asdict(result),
                "thread_name": (
                    rows[result.physics_thread_id].name
                    if result.physics_thread_id in rows
                    else None
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
            }
        )
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "root_name": ROOT_NAME,
        "root_thread_id": root_id,
        "model": arguments.actor_model,
        "seed": arguments.seed,
        "max_parallel": arguments.max_parallel,
        "environment_count": len(public_games),
        "selected_count": len(selected_games),
        "completed": sum(item["status"] == "completed" for item in per_game),
        "failed": sum(item["status"] == "failed" for item in per_game),
        "terminal": sum(item["status"] == "terminal" for item in per_game),
        "interrupted": sum(item["status"] == "interrupted" for item in per_game),
        "pending": sum(item["status"] == "pending" for item in per_game),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost_usd": total_cost,
        "environments": per_game,
    }


def _write_summary(
    arguments,
    root_id,
    public_games,
    selected_games,
    results,
    physics_ids,
    threads,
    llm,
) -> Path:
    destination = Path(arguments.run_dir).expanduser().resolve() / "summary.json"
    _write_json(
        destination,
        _summary(
            arguments,
            root_id,
            public_games,
            selected_games,
            results,
            physics_ids,
            threads,
            llm,
        ),
    )
    return destination


def _write_status(run_dir: Path, value: dict[str, Any]) -> None:
    _write_json(
        run_dir / "status.json",
        {"updated_at": datetime.now(UTC).isoformat(), **value},
    )


def _ensure_configuration(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        existing = _read_json(path)
        canonical = json.loads(json.dumps(value))
        if existing != canonical:
            raise ValueError(
                f"benchmark configuration changed for existing run: {path}; "
                "use a new run directory"
            )
        return
    _write_json(path, value)


def _configuration(
    arguments, games, environments_dir: Path, models: str | None = None
) -> dict[str, Any]:
    return {
        "root_name": ROOT_NAME,
        "model": arguments.actor_model,
        "models": str(Path(models).resolve()) if models is not None else None,
        "seed": arguments.seed,
        "games": games,
        "environments_dir": str(environments_dir),
        "max_actions": arguments.max_actions,
        "max_cycles": arguments.max_cycles,
        "actor_context_limit": arguments.actor_context_limit,
        "max_plan_depth": arguments.max_plan_depth,
        "max_plan_nodes": arguments.max_plan_nodes,
    }


def _validate_luna_max(model: str, models: str, all_models: str) -> None:
    if model != DEFAULT_MODEL:
        return
    llm = create_llm_client(models_path=models, all_models_path=all_models)
    config = llm.registry.get_model_config(model)
    if (
        config.get("model_name") != "gpt-5.6-luna"
        or config.get("parameters", {}).get("reasoning_effort") != "max"
    ):
        raise ValueError(f"{DEFAULT_MODEL!r} must resolve to Luna with max reasoning")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ARC metadata: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"ARC metadata is not an object: {path}")
    return value


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


if __name__ == "__main__":
    raise SystemExit(main())

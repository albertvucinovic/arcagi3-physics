# Luna ARC-AGI-3 public benchmark

Install the project first as documented in [README.md](README.md). The
benchmark uses the installed, commit-pinned Eggmono dependency; it does not
require a local Eggmono checkout or `PYTHONPATH` setup.

Run every locally downloaded public ARC-AGI-3 environment with Luna at maximum
reasoning effort:

```bash
./runLunaBenchmark.sh
```

Before starting a benchmark intended for a hosted scorecard, synchronize exact
current API versions and follow the separate submission workflow in
[leaderboard-submission/README.md](leaderboard-submission/README.md). The
benchmark's 25 local base-game check is not a substitute for current versioned
API discovery.

The default command verifies that all 25 public environments are present before
starting. `--games` explicitly opts into a partial diagnostic/recovery invocation.

The default run is durable and resumable at `runs/luna-public-benchmark`:

```text
runs/luna-public-benchmark/
├── .egg/
│   ├── flow.db
│   └── threads.sqlite
├── benchmark.json
├── summary.json
├── Physics ar25/
│   ├── status.json
│   ├── result.json                 # after completion
│   └── workspace/
└── ...
```

The single Eggthreads tree mirrors it:

```text
arc-agi-3-public
├── Physics ar25
│   └── Critic
│       └── Actor
├── Physics bp35
│   └── Critic
│       └── Actor
└── ...
```

One shared `SubtreeScheduler` admits at most three concurrent LLM turns by
default. Tool turns remain independently schedulable according to Eggthreads'
resource-class policy. Each Actor defaults to a 300,000-token full-history limit,
matching the configured Luna context; set `ARC_ACTOR_CONTEXT_LIMIT=0` to make the
Eggopt limit unlimited.

Sticky scheduling is disabled for benchmark runs, so an Actor can yield its LLM
slot while its trusted Critic runs the bounded evaluator. The configured sticky
idle threshold remains 5 seconds but has no effect while sticky scheduling is
disabled.

ARC domain completion is independent of Physics safety budgets: trusted public
state `WIN` stops successfully, while `GAME_OVER` or an empty legal-action list
stops as an unsuccessful terminal game state. `max_actions`, `max_cycles`, and
the context limit remain separate safety bounds.

## Recovery

Every environment is an independent cached Eggflow composite. If one fails, its
`status.json` and `failure.txt` record the failure while sibling environments
continue. The launcher exits nonzero after the batch if any environment remains
failed or terminal. It writes aggregate `summary.json` after ordinary completion
and after handled scheduler/batch errors; `status.json` remains the durable
per-environment checkpoint if the OS terminates the process abruptly. Running
the same command again:

1. reopens the same root and child threads;
2. repairs interrupted descendants using Eggthreads diagnosis and manual
   continuation (the EvolveTropy restart pattern);
3. reuses completed cached Physics tasks; and
4. retries failed or interrupted non-terminal work without repeating its durable
   primitive effects.

A terminal full-context limit is not automatically retried. Increase the limit
and use a new benchmark run directory for that environment or benchmark.

Run or resume selected environments only:

```bash
ARC_BENCHMARK_RUN_DIR="$PWD/runs/luna-ar25-bp35" \
  ./runLunaBenchmark.sh --games ar25 bp35
```

`--games` creates a partial diagnostic benchmark, so use a different
`ARC_BENCHMARK_RUN_DIR` from the complete 25-environment benchmark. Use a new run
directory for any different model, seed, selected suite, or resource
configuration:

```bash
ARC_BENCHMARK_RUN_DIR="$PWD/runs/luna-smoke" \
  ./runLunaBenchmark.sh --games ar25
```

Review one environment with the existing viewer:

```bash
./reviewPhysics.sh \
  --run-dir "$PWD/runs/luna-public-benchmark/Physics ar25"
```

Useful overrides include `ARC_MAX_PARALLEL`, `ARC_MAX_ACTIONS`,
`ARC_MAX_CYCLES`, `ARC_ACTOR_CONTEXT_LIMIT`, `ARC_CRITIC_TIMEOUT`, and
`ARC_BENCHMARK_RUN_DIR`. `ARC_CRITIC_TIMEOUT` defaults to 300 seconds per trusted
evaluator subprocess.

The shared scheduler retains Eggthreads' normal defaults: a 600-second provider
inactivity timeout and a 30-second tool-call timeout. The Critic evaluator timeout
is an additional, independently configured boundary around submitted
`world_model.py` code.

Attach the Egg UI to the shared tree without starting another scheduler:

```bash
cd runs/luna-public-benchmark
NO_API_CALLS=1 egg.sh
```

The optional Egg UI executable is not part of this repository's runtime
dependency. Install the Egg frontend separately if you want this inspection
workflow; the benchmark itself and `arcagi3-physics-review` do not require it.

# Luna ARC-AGI-3 public benchmark

Run every locally downloaded public ARC-AGI-3 environment with Luna at maximum
reasoning effort:

```bash
./runLunaBenchmark.sh
```

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

Attach the Egg UI to the shared tree without starting another scheduler:

```bash
cd runs/luna-public-benchmark
NO_API_CALLS=1 egg.sh
```

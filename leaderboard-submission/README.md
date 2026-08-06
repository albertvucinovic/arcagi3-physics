# ARC-AGI-3 leaderboard submission

Everything concerned with local RHAE scoring, official environment-version
synchronization, Competition Mode trajectory replay, and hosted scorecard
submission lives in this directory. The core `arcagi3_physics` package only
solves and records authoritative Timelines; it never sees a score.

The root project install is sufficient for `leaderboard.sh`. This folder also
has its own `pyproject.toml` and can be installed as a separate command after
installing the sibling core project:

```bash
venv/bin/python -m pip install ./leaderboard-submission
arcagi3-leaderboard environments
```

## What a hosted scorecard requires

A scorecard is not an upload of `score.json` or per-level action counts. The ARC
API must receive every recorded environment action. Expensive model inference
does not need to be repeated: this workflow can rapidly replay complete
Timelines produced by local Physics runs.

Competition Mode scores **every environment currently advertised by the API**,
including unplayed environments as zero. Therefore the required set is neither
“whatever is downloaded” nor a permanent hard-coded list. It is the exact
versioned set returned immediately before replay by:

```bash
./leaderboard-submission/leaderboard.sh environments
```

On 2026-08-05 the API advertised exactly these 25 environments:

```text
ar25-0c556536  bp35-0a0ad940  cd82-fb555c5d  cn04-2fe56bfb  dc22-fdcac232
ft09-0d8bbf25  g50t-5849a774  ka59-38d34dbb  lf52-271a04aa  lp85-305b61c3
ls20-9607627b  m0r0-492f87ba  r11l-495a7899  re86-8af5384d  s5i5-18d95033
sb26-7fbdac44  sc25-635fd71a  sk48-d8078629  sp80-589a99af  su15-1944f8ab
tn36-ef4dde99  tr87-cd924810  tu93-0768757b  vc33-5430563c  wa30-ee6fef47
```

The command is authoritative if this list changes. It deliberately creates an
API-only client in an empty temporary environment directory, so stale local
metadata cannot pollute discovery.

## 1. Configure keys

The root `.env` is sourced by `leaderboard.sh`. Keep it private:

```bash
export ARC_API_KEY=***
export OPENAI_API_KEY=***       # only needed for solving, not replay
```

An anonymous ARC key is sufficient for discovery/download during the preview,
but use the account/API key under which the final scorecard should be hosted.

## 2. Synchronize current environment versions

Preview the difference without modifying files:

```bash
./leaderboard-submission/leaderboard.sh environments \
  --environments-dir environment_files
```

Download every missing current version:

```bash
./leaderboard-submission/leaderboard.sh environments \
  --environments-dir environment_files \
  --sync
```

Synchronization retains superseded version directories so their old runs can
still be checked, while it refreshes the metadata and source for all 25 exact
current versions (including same-version upstream source/baseline updates).
A stable `game_id` therefore does not by itself prove byte-identical behavior;
the strict `validate` command below must replay the whole Timeline after every
refresh.
Any local base games not advertised by the API are reported separately and do
not become scorecard requirements.
New base-game runs select the newest
`date_downloaded`; passing a full versioned ID selects that exact version.
Each new run records its resolved `game_id` in `<run-dir>/run.json`.
For legacy runs created before that field existed, the collector tries every
local version and accepts only a unique exact full-Timeline replay. If multiple
versions behave identically over the recorded prefix, it refuses to guess.

After synchronization, do **not** resume an old unfinished run when its selected
environment version changed. Start a new run directory for the current version.
The launchers enforce this for legacy workspaces that predate `run.json` version
recording, preventing a resumed Timeline from silently switching environments.

## 3. Produce one run per current environment

Independent `runPhysics.sh` runs are valid inputs. They do not need to share a
benchmark root. Use a distinct directory and preferably the exact versioned ID:

```bash
ARC_GAME=ls20-9607627b \
ARC_RUN_DIR="$PWD/runs/scorecard-ls20" \
./runPhysics.sh

ARC_GAME=re86-8af5384d \
ARC_RUN_DIR="$PWD/runs/scorecard-re86" \
./runPhysics.sh
```

Repeat for all IDs reported by `leaderboard.sh environments`. The scorecard
collector accepts repeated `--run-dir` flags in any order. Each run must have:

- a Critic `workspace/critic-repository` containing `.trusted/state.json`;
- a complete Timeline starting at `levels_completed=0`;
- every real action in exact order, including ACTION6 coordinates;
- the same exact versioned `game_id` currently advertised by the API;
- no missing environment and no duplicate base game.

No-reset runs are ideal. The initial API `RESET` that creates a game session is
unavoidable and is not a replayed agent action; the collector submits no extra
resets. A Timeline may end in `WIN`, `GAME_OVER`, or still active state; its
completed levels/actions are what ARC scores.

## 4. Inspect local RHAE reports (optional)

This is reporting only and does not mint a hosted scorecard:

```bash
./leaderboard-submission/leaderboard.sh score \
  --run-dir runs/scorecard-ls20 \
  --run-dir runs/scorecard-re86 \
  --output runs/local-rhae.json
```

Add `--current-metadata` to fetch current API baselines. A changed game version
is rejected rather than combining old actions with a new baseline. Each run gets
`score.json`; the optional output contains their aggregate.

## 5. Safe preflight—does not open a scorecard

First replay all gathered Timelines against the refreshed exact local sources:

```bash
./leaderboard-submission/leaderboard.sh validate \
  --environments-dir environment_files \
  --run-dir runs/scorecard-ar25 \
  ... \
  --run-dir runs/scorecard-wa30
```

This consumes no API scorecard or API environment attempt.

List all 25 run directories and omit `--confirm-replay`:

```bash
./leaderboard-submission/leaderboard.sh gather \
  --source-url https://github.com/<owner>/<public-repository> \
  --output runs/official-scorecard.json \
  --run-dir runs/scorecard-ar25 \
  --run-dir runs/scorecard-bp35 \
  --run-dir runs/scorecard-cd82 \
  ... \
  --run-dir runs/scorecard-wa30
```

Preflight contacts only environment discovery. It does **not** call
`create_scorecard()` or `make()`. It returns JSON reporting missing, extra,
duplicate, and version-mismatched trajectories. Continue only when it prints
`"ready": true`; `--confirm-replay` turns any such problem into a hard failure
before opening the scorecard.

Randomness is handled strictly: before the first replayed action, and after
every action, the API observation must equal the recorded Timeline observation
(grid, legal actions, state, levels completed, and win target). Any difference
aborts immediately. Exact local dry replay is already enforced by Physics when
recovering a lost process, but only the final API replay can prove that the
remote initial state agrees.

### Optional online rehearsal

After local validation and complete preflight, the safest way to test remote
random state and API/source equivalence is a full **ordinary Online Mode**
rehearsal. It creates a disposable non-Competition scorecard and therefore does
not consume Competition Mode's one scorecard/one-`make` restrictions:

```bash
./leaderboard-submission/leaderboard.sh rehearse \
  --source-url https://github.com/<owner>/<public-repository> \
  --output runs/scorecard-rehearsal.json \
  --run-dir runs/scorecard-ar25 \
  ... \
  --run-dir runs/scorecard-wa30
```

It performs the same exact observation checks and saves traces. Run rehearsal
close to the final submission, then rerun safe preflight because the advertised
suite may change. A successful rehearsal greatly reduces random-state risk but
cannot guarantee that a later Competition session starts identically.

## 6. One-shot Competition Mode replay

This consumes the sole scorecard and each environment's one permitted `make()`.
Run it only after successful preflight:

```bash
./leaderboard-submission/leaderboard.sh gather \
  --source-url https://github.com/<owner>/<public-repository> \
  --tag physics-strategy \
  --output runs/official-scorecard.json \
  --recordings-dir recordings/official-scorecard \
  --run-dir runs/scorecard-ar25 \
  --run-dir runs/scorecard-bp35 \
  ... \
  --run-dir runs/scorecard-wa30 \
  --confirm-replay
```

The operation:

1. rediscovers the exact current 25-version suite;
2. validates complete exact-version coverage before opening anything;
3. opens one `OperationMode.COMPETITION` scorecard;
4. calls `make()` once per environment;
5. submits every recorded action, preserving ACTION6 payloads;
6. compares every returned public observation to the immutable Timeline;
7. closes the scorecard and atomically saves the returned JSON plus:
   `https://arcprize.org/scorecards/<card-id>`.

The toolkit also writes local JSONL action/frame traces under the selected
`--recordings-dir`, grouped by scorecard ID. Preserve them with the hosted
replays and public solver code. A sibling `*.progress.json` durably records the
card ID/URL and each fully replayed environment as the one-shot operation runs.

At the documented 600 requests/minute limit, replay time is governed mostly by
total actions, not the original 36-hour inference time. Do not interrupt the
process. ARC notes that scorecards auto-close after inactivity and an abrupt
Ctrl-C can prevent result retrieval. If replay fails, a sibling
`*.failure.json` preserves the opened card ID/URL and error; the code attempts a
best-effort close, but a failed one-shot scorecard should not be submitted.

## 7. Community Leaderboard submission

Publish the complete general-purpose solver and disclose that the model solved
the exact versions locally and that the resulting authoritative trajectories
were subsequently submitted unchanged through Competition Mode. Do not present
per-game action files as the method. The Community Leaderboard requires the
hosted `scorecard_url` and derives the ARC-AGI-3 score from it instead of taking
a self-reported numeric score.

This hosted Community scorecard is not the same as an ARC Prize Verified or
hidden/Kaggle evaluation. Confirm deferred trajectory replay acceptability with
maintainers if their review policy changes.

## Official references

- <https://docs.arcprize.org/toolkit/competition_mode>
- <https://docs.arcprize.org/scorecards>
- <https://docs.arcprize.org/methodology>
- <https://docs.arcprize.org/rate_limits>
- <https://github.com/arcprize/ARC-AGI-Community-Leaderboard>

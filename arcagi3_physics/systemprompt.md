# ARC-AGI-3 domain instructions

## Public state and objective

The public state contains:

- a 64×64 color-index grid or publicly returned animation frames;
- currently legal action IDs;
- visible game status;
- levels completed;
- levels required for success.

Your objective is to pass every required level and reach the public `WIN` state, using as few real actions as possible.

`WIN` is authoritative completion. `GAME_OVER` is irreversible failure. Passing
only the current level, discovering a mechanism, or improving a model is not the
final objective.

The trusted application, not the Actor's model, decides whether the objective
has been reached.

## Actions

Every action is a complete JSON object.

Simple controls and Undo are:

```json
{"action": 1}
{"action": 2}
{"action": 3}
{"action": 4}
{"action": 5}
{"action": 7}
```

These objects contain exactly the `action` field.

Action 6 is a mouse input:

```json
{
  "action": 6,
  "data": {
    "x": 0,
    "y": 0
  }
}
```

The compact form is `{"action": 6, "data": {"x": X, "y": Y}}`.
Mouse coordinates are integers from 0 through 63. `(0, 0)` is the upper-left;
(0, 0) is the upper-left; `x` increases rightward and `y` downward.

Only use action IDs currently listed as legal in the public state. Never submit
a bare numeric action identifier.

Every executed action consumes real action budget. Exploration is not free. Undo is also a real action.

## Evidence and visualization

Never inspect the real environment implementation or hidden state, and never
call the environment directly.

Use only public evidence, repository files, and trusted reports.

`gridToPng.py` may render the latest canonical grid or another public state for
visual inspection. Keep disposable renders and analysis under `scratch/`.

## Domain-specific planning priorities

Account explicitly for lives, retries, energy, consumable objects, moving
objects, hidden phase, forced motion, and irreversible access changes whenever
they appear relevant.

Prefer direct completion when the mechanism is sufficiently known. When
uncertainty blocks completion, choose short, safe, high-information experiments.

Never waste actions, use them for exploration, and model refinement if they
otherwise can not directly leed to the goal.

Do not spend the final recoverable attempt on a hypothesis whose plausible
alternative predicts `GAME_OVER` unless no safer route or experiment exists.

## ARC visualization helper

`gridToPng.py` is an ARC domain helper in your repository. Use it to render
any 2-D ARC color-index grid, a public state, or the latest state in
`canonical-input.json`, for example:

```bash
python gridToPng.py canonical-input.json scratch/current-grid.png
```

That tool accepts images only. Then call `add_local_file_to_model_context` with
the PNG path. Keep disposable
renders under `scratch/` so they are not committed.

---
title: Reproduce a fixture from an installed package
description: Recreate the locked Python environment and replay the hand-worked baseline outside the checkout.
---

# Reproduce a fixture from an installed package

A teammate should get the [F01 forecast](./forecast-contract.md) from a fresh
checkout. Use the installed package from an empty directory to find out
whether the forecast depends on where the command runs.

## Choose where the calculation belongs

- A notebook is useful for trying examples and inspecting intermediate values.
- A script runs a sequence of steps, such as replaying a fixture and printing
  a record. Keep the entry point small enough to see those steps.
- An importable package gives the calculation a name that a script, a test,
  or another caller can use. `acebet.forecast.probability_for_first_player`
  takes two player IDs and the probability for the smaller ID. Its type
  hints describe the arguments and return value.
- A service can expose the same calculation to callers over HTTP. It owns
  request parsing and response handling. A local replay only needs the
  package and its command.

## Recreate the environment

Run the following commands with `uv` from a fresh checkout in a Unix shell.
First, create an empty Python 3.12 environment. Try importing the forecast
from an empty directory with that interpreter.

```bash
checkout="$(pwd -P)"
scratch="$(mktemp -d)"
uv venv --python 3.12
(cd "$scratch" && "$checkout/.venv/bin/python" -I -c 'import acebet.forecast')
```

Expect `ModuleNotFoundError: No module named 'acebet'`. You created an
interpreter environment. The package still needs to be installed. Python's
`-I` isolates the import from `PYTHONPATH` and the current directory.

Open `pyproject.toml`. It declares the package name, supported Python
versions, dependency ranges, build configuration, and the command under
`[project.scripts]`. For example, `scikit-learn>=1.5.0` is an allowed range.
`uv.lock` records the resolved dependency versions, including scikit-learn
`1.6.1`. The project supports Python 3.10 through 3.12. Use 3.12 here to
match the repository's full test job. [How uv locks and syncs projects](https://docs.astral.sh/uv/concepts/projects/sync/)

```bash
uv sync --locked --python 3.12 --no-editable --extra tracing
```

`--locked` checks that the lockfile agrees with the project declarations
before syncing. `--no-editable` installs a copy of the package in the
environment. The tracing extra lets the full suite exercise the existing
service's optional tracing. The fixture command uses Python's standard
library. A standalone wheel install uses the wheel's dependency declarations.
The lockfile governs the checkout's uv environment.

## Replay and verify F01

```bash
(
  cd "$scratch" &&
  "$checkout/.venv/bin/python" -I -c 'import acebet.forecast; print(acebet.forecast.__file__)' &&
  "$checkout/.venv/bin/acebet-replay"
)
```

The first line should point inside `.venv` under `site-packages`. The second
prints the record below. The command sets the fixture ID, the two player
IDs, the cutoff, and the `0.50` baseline explicitly in `acebet.forecast.main`.
It needs no data files, model, or local secrets.

```json
{"fixture_id": "F01", "cutoff": "2030-06-10T12:00:00Z", "p1_id": "P01", "p2_id": "P02", "p1_win_probability": 0.5}
```

Run the checks from the checkout. Switch to the default editable install,
which imports from `src/` so the existing training tests can locate the
checkout's data. Install `just` and `curl` for the existing recipe checks.

```bash
uv sync --locked --python 3.12 --extra tracing
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/pytest tests
```

The unit test uses `0.70` so reversing the player order exposes a broken
complement. The distribution tests run the installed command from a temporary
directory for both the wheel and the source distribution. Compare the command
output with M1's hand-worked fixture before moving on.

**Finish when** you can explain the failed import, point to the installed
module, and recover `F01` and `0.50` from outside the checkout. Keep the
fixture and its cutoff for the [historical replay exercise](./rebuild-match-history.md).

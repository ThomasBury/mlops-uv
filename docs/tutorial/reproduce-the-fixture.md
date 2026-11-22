# 2. Put the forecast in a Python package

The reader can name Robin first. A coin-flip forecast would hide a player-order bug because its complement is also 0.50. Try 0.70 for Alex. What should a caller see for Robin?

In `src/acebet/forecast.py`, the **module** (one Python file) defines `probability_for_first_player`. Its two `str` arguments identify the players. Its `float` argument holds the smaller ID’s probability. The return annotation `-> float` tells readers and type checkers what to expect. The function checks empty or identical IDs and the probability range at runtime, then converts the probability to the caller’s player order.

```python
if first_player_id < second_player_id:
    return canonical_probability
return 1.0 - canonical_probability
```

The `acebet` **package** gives the module an import name. The `[project.scripts]` entry in `pyproject.toml` points `acebet-replay` to `acebet.forecast:main`. Installing the project makes that command available to a caller in another directory.

## Make the import work

Use Git to get a fresh checkout. The examples use uv and Python 3.12.

```bash
git clone https://github.com/ThomasBury/mlops-uv.git
cd mlops-uv
```

Start at the checkout root. Use Bash for the course commands. Keep this shell, the checkout and one scratch directory throughout. If you open another terminal, set `work` to the absolute path printed here and return to the checkout root.

```bash
checkout="$(pwd -P)"
work="$(mktemp -d)"
echo "$work"
uv venv --python 3.12
(cd "$work" && "$checkout/.venv/bin/python" -I -c 'import acebet.forecast')
```

Expect `ModuleNotFoundError`. A virtual environment gives the project its own interpreter entry point and installed packages. The empty environment has no AceBet installation. Isolated mode (`-I`) keeps the current directory and `PYTHONPATH` out of that import.

`pyproject.toml` declares acceptable dependency versions. Resolution chooses versions satisfying those ranges together. `uv.lock` records that choice across the supported platforms. Sync checks the declarations against the lock and installs the applicable versions. A non-editable installation copies the package into the environment. [uv project syncing](https://docs.astral.sh/uv/concepts/projects/sync/)

```bash
uv sync --locked --python 3.12 --no-editable --group mlops --extra tracing
(cd "$work" && "$checkout/.venv/bin/python" -I -c 'import acebet.forecast; print(acebet.forecast.__file__)')
(cd "$work" && "$checkout/.venv/bin/acebet-replay")
```

The path should point into `site-packages`. The command prints F01, a noon UTC cutoff, `P01` first, and 0.50. That **installation test** would catch a missing script entry even if the function itself were correct.

## Challenge the ordering rule

Switch to an editable install so changes to the checkout reach the interpreter. Run this **unit test** of the calculation, then open `tests/test_forecast.py` and locate its order-reversal assertion.

```bash
uv sync --locked --python 3.12 --group mlops --extra tracing
.venv/bin/python - <<'PY'
from math import isclose
from acebet.forecast import probability_for_first_player

assert isclose(probability_for_first_player("P02", "P01", 0.70), 0.30)
PY
.venv/bin/ruff check src/acebet/forecast.py
.venv/bin/pytest -q tests/test_forecast.py
```

Ruff checks source conventions and flags mistakes such as unused imports. To see what the behavioral check contributes, temporarily return `canonical_probability` in the last line of the function. Predict which check will fail, run both, then restore `1.0 - canonical_probability` and rerun. Keep that repair in place for the [dataset rebuild](./rebuild-match-history.md).

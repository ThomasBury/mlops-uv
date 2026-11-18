---
title: Recreate the working environment
description: Use the locked checkout and inspect the recipe behind the first data check.
---

# 1. Recreate the working environment

A teammate needs the same tools and dependencies to reproduce your work.
Start by checking that this checkout can read the prepared match data. These
are historical match records already arranged in the format AceBet expects.

```bash
UV_PYTHON=3.12 just data-check
```

If setup succeeds, the last line reports 44,617 validated rows from
`data/atp_data_production.feather`. If the command stops before data validation,
read the first error to see whether it names a missing tool, a Python version,
or a dependency.

Open `justfile` and find the recipe you ran.

```just
data-check: setup
    uv run python -m acebet.dataprep.dataprep
```

A **recipe** is a named command teammates can run the same way. AceBet
declares its dependencies in `pyproject.toml`. For example,
`scikit-learn>=1.5.0` allows version 1.5.0 or later. The lockfile, `uv.lock`,
records the selected version, `1.6.1`, along with versions of the packages
that scikit-learn and other dependencies need.

The `setup` prerequisite runs `uv sync --locked`. It installs the locked
versions that apply to your project environment and fails if the lockfile
needs an update. Then `uv run` uses that environment to run the data check.
See [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
for how the lockfile works and
[Developer Workflow](../uv/uv-workflows-for-mlops.md) for troubleshooting setup.

**Finish when** you can point to the check's output and explain which command
installed the dependencies and which validated the data. Keep this checkout
for [lesson 2](./01-notebook-to-data-contract.md).

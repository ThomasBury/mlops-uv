---
title: Hand the application to another Python user
description: Build a wheel and source distribution, then check the installed application and its bundled default outside the checkout.
---

# 8. Hand the application to another Python user

Your colleague needs a way to install the application without a source
checkout. A **wheel** is a ready-to-install Python package. A **source
distribution** contains the source and build information needed to build
the package.

In terminal B from the repository root, build both into a temporary
directory. Install the wheel into a fresh Python environment there.
The parentheses run these commands in a shell that cleans up its temporary
directory when it exits.

```bash
(
  set -e
  wheel_check="$(mktemp -d)"
  trap 'rm -rf "$wheel_check"' EXIT
  uv build --out-dir "$wheel_check/dist"
  uv venv --python 3.12 "$wheel_check/.venv"
  cd "$wheel_check"
  uv pip install --python .venv/bin/python dist/acebet-*.whl

  ACEBET_ENV=development ACEBET_ALLOW_INSECURE_DEV_SECRET=true .venv/bin/python - <<'PY'
from fastapi.testclient import TestClient

from acebet.app.main import app

with TestClient(app) as client:
    assert client.get("/readyz").json() == {"status": "ready"}
    print("Installed default:", app.state.runtime_assets.production.model_identity.version)
PY
)
```

The build produces a `.whl` and a `.tar.gz` in the temporary `dist/` directory.
`uv build` calls Setuptools, as specified in `pyproject.toml`. Setuptools
includes the default model and player snapshots in the wheel and writes
the declared runtime requirements into the wheel's metadata. See the
[uv build guide](https://docs.astral.sh/uv/concepts/projects/build/) and the
[Python packaging specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/#dependencies).

The check starts the application in the test client, which sends a request
without opening a network port. You should see the packaged model version
from the [example response](./index.md#tutorial).

When `uv pip install` installs the wheel, it uses the wheel's
`scikit-learn>=1.5.0` requirement. It may select a different version from the
`1.6.1` recorded in `uv.lock` and installed in your
[lesson 1 checkout](./00-recreate-the-working-environment.md). The Docker
build uses `uv.lock` for its runtime dependencies.

**Finish when** you can explain what the wheel gives your colleague and
what they would have to provide to use your promoted bundle. Why can
the scikit-learn version installed from the wheel's requirements differ from
the one in your checkout? The next lesson passes your bundle and player
snapshots to the container explicitly.
Continue with [lesson 9](./09-reproduce-the-service-runtime.md).

---
title: Reproduce the service runtime
description: Run AceBet in Docker with the same model and player snapshots and compare its response with the local one.
---

# 9. Reproduce the service runtime

Docker gives your colleague a runnable image that includes Python and system
libraries as well as the installed package. In `Dockerfile`, the builder
installs the locked dependencies and AceBet. The runtime stage uses the same
Python base, installs the system library `libgomp1`, and copies the installed
Python environment from the builder.

## Run your bundle in the container

With the local server stopped, use terminal A from the repository root.

```bash
just docker-build
docker run --rm --env-file .env -p 127.0.0.1:8000:8000 \
  --env ACEBET_MODEL_DIR=/assets/models \
  --env ACEBET_DATA_DIR=/assets/data \
  --mount "type=bind,src=$PWD/var/tutorial/models,dst=/assets/models,readonly" \
  --mount "type=bind,src=$PWD/var/tutorial/data,dst=/assets/data,readonly" \
  acebet
```

A **bind mount** makes a host directory visible inside the container.
`readonly` lets the process read your files while protecting them from
container writes. The two settings tell AceBet which paths to open inside
the container. The port mapping exposes the container service at the same
loopback address you used earlier.

## Compare the model identities

In terminal B, wait for `just readyz` to return `{"status":"ready"}`. Then
repeat the saved request with a token issued by this instance.

```bash
(
  set -e
  response=var/tutorial/container-response.json
  rm -f "$response"
  pending="$(mktemp "$response.XXXXXX")"
  trap 'rm -f "$pending"' EXIT
  token="$(just --quiet token)"
  printf 'Authorization: Bearer %s\n' "$token" |
    curl -fsS "$api/predict/production" \
      -H @- \
      -H 'Content-Type: application/json' \
      --data-binary @var/tutorial/request.json \
      -o "$pending"
  mv "$pending" "$response"

  uv run python - <<'PY'
import json
from pathlib import Path

root = Path("var/tutorial")
local = json.loads((root / "local-response.json").read_text())
container = json.loads((root / "container-response.json").read_text())
assert container["model"] == local["model"]
print("Both processes served:", container["model"]["version"])
PY
)
```

If login or the request fails, fix the reported error and rerun the block.

**Finish when** you can show that the version and checksum agree and explain
how the mounted bundle reached the container. Keep the container running in
terminal A for [lesson 10](./07-validation-ci-deployment-and-boundaries.md).
For other port mappings or runtime settings, see the
[Docker reference](../deployment/docker-and-github-actions.md#configuration-and-assets).

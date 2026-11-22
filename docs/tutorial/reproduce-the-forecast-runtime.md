# 8. Reproduce the service on another machine

Suppose a colleague has the same request and model file. They also need compatible inference libraries, the code that constructs the features and the exact history used by that model. A **wheel** supplies the installed Python package. The **lockfile** selects dependency versions for the platform. The **runtime image** adds the interpreter, operating-system libraries, non-root user, startup command and serving assets.

In the Dockerfile, `COPY --from=forecast_assets` takes three files from a named build context. The model supplies fitted coefficients **and** preprocessing, `selection.json` binds that model to its inputs, and Parquet supplies the results used to reconstruct the state at the cutoff.

## Build from the bundle you trained

Run the following with Docker Engine and its Buildx plugin. On Linux you can install Engine directly. The project root provides the Dockerfile and source. The second directory provides your assets. [Docker Engine installation](https://docs.docker.com/engine/install/) · [Buildx named contexts](https://docs.docker.com/reference/cli/docker/buildx/build/#build-context)

```bash
mkdir "$work/image-assets"
install -m 0644 "$work/matches.parquet" "$work/first/selection.json" \
  "$work/first/model.joblib" "$work/image-assets/"
docker buildx build --target historical-forecast --build-context forecast_assets="$work/image-assets" \
  --load -t acebet:course-replay .
engine=docker
```

**Native Podman alternative.** Prepare the same asset directory. Replace the Docker build and `engine=docker` above with the following. Docker image format preserves the Dockerfile health check. Podman can run daemonless on Linux. On macOS and Windows Podman uses a Linux VM. [Podman build options](https://docs.podman.io/en/latest/markdown/podman-build.1.html) · [Podman machines](https://docs.podman.io/en/latest/markdown/podman-machine.1.html)

```bash
podman build --format docker --file Dockerfile --target historical-forecast \
  --build-context forecast_assets="$work/image-assets" -t acebet:course-replay .
engine=podman
```

## Verify the running image

Run the resulting image by ID. Inspect the actual user and mounts, then send the same requests that you saved in lesson 7. The example publishes only on the loopback address and supplies no asset mounts.

```bash
image_id="$("$engine" image inspect acebet:course-replay --format '{{.Id}}')"
echo "$image_id"
container="$("$engine" run --rm -d -p 127.0.0.1:8011:8000 "$image_id")"
"$engine" exec "$container" id -u
"$engine" inspect "$container" --format '{{len .Mounts}}'
curl -fsS --retry 12 --retry-all-errors --retry-delay 1 http://127.0.0.1:8011/readyz
for id in F01 424816; do
  curl -fsS -H 'Content-Type: application/json' --data-binary @"$work/$id-request.json" \
    http://127.0.0.1:8011/forecast > "$work/$id-image.json"
done
.venv/bin/python "$work/compare-replies.py" http image
"$engine" stop "$container"
```

Expect a nonzero UID, zero mounts, a ready service and two matching replies. Checking F01 alone would exercise synthetic history. The archived match also checks the feature state reconstructed from the bundled results. If readiness fails, read the container logs and compare the selected history and model digests before rebuilding. Next, [investigate a served forecast](./diagnose-and-roll-back-a-forecast.md).

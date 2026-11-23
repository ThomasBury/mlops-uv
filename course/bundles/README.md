# Course prerequisite archives

- **M5** contains canonical `matches.parquet`, its preparation report and the evaluated `bundle/` from M4c. It supplies both the fitted pipeline and the archived batch forecast used in the HTTP and container comparisons.
- **M9a** contains the original archive run, the delayed-history incumbent run, both tracking databases and artifact stores, the registry database, and the staged incumbent. Open it with [the M9a setup](../m9a.md#start-with-two-identifiable-pairs) to bind its artifact URIs to your scratch directory.

Both archives derive from the pinned 2021 extract. The incumbent is a teaching variation. The Parquet metadata records the source, attribution and canonical content digest. [Data provenance and preparation](../m1.md)

Verify the downloads from this directory with `sha256sum -c SHA256SUMS` on Linux or `shasum -a 256 -c SHA256SUMS` on macOS.

To rebuild the teaching assets from the repository root, use a new output directory. Training creates new MLflow IDs. Compare separate runs using [M4c's numerical tolerances](../m4c.md).

```bash
output="$(mktemp -d)/bundles"
uv run --locked --python 3.12 --group mlops --extra tracing python \
    course/build_prerequisites.py "$output"
```

# Modern MLOps and ML Engineering

![AceBet tennis court and ball](https://raw.githubusercontent.com/ThomasBury/mlops-uv/main/docs/images/acebet-mark.svg)

At noon, what could we have known about two players before their match? Start with a hand calculation. Rebuild a named 2021 tennis dataset, train a probability model, compare it with Elo on later fixtures, and serve the same forecast offline, over HTTP and in a container. Use delayed results to investigate quality and decide whether to release the candidate.

[Start the course](course/README.md) for the milestone index, practice checkpoints, and ready-to-use prerequisites for the serving and registry labs.

[Hosted course overview](https://mlops-uv-tutorial.bury-thomas.workers.dev/) · [Hosted lessons and verification exercises](https://mlops-uv-tutorial.bury-thomas.workers.dev/tutorial/)

The first lesson needs only a pencil. Use a fresh checkout for the practical work.

```bash
git clone https://github.com/ThomasBury/mlops-uv.git
cd mlops-uv
```

The course uses Python 3.12, uv and a Unix-style shell. The runtime lesson uses Docker Engine with Buildx or native Podman on Linux. The final lesson introduces just and the existing CI workflow.

**Looking for the packaged predictor?** AceBet also ships the Elo model for the Fognini–Jarry 2018 replay. Run its authenticated API and compare the model fingerprint with the committed promotion record. [Packaged quickstart](https://mlops-uv-tutorial.bury-thomas.workers.dev/getting-started/)

Licensed under Apache 2.0.

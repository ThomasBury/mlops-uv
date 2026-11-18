---
title: Start Here
description: Begin the model-validity-first AceBet tutorial and distinguish tutorial, guide, and reference material.
hide:
  - toc
---

# Start Here { .brand-title }

AceBet is a compact classical MLOps tutorial. The tennis predictor is only a
vehicle: it is not a betting product or a claim of production readiness.

## The Mental Model

Start with prepared data, build features, compare models over time, select a
model bundle, serve it through a validated API, and run operational checks.
Each step makes one decision inspectable before the next one depends on it.

## Choose A Path

- [Tutorial](./tutorial/index.md): ten lessons from training to diagnosing a running prediction service.
- [Getting Started](./getting-started.md): clean-clone setup and local commands.
- [Companion Guides](./guides/index.md): training, artifacts, and tooling.
- [Reference](./reference/index.md): exact API, CI, and Docker behavior.

The canonical training boundary is the committed 44,617-row prepared Feather
file. The 100-row raw sample and starter notebook demonstrate transformations
and in-memory comparison only; they cannot reproduce the production dataset.

The canonical comparison selects `elo_baseline` because the LightGBM
challenger does not beat Elo by the required log-loss margin. The API then
serves that checksum-validated promoted bundle and reports exact probability,
model identity, and snapshot freshness.

`just review` validates the non-Docker loop without modifying tracked files.
`just docker-smoke` separately proves the non-root authenticated container
runtime. Use [Tutorial Boundaries](./tutorial-shortcuts-vs-production.md) for
the single implemented-versus-missing matrix.

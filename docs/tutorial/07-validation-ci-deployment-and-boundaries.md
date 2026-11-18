---
title: Decide what to verify before release
description: Diagnose a possible model mix-up and connect your investigation to the repository's release checks.
---

# 10. Decide what to verify before release { #tutorial-7-validation-and-release-boundaries }

<a id="why-this-exists"></a>

Your colleague says the container is ready and returns a plausible
prediction, but suspects it loaded the packaged default. Decide what
evidence would settle the question, then check it against the running
container or the response you saved in lesson 9.

Write a short reply to your colleague. Include the observation that settles
the question, the configuration you would inspect if the suspicion were
correct, and what a passing readiness check establishes. If Docker is
unavailable, use your local prediction, the documented packaged response,
and the container command from lesson 9 to explain how you would check.

??? note "Check your diagnosis after choosing a check"

    Compare the prediction's model version and full checksum with the values
    in your promotion record. The version is the promoted model filename
    without its extension. Compare against the packaged default's identity
    if the values differ. A plausible probability or a matching model kind
    can come from either bundle.

    If the container returned the packaged identity, inspect whether
    `ACEBET_MODEL_DIR` points to the mounted directory and whether the mount
    source contains the intended `promoted_model.json`. With the setting
    unset, the app uses its packaged default. A configured directory with
    missing required files fails startup. `/readyz` says the service loaded
    the assets required by its current configuration. Either valid bundle can
    satisfy that check.

## Connect your diagnosis to automated checks { #what-contract-tests-prevent }

<a id="run-the-repository-gate"></a>
<a id="docker-is-a-separate-runtime-proof"></a>
<a id="experiment-override-the-published-port"></a>

Maintainers run `just review` for data checks, candidate comparison, tests
with coverage, distribution installs, code checks, the notebook, docs, and
an authenticated local API smoke test. The tutorial contracts compare the
documented default response and selection evidence with the real packaged
bundle. These checks would catch a change in the documented default.

`just docker-smoke` builds the image and exercises readiness, authentication,
and prediction with packaged assets and with read-only mounts. Your
diagnosis checks which specific export answered on this invocation. CI,
short for continuous integration, repeats the Python checks and runs the
container smoke in a separate job. The docs job builds and publishes the
tutorial site from the main branch. See the
[CI/CD reference](../cicd/github-actions-pipeline.md) and
[Docker reference](../deployment/docker-and-github-actions.md)
for the full checks and port configuration.

**Finish when** your reply names the evidence you used, the model that
answered, and where you would look for a configuration error. Then stop the
container with **Ctrl-C in terminal A**. The files under `var/tutorial/`
are yours to keep or discard.

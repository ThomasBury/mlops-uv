from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import timedelta
import json
from math import log
from pathlib import Path
from shutil import copytree
import sys
from threading import Event
import time

from fastapi.testclient import TestClient
import pandas as pd
from prometheus_client.parser import text_string_to_metric_families
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError
import pytest

from acebet.app import observability
from acebet.app.historical import create_app
from acebet.dataprep.historical_replay import (
    DEVELOPMENT_END,
    F01_SURFACE,
    prior_day_cutoff,
)
from acebet.dataprep.historical_results import (
    build_results,
    content_sha256,
    read_results,
)
from acebet.forecast import F01_CUTOFF, F01_ID, F01_PLAYERS
from acebet.historical_forecast import (
    ForecastRequest,
    ForecastResponse,
    load_forecaster,
)
from acebet.historical_monitoring import (
    main as monitoring_main,
    promotion_check,
    quality_report,
    read_forecasts,
)
from acebet.model_artifacts import read_json, sha256_file, write_json
from acebet.train.historical_evaluation import evaluate_candidate, train_candidate

F01 = {
    "fixture_id": F01_ID,
    "p1_id": F01_PLAYERS[0],
    "p2_id": F01_PLAYERS[1],
    "surface": F01_SURFACE,
    "cutoff": F01_CUTOFF,
}


@pytest.fixture(scope="module")
def frozen_assets(tmp_path_factory):
    mlflow = pytest.importorskip("mlflow")
    work = tmp_path_factory.mktemp("frozen-forecast")
    path = work / "matches.parquet"
    snapshot = (
        Path(__file__).resolve().parents[1] / "data/raw/valuebetennis-2021-atp.csv"
    )
    build_results(snapshot, path)
    results = read_results(path)
    bundle = work / "selection"
    previous_uri = mlflow.get_tracking_uri()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MLFLOW_TRACKING_URI", previous_uri)
        try:
            selection = train_candidate(results, bundle)
            evaluate_candidate(results, bundle)
        finally:
            mlflow.set_tracking_uri(previous_uri)
    return path, bundle, selection


@pytest.mark.parametrize(
    "cutoff",
    ["2030-06-10T12:00:00", 1907323200, "1907323200", "9999-12-31T23:59:59-01:00"],
)
def test_request_requires_representable_timestamp_with_offset(cutoff):
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate({**F01, "cutoff": cutoff})


@pytest.mark.api
def test_frozen_replay_matches_batch_and_concurrent_http(frozen_assets, monkeypatch):
    path, bundle, selection = frozen_assets
    monkeypatch.setenv("ACEBET_FORECAST_RESULTS", str(path))
    monkeypatch.setenv("ACEBET_FORECAST_BUNDLE", str(bundle))
    offline = load_forecaster(path, bundle).predict(ForecastRequest.model_validate(F01))
    assert offline.p1_win_probability == pytest.approx(
        selection["fixture"]["logistic_p1_win_probability"], abs=1e-8, rel=0
    )
    assert offline.model_sha256 == selection["artifacts_sha256"]["model.joblib"]
    assert offline.history_sha256 == selection["data_sha256"]

    batch = pd.read_csv(
        bundle / "final/forecasts.csv",
        dtype={"match_id": str, "p1_id": str, "p2_id": str},
    ).iloc[-1]
    historical = {
        "fixture_id": batch.match_id,
        "p1_id": batch.p1_id,
        "p2_id": batch.p2_id,
        "surface": batch.surface,
        "cutoff": batch.cutoff.replace(" ", "T"),
    }
    reversed_f01 = {**F01, "p1_id": F01["p2_id"], "p2_id": F01["p1_id"]}
    with TestClient(create_app()) as client:
        assert client.get("/healthz").json() == {"status": "alive"}
        assert client.get("/readyz").json() == {"status": "ready"}
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(
                pool.map(
                    lambda payload: client.post("/forecast", json=payload),
                    [F01, historical, reversed_f01] * 3,
                )
            )
        assert all(response.status_code == 200 for response in responses)
        forward, actual, reverse = [response.json() for response in responses[:3]]
        assert [response.json() for response in responses] == [
            forward,
            actual,
            reverse,
        ] * 3
        assert forward == offline.model_dump(mode="json")
        offset = {**F01, "cutoff": "2030-06-10T14:00:00+02:00"}
        assert client.post("/forecast", json=offset).json() == forward
        assert actual["p1_win_probability"] == pytest.approx(
            batch.logistic_tuned, abs=1e-8, rel=0
        )
        assert actual["elo_gap"] == pytest.approx(batch.elo_gap)
        assert actual["feature_state_sha256"] == offline.history_sha256
        assert reverse["p1_win_probability"] == 1.0 - offline.p1_win_probability
        assert reverse["feature_state_sha256"] == offline.feature_state_sha256
        for invalid in (
            {"p1_id": F01["p2_id"]},
            {"surface": "Ice"},
            {"cutoff": "2030-06-10T12:30:00Z"},
            {"fixture_id": "unknown"},
            {"winner_id": F01["p1_id"]},
        ):
            assert client.post("/forecast", json={**F01, **invalid}).status_code == 422


@pytest.mark.api
@pytest.mark.parametrize("broken", ["model", "history", "contract"])
def test_bad_assets_keep_live_process_unready(
    frozen_assets, tmp_path, monkeypatch, broken
):
    path, bundle, selection = frozen_assets
    model_bytes = (bundle / "model.joblib").read_bytes()
    if broken == "model":
        model_bytes = b"damaged"
    elif broken == "history":
        selection = {**selection, "data_sha256": "0" * 64}
    else:
        selection = {
            **selection,
            "config": {**selection["config"], "features": ["surface"]},
        }
    (tmp_path / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    (tmp_path / "model.joblib").write_bytes(model_bytes)
    monkeypatch.setenv("ACEBET_FORECAST_RESULTS", str(path))
    monkeypatch.setenv("ACEBET_FORECAST_BUNDLE", str(tmp_path))
    with TestClient(create_app()) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        assert client.post("/forecast", json=F01).status_code == 503


def _match_request(match):
    return {
        "fixture_id": match.match_id,
        "p1_id": match.p1_id,
        "p2_id": match.p2_id,
        "surface": match.surface,
        "cutoff": prior_day_cutoff(match.event_date).isoformat(),
    }


def _scrape(client):
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for metric in text_string_to_metric_families(client.get("/metrics").text)
        for sample in metric.samples
    }


@pytest.mark.api
@pytest.mark.tracing
def test_monitoring_signals(frozen_assets, tmp_path, monkeypatch):
    exporter_module = pytest.importorskip(
        "opentelemetry.sdk.trace.export.in_memory_span_exporter"
    )
    exporter = exporter_module.InMemorySpanExporter()
    monkeypatch.setattr(observability, "build_span_exporter", lambda *_: exporter)
    path, bundle, _ = frozen_assets
    log_path = tmp_path / "forecast.jsonl"
    for name, value in {
        "ENV": "production",
        "DEMO_PASSWORD": "secret",
        "FORECAST_RESULTS": path,
        "FORECAST_BUNDLE": bundle,
        "LOG_FILE": log_path,
        "LOG_FORMAT": "json",
        "LOG_LEVEL": "INFO",
        "METRICS_ENABLED": "true",
        "TRACING_ENABLED": "true",
        "TRACE_EXPORTER": "console",
    }.items():
        monkeypatch.setenv(f"ACEBET_{name}", str(value))

    results = read_results(path)
    match = next(
        row
        for row in results
        if row.event_date > DEVELOPMENT_END and row.surface == "Hard"
    )
    request = _match_request(match)
    app = create_app()
    invalid_labels = (("method", "POST"), ("path", "/forecast"), ("status_code", "422"))
    invalid_count = ("acebet_http_requests_total", invalid_labels)
    gauge = ("acebet_forecasts_in_progress", ())
    latency_bucket = (
        "acebet_http_request_duration_seconds_bucket",
        (("le", "0.25"), ("method", "POST"), ("path", "/forecast")),
    )
    probability_count = ("acebet_forecast_p1_win_probability_count", ())
    hot_surface = ("acebet_forecast_surface_total", (("surface", "Hard"),))
    cold_surface = ("acebet_forecast_surface_total", (("surface", "Clay"),))
    with TestClient(app) as client:
        before = _scrape(client)
        invalid = client.post("/forecast", json={**F01, "surface": "Ice"})
        assert invalid.status_code == 422
        after = _scrape(client)
        assert after[invalid_count] - before.get(invalid_count, 0) == 1
        assert after[probability_count] == before[probability_count]
        print("Invalid payload: one extra 422, zero extra probabilities.")

        model = app.state.forecaster.model
        original = model.predict_proba
        entered, release = Event(), Event()

        def slow_inference(frame):
            entered.set()
            assert release.wait(5)
            return original(frame)

        with (
            monkeypatch.context() as injection,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            injection.setattr(model, "predict_proba", slow_inference)
            pending = pool.submit(client.post, "/forecast", json=F01)
            try:
                assert entered.wait(5)
                assert _scrape(client)[gauge] == 1
                time.sleep(0.3)
            finally:
                release.set()
            slow = pending.result(timeout=5)
        assert slow.status_code == 200
        after_slow = _scrape(client)
        assert after_slow[gauge] == 0
        assert after_slow[latency_bucket] == after[latency_bucket]
        assert after_slow[probability_count] == after[probability_count] + 1

        for _ in range(3):
            assert client.post("/forecast", json=request).status_code == 200
        assert (
            client.post(
                "/forecast",
                json={**request, "p1_id": match.p2_id, "p2_id": match.p1_id},
            ).status_code
            == 200
        )
        shifted = _scrape(client)
        assert shifted[hot_surface] == after_slow.get(hot_surface, 0) + 4
        assert shifted[cold_surface] == after_slow[cold_surface]
        assert shifted[probability_count] == after_slow[probability_count] + 4
        print("Shifted requests: four more Hard inputs, no extra Clay inputs.")
        if ("process_resident_memory_bytes", ()) in shifted:
            print(
                "Process RSS in bytes:",
                shifted[("process_resident_memory_bytes", ())],
                "CPU seconds:",
                shifted[("process_cpu_seconds_total", ())],
            )

        app.state.tracer_provider.force_flush()
        spans = exporter.get_finished_spans()

    log_lines = log_path.read_text().splitlines()
    entries = [json.loads(line) for line in log_lines]
    invalid_log = next(
        entry
        for entry in entries
        if entry.get("request_id") == invalid.headers["X-Request-ID"]
    )
    assert invalid_log["status_code"] == 422
    served = next(
        entry
        for entry in entries
        if entry["event"] == "forecast_served"
        and entry["request_id"] == slow.headers["X-Request-ID"]
    )
    trace_id = int(served["trace_id"], 16)
    related = [span for span in spans if span.context.trace_id == trace_id]
    lookup = next(span for span in related if span.name == "forecast.feature_lookup")
    inference = next(span for span in related if span.name == "forecast.inference")
    parent = next(
        span for span in related if span.context.span_id == lookup.parent.span_id
    )
    assert "POST" in parent.name and "/forecast" in parent.name
    assert inference.parent.span_id == lookup.parent.span_id
    inference_ms = (inference.end_time - inference.start_time) / 1e6
    assert inference_ms >= 300
    print(
        "Slow forecast:",
        served["forecast_id"],
        "trace:",
        served["trace_id"],
        f"lookup {(lookup.end_time - lookup.start_time) / 1e6:.2f} ms, inference {inference_ms:.2f} ms",
    )

    forecasts = read_forecasts(log_lines)
    assert len(forecasts) == 2
    reversed_only = read_forecasts(reversed(log_lines))
    assert reversed_only.keys() == forecasts.keys()
    early = quality_report(
        forecasts.values(), results, match.available_at - timedelta(seconds=1)
    )
    mature = quality_report(forecasts.values(), results, match.available_at)
    (pending,) = early["periods"]
    (scored,) = mature["periods"]
    assert pending["forecasts"] == scored["forecasts"] == 1
    assert pending["label_coverage"] == 0 and pending["log_loss"] is None
    assert scored["label_coverage"] == 1
    probability = next(
        row.p1_win_probability
        for row in forecasts.values()
        if row.fixture_id == match.match_id and row.p1_id == match.p1_id
    )
    expected_loss = -log(probability if match.label == 1 else 1 - probability)
    assert scored["log_loss"] == pytest.approx(expected_loss)
    assert sum(row["matches"] for row in scored["calibration"]) == 1
    print(
        "One fixture forecast across four requests:",
        pending["label_coverage"],
        "→",
        scored["label_coverage"],
        "label coverage.",
    )


@pytest.mark.api
def test_gate_and_rollback(frozen_assets, tmp_path, monkeypatch):
    path, bundle, _ = frozen_assets
    monkeypatch.setenv("ACEBET_FORECAST_RESULTS", str(path))
    monkeypatch.setenv("ACEBET_FORECAST_BUNDLE", str(bundle))
    results = read_results(path)
    replay = _match_request(
        next(row for row in results if row.event_date > DEVELOPMENT_END)
    )

    with TestClient(create_app()) as client:
        original = client.post("/forecast", json=replay).json()
        original_f01 = client.post("/forecast", json=F01).json()
        gate = promotion_check(bundle / "final", "prevalence", "logistic_tuned")
        assert not gate["accepted"] and gate["interval_95"][0] > 0
        assert client.post("/forecast", json=replay).json() == original
        print("Rejected prevalence score:", gate)

    # Use an alternate, compatible snapshot with a simulated late result.
    results[0] = replace(results[0], available_at=results[-1].available_at)
    alternate_path = tmp_path / "matches.parquet"
    schema = pq.read_schema(path)
    metadata = {**schema.metadata, b"content_sha256": content_sha256(results).encode()}
    pq.write_table(
        pa.Table.from_pylist(
            [asdict(row) for row in results], schema=schema.with_metadata(metadata)
        ),
        alternate_path,
    )
    alternate_bundle = tmp_path / "alternate"
    import mlflow

    previous_uri = mlflow.get_tracking_uri()
    try:
        train_candidate(read_results(alternate_path), alternate_bundle)
    finally:
        mlflow.set_tracking_uri(previous_uri)

    def deploy(results_path, model_bundle):
        monkeypatch.setenv("ACEBET_FORECAST_RESULTS", str(results_path))
        monkeypatch.setenv("ACEBET_FORECAST_BUNDLE", str(model_bundle))
        with TestClient(create_app()) as client:
            assert client.get("/readyz").status_code == 200
            response = client.post("/forecast", json=replay)
            response.raise_for_status()
            f01 = client.post("/forecast", json=F01)
            f01.raise_for_status()
            return ForecastResponse.model_validate(response.json()), f01.json()

    alternate, alternate_f01 = deploy(alternate_path, alternate_bundle)
    assert alternate.model_sha256 != original["model_sha256"]
    assert alternate.feature_state_sha256 != original["feature_state_sha256"]
    assert alternate_f01["model_sha256"] != original_f01["model_sha256"]

    monkeypatch.setenv("ACEBET_FORECAST_RESULTS", str(path))
    with TestClient(create_app()) as client:
        assert client.get("/readyz").status_code == 503

    restored, restored_f01 = deploy(path, bundle)
    assert restored.model_dump(mode="json") == original
    assert restored_f01 == original_f01
    print(
        "Replayed and restored F01 plus",
        replay["fixture_id"],
        "model",
        restored.model_sha256,
        "and feature state",
        restored.feature_state_sha256,
    )


def test_candidate_review_and_explicit_promotion(
    frozen_assets, tmp_path, monkeypatch, capsys
):
    results, frozen_bundle, _ = frozen_assets
    bundle = copytree(frozen_bundle, tmp_path / "candidate")
    record = bundle / "promotion.json"
    evaluation_path = bundle / "final/evaluation.json"
    forecasts_path = bundle / "final/forecasts.csv"
    evaluation = read_json(evaluation_path)

    def run(command):
        monkeypatch.setattr(
            sys, "argv", ["monitoring", command, str(results), str(bundle)]
        )
        monitoring_main()
        return json.loads(capsys.readouterr().out)

    review = run("review")
    assert review["gate"]["accepted"] is False
    assert review["gate"]["forecasts_sha256"] == sha256_file(forecasts_path)
    assert review["fixture"]["fixture_id"] == "F01"
    assert review["fixture"]["model_sha256"] == evaluation["model_sha256"]
    assert not record.exists()
    with pytest.raises(SystemExit) as rejected:
        run("promote")
    assert rejected.value.code == 1
    assert json.loads(capsys.readouterr().out) == review
    assert not record.exists()

    # Synthetic scores exercise acceptance without treating the archive as a win.
    forecasts = pd.read_csv(forecasts_path)
    forecasts["elo"] = 1 - forecasts["label"]
    forecasts.to_csv(forecasts_path, index=False)
    evaluation["forecasts_sha256"] = sha256_file(forecasts_path)
    write_json(evaluation_path, {**evaluation, "selection_sha256": "other"})
    with pytest.raises(ValueError, match="does not belong"):
        run("promote")
    assert not record.exists()

    write_json(evaluation_path, evaluation)
    assert run("review")["gate"]["accepted"] is True
    assert not record.exists()
    approved = run("promote")
    assert approved["gate"]["interval_95"][1] < 0
    assert approved["evaluation_sha256"] == sha256_file(evaluation_path)
    assert approved["approved_at"]
    assert read_json(record) == approved

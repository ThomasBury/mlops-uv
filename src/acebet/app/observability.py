"""Observability helpers for logs, metrics, traces, and request correlation."""

from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar
from importlib import import_module
import io
import logging
from logging.handlers import RotatingFileHandler
import re
import time
from types import ModuleType
from typing import Any, Protocol, cast
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from structlog.stdlib import ProcessorFormatter
from structlog.typing import Processor

from acebet.settings import AppSettings

HTTP_REQUESTS_TOTAL = Counter(
    "acebet_http_requests_total",
    "Total HTTP requests handled by the AceBet API.",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "acebet_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)
PREDICTION_REQUESTS_TOTAL = Counter(
    "acebet_prediction_requests_total",
    "Total prediction requests handled by the AceBet API.",
    ["mode"],
)
PREDICTION_DURATION_SECONDS = Histogram(
    "acebet_prediction_duration_seconds",
    "Prediction latency in seconds.",
    ["mode"],
)
PREDICTION_FAILURES_TOTAL = Counter(
    "acebet_prediction_failures_total",
    "Prediction failures handled by the AceBet API.",
    ["mode", "error_type"],
)
PREDICTING_WITH_SAMPLE_DATA_TOTAL = Counter(
    "acebet_predicting_with_sample_data_total",
    "Successful prediction responses served with packaged sample/testing data.",
)

LOGGER = structlog.get_logger("acebet.app")
EXCLUDED_METRIC_PATHS = {"/metrics"}
UNMATCHED_METRIC_PATH = "__unmatched__"
STANDARD_HTTP_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"}
)


class SpanContextProtocol(Protocol):
    """Trace identifiers exposed by an active span context."""

    is_valid: bool
    trace_id: int
    span_id: int


class SpanProtocol(Protocol):
    """Minimal OpenTelemetry span surface used by structured logging."""

    def get_span_context(self) -> SpanContextProtocol:
        """Return identifiers for this span."""
        ...


class TracerProtocol(Protocol):
    """Minimal tracer surface used around predictions."""

    def start_as_current_span(self, name: str) -> AbstractContextManager[Any]:
        """Start a span context manager."""
        ...


class TracerProviderProtocol(Protocol):
    """Lifecycle surface required from an OpenTelemetry tracer provider."""

    def get_tracer(self, name: str) -> TracerProtocol:
        """Return a tracer bound to a logical component."""
        ...

    def add_span_processor(self, processor: Any) -> None:
        """Register one exporter-backed span processor."""
        ...

    def force_flush(self) -> object:
        """Flush queued spans."""
        ...

    def shutdown(self) -> object:
        """Shut down tracing resources."""
        ...


_request_trace_api: ContextVar[ModuleType | None] = ContextVar(
    "request_trace_api", default=None
)


def _load_tracing_modules() -> tuple[ModuleType, ...]:
    """Import optional tracing modules only after tracing is enabled."""
    module_names = (
        "opentelemetry.trace",
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        "opentelemetry.instrumentation.fastapi",
        "opentelemetry.sdk.resources",
        "opentelemetry.sdk.trace",
        "opentelemetry.sdk.trace.export",
    )
    try:
        return tuple(import_module(name) for name in module_names)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Tracing requires the optional 'tracing' extra. Install it with "
            "'pip install acebet[tracing]' or 'uv sync --extra tracing'."
        ) from exc


def add_trace_context(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Attach trace identifiers when an OpenTelemetry span is active."""
    trace_api = _request_trace_api.get()
    if trace_api is None:
        return event_dict
    span = cast(SpanProtocol, trace_api.get_current_span())
    span_context = span.get_span_context()
    if span_context.is_valid:
        event_dict["trace_id"] = f"{span_context.trace_id:032x}"
        event_dict["span_id"] = f"{span_context.span_id:016x}"
    return event_dict


def get_log_renderer(settings: AppSettings) -> Processor:
    """Return the configured structlog renderer."""
    if settings.log_format == "json":
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer()


def configure_logging(
    settings: AppSettings,
    stream: io.TextIOBase | None = None,
) -> None:
    """Configure stdlib logging plus structlog processors."""
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        add_trace_context,
    ]

    structlog.configure(
        processors=[*shared_processors, ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(settings.log_level)

    if settings.log_file:
        handler = RotatingFileHandler(
            settings.log_file,
            maxBytes=10_000_000,
            backupCount=3,
            encoding="utf-8",
        )
    else:
        handler = logging.StreamHandler(stream)
    handler.setFormatter(
        ProcessorFormatter(
            processor=get_log_renderer(settings),
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                timestamper,
                add_trace_context,
            ],
        )
    )
    root_logger.addHandler(handler)
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(logger_name)
        if logger_name == "uvicorn.access" and not uv_logger.hasHandlers():
            continue
        uv_logger.handlers.clear()
        uv_logger.propagate = True


def validate_trace_exporter_settings(settings: AppSettings) -> None:
    """Validate exporter configuration before optional tracing imports."""
    if settings.trace_exporter == "otlp" and settings.otlp_endpoint is None:
        raise RuntimeError(
            "ACEBET_OTLP_ENDPOINT is required when ACEBET_TRACE_EXPORTER=otlp."
        )


def build_span_exporter(
    settings: AppSettings,
    otlp_module: ModuleType | None = None,
    sdk_export_module: ModuleType | None = None,
) -> Any:
    """Return an exporter loaded from the optional tracing extra."""
    validate_trace_exporter_settings(settings)
    if otlp_module is None or sdk_export_module is None:
        _, otlp_module, _, _, _, sdk_export_module = _load_tracing_modules()
    if settings.trace_exporter == "otlp":
        exporter_type = getattr(otlp_module, "OTLPSpanExporter")
        return exporter_type(endpoint=settings.otlp_endpoint)
    exporter_type = getattr(sdk_export_module, "ConsoleSpanExporter")
    return exporter_type()


def configure_tracing(
    app: FastAPI, settings: AppSettings
) -> TracerProviderProtocol | None:
    """Configure tracing resources owned exclusively by one application."""
    if not settings.tracing_enabled:
        app.state.tracer_provider = None
        app.state.tracer = None
        app.state.tracing_instrumentor = None
        app.state.tracing_instrumented = False
        app.state.trace_api = None
        return None

    validate_trace_exporter_settings(settings)
    (
        trace_api,
        otlp_module,
        instrumentation_module,
        resources_module,
        sdk_trace_module,
        sdk_export_module,
    ) = _load_tracing_modules()
    resource_type = getattr(resources_module, "Resource")
    provider_type = getattr(sdk_trace_module, "TracerProvider")
    processor_type = getattr(sdk_export_module, "BatchSpanProcessor")
    instrumentor = getattr(instrumentation_module, "FastAPIInstrumentor")
    provider = cast(
        TracerProviderProtocol,
        provider_type(
            resource=resource_type.create({"service.name": settings.service_name})
        ),
    )
    app.state.tracer_provider = provider
    app.state.tracer = provider.get_tracer("acebet.app")
    app.state.tracing_instrumentor = instrumentor
    provider.add_span_processor(
        processor_type(build_span_exporter(settings, otlp_module, sdk_export_module))
    )
    instrumentor.instrument_app(
        app,
        tracer_provider=provider,
        excluded_urls="metrics",
    )
    app.state.tracing_instrumented = True
    # Lifespan startup runs after Starlette has cached the initial middleware stack.
    app.middleware_stack = app.build_middleware_stack()
    app.state.trace_api = trace_api
    return provider


def prediction_span(request: Request, name: str) -> AbstractContextManager[Any]:
    """Return an active optional span or a core-runtime no-op context."""
    tracer = getattr(request.app.state, "tracer", None)
    if tracer is None:
        return nullcontext()
    return cast(TracerProtocol, tracer).start_as_current_span(name)


def shutdown_tracing(app: FastAPI) -> None:
    """Uninstrument, flush, shut down, and clear one app's tracing state."""
    provider = getattr(app.state, "tracer_provider", None)
    instrumentor = getattr(app.state, "tracing_instrumentor", None)
    try:
        if getattr(app.state, "tracing_instrumented", False) and instrumentor:
            instrumentor.uninstrument_app(app)
    finally:
        try:
            if provider is not None:
                provider.force_flush()
        finally:
            try:
                if provider is not None:
                    provider.shutdown()
            finally:
                app.state.tracer_provider = None
                app.state.tracer = None
                app.state.tracing_instrumentor = None
                app.state.tracing_instrumented = False
                app.state.trace_api = None


REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def get_request_id(request: Request) -> str:
    """Return a validated inbound request ID or generate a new one."""
    inbound = request.headers.get("X-Request-ID")
    if inbound is not None and REQUEST_ID_PATTERN.fullmatch(inbound):
        return inbound
    return str(uuid4())


def resolve_metric_path(request: Request) -> str:
    """Resolve a bounded route-template label for HTTP metrics."""
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return str(route.path)
    return UNMATCHED_METRIC_PATH


def should_record_http_metrics(path: str) -> bool:
    """Return whether the path should be counted in generic HTTP metrics."""
    return path not in EXCLUDED_METRIC_PATHS


def record_http_metrics(
    method: str, path: str, status_code: int, duration_seconds: float
) -> None:
    """Record HTTP request counters and duration histograms."""
    if not should_record_http_metrics(path):
        return
    method = method if method in STANDARD_HTTP_METHODS else "OTHER"
    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        path=path,
        status_code=str(status_code),
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(
        duration_seconds
    )


def record_prediction_metrics(
    mode: str,
    duration_seconds: float,
    error_type: str | None = None,
) -> None:
    """Record prediction metrics for success or failure."""
    PREDICTION_REQUESTS_TOTAL.labels(mode=mode).inc()
    PREDICTION_DURATION_SECONDS.labels(mode=mode).observe(duration_seconds)
    if error_type is not None:
        PREDICTION_FAILURES_TOTAL.labels(mode=mode, error_type=error_type).inc()


def record_sample_data_prediction() -> None:
    """Record a successful prediction served from sample/testing assets."""
    PREDICTING_WITH_SAMPLE_DATA_TOTAL.inc()


def render_metrics() -> tuple[bytes, str]:
    """Return the Prometheus metrics payload and content type."""
    return generate_latest(), CONTENT_TYPE_LATEST


def bind_request_context(
    request_id: str, method: str, path: str, trace_api: ModuleType | None
) -> None:
    """Bind common request context for logs."""
    structlog.contextvars.clear_contextvars()
    _request_trace_api.set(trace_api)
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=method,
        path=path,
    )


def clear_request_context() -> None:
    """Clear request-bound log context."""
    structlog.contextvars.clear_contextvars()
    _request_trace_api.set(None)


def get_request_logger() -> structlog.stdlib.BoundLogger:
    """Return the request-scoped application logger."""
    return LOGGER


def request_duration_seconds(start_time: float) -> float:
    """Return elapsed request duration in seconds."""
    return time.perf_counter() - start_time

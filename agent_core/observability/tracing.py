from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

_tracer: Tracer | None = None
_provider: TracerProvider | None = None


def configure_tracing(
    *,
    service_name: str = "agent-core",
    exporter: SpanExporter | None = None,
    enabled: bool = True,
    use_batch_processor: bool = True,
) -> None:
    global _provider, _tracer

    shutdown_tracing()

    if not enabled:
        _provider = None
        _tracer = trace.NoOpTracer()
        return

    resource = Resource.create({"service.name": service_name})
    _provider = TracerProvider(resource=resource)

    resolved_exporter = exporter or ConsoleSpanExporter()
    if use_batch_processor:
        processor = BatchSpanProcessor(resolved_exporter)
    else:
        processor = SimpleSpanProcessor(resolved_exporter)
    _provider.add_span_processor(processor)

    _tracer = _provider.get_tracer("agent_core")


def shutdown_tracing() -> None:
    global _provider, _tracer

    if _provider is not None:
        _provider.shutdown()
    _provider = None
    _tracer = None


def get_tracer() -> Tracer:
    global _tracer

    if _tracer is None:
        configure_tracing(enabled=False)
    assert _tracer is not None
    return _tracer


class SpanScope:
    """Manage an OpenTelemetry span with optional sync or async context manager use."""

    def __init__(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ):
        self._name = name
        self._attributes = {
            key: value
            for key, value in (attributes or {}).items()
            if value is not None
        }
        self._kind = kind
        self._span: Span | None = None
        self._cm: Any | None = None

    def __enter__(self) -> Span:
        tracer = get_tracer()
        self._cm = tracer.start_as_current_span(
            self._name,
            kind=self._kind,
            attributes=self._attributes,
        )
        self._span = self._cm.__enter__()
        return self._span

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        if exc_type is not None and self._span is not None:
            if exc_val is not None:
                self._span.record_exception(exc_val)
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            else:
                self._span.set_status(Status(StatusCode.ERROR))
        if self._cm is not None:
            self._cm.__exit__(exc_type, exc_val, exc_tb)
        return False

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)

    @property
    def trace_id(self) -> str | None:
        if self._span is None:
            return None
        return f"{self._span.get_span_context().trace_id:032x}"

    @property
    def span_id(self) -> str | None:
        if self._span is None:
            return None
        return f"{self._span.get_span_context().span_id:016x}"

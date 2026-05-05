from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from typing import Any

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
    reset_contextvars,
)

_LOG_CONTEXT_KEYS = {
    "session_id",
    "step",
    "llm_call_id",
    "tool_call_id",
}


def _add_otel_context(
    logger: Any,
    method_name: str,
    event_dict: structlog.typing.EventDict,
) -> structlog.typing.EventDict:
    del logger, method_name

    try:
        from opentelemetry import trace
    except ImportError:
        return event_dict

    span = trace.get_current_span()
    span_context = span.get_span_context()
    if not span_context.is_valid:
        return event_dict

    event_dict["trace_id"] = f"{span_context.trace_id:032x}"
    event_dict["span_id"] = f"{span_context.span_id:016x}"
    return event_dict


def configure_logging(
    *,
    level: str = "INFO",
    json_output: bool = False,
) -> None:
    resolved_level = getattr(logging, level.upper())
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=resolved_level,
        force=True,
    )

    processors: list[Any] = [
        merge_contextvars,
        structlog.processors.add_log_level,
        _add_otel_context,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(resolved_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger(name)


def clear_logging_context() -> None:
    clear_contextvars()


class LoggingContext:
    """Thin wrapper over structlog contextvars for explicit runtime scopes."""

    def __init__(self, **kwargs: Any):
        self._context = {
            key: value
            for key, value in kwargs.items()
            if key in _LOG_CONTEXT_KEYS and value is not None
        }
        self._tokens: Mapping[str, object] | None = None

    def __enter__(self) -> "LoggingContext":
        if self._context:
            self._tokens = bind_contextvars(**self._context)
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        if self._tokens is not None:
            reset_contextvars(**self._tokens)
            self._tokens = None

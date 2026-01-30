"""
Lightweight observability hooks for structured simulator objects.

Provides a global logger registry that dataclasses inside the simulator can
notify as they are instantiated. The registry is intentionally minimal to
avoid introducing heavy dependencies at the core of the simulation loop.
"""
from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Any, Mapping, Protocol


class StructuredLogger(Protocol):
    """Consumer interface for structured observability events."""

    def record(self, event: str, payload: Mapping[str, Any]) -> None:
        ...


class _HookRegistry:
    """Thread-safe registry for a single structured logger."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._logger: StructuredLogger | None = None

    def set_logger(self, logger: StructuredLogger | None) -> None:
        with self._lock:
            self._logger = logger

    def get_logger(self) -> StructuredLogger | None:
        with self._lock:
            return self._logger

    def emit(self, event: str, payload: Mapping[str, Any]) -> None:
        logger = self.get_logger()
        if logger is None:
            return
        try:
            logger.record(event, payload)
        except Exception:
            # Observability must never break the core simulation loop.
            # Downstream loggers are responsible for their own error handling.
            pass


_REGISTRY = _HookRegistry()


def set_global_logger(logger: StructuredLogger | None) -> None:
    """Register a global structured logger (or clear it with None)."""
    _REGISTRY.set_logger(logger)


def get_global_logger() -> StructuredLogger | None:
    """Return the currently registered global logger, if any."""
    return _REGISTRY.get_logger()


def emit_event(event: str, payload: Mapping[str, Any]) -> None:
    """Emit a structured event if a logger is registered."""
    _REGISTRY.emit(event, payload)


@contextmanager
def use_logger(logger: StructuredLogger | None):
    """Context manager that temporarily sets the global logger."""
    previous = get_global_logger()
    set_global_logger(logger)
    try:
        yield
    finally:
        set_global_logger(previous)

"""Observability utilities for the MOS simulator."""

from .hooks import (
    emit_event,
    get_global_logger,
    set_global_logger,
    use_logger,
)

__all__ = [
    "emit_event",
    "get_global_logger",
    "set_global_logger",
    "use_logger",
]

try:  # optional TensorBoard integration
    from .tensorboard import TensorBoardLogger, create_logger

    __all__.extend(["TensorBoardLogger", "create_logger"])
except Exception:  # pragma: no cover - optional dependency
    TensorBoardLogger = None  # type: ignore

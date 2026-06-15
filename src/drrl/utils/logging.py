"""Structured logging setup with a single configuration entry point.

We keep logging dependency-free (stdlib :mod:`logging`) but emit a consistent,
parse-friendly line format so runs are greppable. Experiment tracking is a
separate concern handled by a thin adapter (see :mod:`drrl.utils.tracking`)
so the codebase never hard-depends on a tracker.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def configure_logging(level: LogLevel = "INFO", *, force: bool = True) -> None:
    """Install a root stream handler with the canonical DR-RL line format.

    Args:
        level: Minimum level to emit.
        force: If ``True``, replace any pre-existing root handlers (idempotent
            re-configuration across re-entrant test runs / notebooks).
    """
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logging.basicConfig(
        level=getattr(logging, level),
        handlers=[handler],
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Prefer module-qualified names (``__name__``) so log lines carry their
    origin in the layered architecture.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        A standard library :class:`logging.Logger`.
    """
    return logging.getLogger(name)

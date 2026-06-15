"""Tests for logging setup and the tracking adapter."""

import logging

from drrl.utils.logging import configure_logging, get_logger
from drrl.utils.tracking import NullTracker, Tracker, get_tracker


def test_configure_logging_sets_level():
    configure_logging(level="WARNING")
    assert logging.getLogger().level == logging.WARNING
    # restore a sane default for other tests
    configure_logging(level="INFO")


def test_get_logger_returns_named_logger():
    log = get_logger("drrl.test")
    assert isinstance(log, logging.Logger)
    assert log.name == "drrl.test"


def test_null_tracker_records_history():
    t = NullTracker()
    t.log({"loss": 1.0}, step=0)
    t.log({"loss": 0.5}, step=1)
    t.finish()
    assert t.history == [(0, {"loss": 1.0}), (1, {"loss": 0.5})]


def test_get_tracker_defaults_to_null():
    t = get_tracker("null")
    assert isinstance(t, NullTracker)
    assert isinstance(t, Tracker)


def test_get_tracker_wandb_falls_back_when_absent():
    # wandb is not in the base env; must degrade to NullTracker, not raise.
    t = get_tracker("wandb", project="drrl")
    assert isinstance(t, NullTracker)

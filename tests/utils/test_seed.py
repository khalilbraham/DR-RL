"""Tests for deterministic seeding."""

import random

import numpy as np
import pytest

from drrl.utils.seed import SeedState, set_seed


def test_set_seed_returns_state_with_core_backends():
    state = set_seed(123)
    assert isinstance(state, SeedState)
    assert state.seed == 123
    assert "python" in state.seeded
    assert "numpy" in state.seeded


def test_set_seed_is_reproducible_for_python_and_numpy():
    set_seed(7)
    a_py = [random.random() for _ in range(5)]
    a_np = np.random.rand(5)

    set_seed(7)
    b_py = [random.random() for _ in range(5)]
    b_np = np.random.rand(5)

    assert a_py == b_py
    assert np.array_equal(a_np, b_np)


def test_different_seeds_diverge():
    set_seed(1)
    x = np.random.rand(10)
    set_seed(2)
    y = np.random.rand(10)
    assert not np.array_equal(x, y)


def test_negative_seed_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)


def test_seeded_tuple_is_sorted():
    state = set_seed(0)
    assert list(state.seeded) == sorted(state.seeded)

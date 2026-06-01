"""Tests for utilities that should work today (no simulator needed)."""

import numpy as np

from lagrangian_mbrl.utils import seed_everything


def test_seed_everything_returns_seed_and_is_reproducible():
    s = seed_everything(123)
    assert s == 123
    a = np.random.rand(5)

    seed_everything(123)
    b = np.random.rand(5)

    np.testing.assert_array_equal(a, b)

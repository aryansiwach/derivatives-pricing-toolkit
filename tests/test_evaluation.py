import numpy as np
import pytest

from evaluation import rmse, mae, iv_rmse, time_call


def test_rmse_zero_for_identical_arrays():
    a = [1.0, 2.0, 3.0]
    assert rmse(a, a) == 0.0


def test_rmse_known_value():
    a = [0.0, 0.0]
    b = [3.0, 4.0]
    assert rmse(a, b) == pytest.approx(np.sqrt((9 + 16) / 2))


def test_mae_known_value():
    a = [0.0, 0.0]
    b = [3.0, -4.0]
    assert mae(a, b) == pytest.approx(3.5)


def test_iv_rmse_matches_rmse():
    model = [0.2, 0.22, 0.25]
    market = [0.19, 0.23, 0.24]
    assert iv_rmse(model, market) == rmse(model, market)


def test_time_call_returns_positive_timing():
    def slow_add(a, b):
        return a + b
    result, mean_t, std_t = time_call(slow_add, 2, 3, n_reps=3)
    assert result == 5
    assert mean_t >= 0
    assert std_t >= 0

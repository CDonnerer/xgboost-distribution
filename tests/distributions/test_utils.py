import pytest

import numpy as np
import pandas as pd

from xgboost_distribution.distributions.utils import (
    check_all_ge_zero,
    check_all_gt_zero,
    check_all_integer,
    safe_exp,
)


@pytest.mark.parametrize(
    "x",
    [np.array([0, 1]), pd.Series([-1, 0, 2])],
)
def test_check_is_integer(x):
    check_all_integer(x)


@pytest.mark.parametrize(
    "x",
    [np.array([0.1, 1.2]), pd.Series([-1.1, 0.4, 2.3])],
)
def test_check_is_integer_raises(x):
    with pytest.raises(ValueError):
        check_all_integer(x)


@pytest.mark.parametrize(
    "x",
    [np.array([0, 1]), pd.Series([0, 1.2, 2])],
)
def test_check_is_ge_zero(x):
    check_all_ge_zero(x)


@pytest.mark.parametrize(
    "x",
    [np.array([-0.1, 1.2]), pd.Series([-1.1, 0.4, 2.3])],
)
def test_check_is_ge_zero_raises(x):
    with pytest.raises(ValueError):
        check_all_ge_zero(x)


@pytest.mark.parametrize(
    "x",
    [np.array([1, 2]), pd.Series([0.1, 1.2, 2.2])],
)
def test_check_is_gt_zero(x):
    check_all_gt_zero(x)


@pytest.mark.parametrize(
    "x",
    [np.array([0, 1]), pd.Series([-1.1, 0.4, 2.3])],
)
def test_check_is_gt_zero_raises(x):
    with pytest.raises(ValueError):
        check_all_gt_zero(x)


@pytest.mark.parametrize(
    "x",
    [
        np.array([100], dtype="float32"),  # will get clipped to max
        np.array([-100], dtype="float32"),  # will get clipped to min
    ],
)
def test_safe_exp(x):
    x_exp = safe_exp(x)

    # ensure no infs and zeros
    assert np.isfinite(x_exp).all()
    assert x_exp.all()

    # check stabiilty in divison setting
    large_div = np.float32(1e6) / x_exp
    assert np.isfinite(large_div).all()

    small_div = np.float32(1e-6) / x_exp
    assert small_div.all()

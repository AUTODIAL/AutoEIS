import numpy as np
import sklearn.metrics as skmetrics

from autoeis import utils

# Real numbers
x1 = np.random.rand(10)
x2 = np.random.rand(10)
# Complex numbers
y1 = x1 + np.zeros(10) * 1j
y2 = x2 + np.zeros(10) * 1j


def test_mse_score():
    # Real numbers
    assert utils.mse_score(x1, x2) == skmetrics.mean_squared_error(x1, x2)
    # Complex numbers
    assert utils.mse_score(y1, y2) == skmetrics.mean_squared_error(x1, x2)


def test_rmse_score():
    # Real numbers
    assert utils.rmse_score(x1, x2) == skmetrics.mean_squared_error(x1, x2, squared=False)
    # Complex numbers
    assert utils.rmse_score(y1, y2) == skmetrics.mean_squared_error(x1, x2, squared=False)


def test_mape_score():
    # Real numbers
    assert utils.mape_score(x1, x2) == skmetrics.mean_absolute_percentage_error(x1, x2) * 100
    # Complex numbers
    assert utils.mape_score(y1, y2) == skmetrics.mean_absolute_percentage_error(x1, x2) * 100


def test_r2_score():
    # Real numbers
    assert utils.r2_score(x1, x2) == skmetrics.r2_score(x1, x2)
    # Complex numbers
    assert utils.r2_score(y1, y2) == skmetrics.r2_score(x1, x2)

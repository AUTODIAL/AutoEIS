"""
Collection of functions for calculating metrics.

.. currentmodule:: autoeis.metrics

.. autosummary::
   :toctree: generated/

    mape_score
    mse_score
    rmse_score
    r2_score

"""

import numpy as np


def _assert_compatible_shapes(y_true: np.ndarray, y_pred: np.ndarray, axis: int):
    # y_true must be 1D, but y_pred can be ND (ND y_pred is complicated)
    assert y_true.squeeze().ndim == 1, "y_true must be 1D."
    # This is not necessary (numpy catches it), but it's a good sanity check
    msg = (
        f"y_pred {y_pred.shape} is incompatibe with y_true {y_true.shape} "
        f"given axis = {axis}"
    )
    assert y_true.shape[axis] == y_pred.shape[axis], msg


def _reshape_given_axis(y_true: np.ndarray, y_pred: np.ndarray, axis: int):
    y_true = y_true.squeeze()
    # For broadcasting to work correctly, ensure y_true is expanded in the correct axis
    other_axes = [i for i in range(y_pred.ndim) if i != axis]
    y_true = np.expand_dims(y_true, axis=other_axes)
    return y_true


def mape_score(y_true: np.ndarray, y_pred: np.ndarray, axis=0):
    """
    Calculates the generalized MAPE (Mean Absolute Percentage Error) score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (true) values.
    y_pred : np.ndarray
        Predicted values.
    axis : int, optional
        Axis along which to calculate the MAPE score. Default is 0.

    Returns
    -------
    float
        The MAPE score as a percentage.

    Notes
    -----
    This function handles complex numbers in the input arrays.
    """
    y_true = _reshape_given_axis(y_true, y_pred, axis)
    _assert_compatible_shapes(y_true, y_pred, axis)
    # NOTE: abs is needed to handle complex numbers
    return np.mean(np.abs((y_true - y_pred) / y_true), axis=axis) * 100


def mse_score(y_true: np.ndarray, y_pred: np.ndarray, axis=0):
    """
    Calculates the generalized MSE (Mean Squared Error) score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (true) values.
    y_pred : np.ndarray
        Predicted values.
    axis : int, optional
        Axis along which to calculate the MSE score. Default is 0.

    Returns
    -------
    float
        The MSE score.

    Notes
    -----
    This function handles complex numbers in the input arrays.
    """
    y_true = _reshape_given_axis(y_true, y_pred, axis)
    _assert_compatible_shapes(y_true, y_pred, axis)
    # NOTE: abs is needed to handle complex numbers
    return np.mean(np.abs(y_true - y_pred) ** 2, axis=axis)


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray, axis=0):
    """
    Calculates the generalized RMSE (Root Mean Squared Error) score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (true) values.
    y_pred : np.ndarray
        Predicted values.
    axis : int, optional
        Axis along which to calculate the RMSE score. Default is 0.

    Returns
    -------
    float
        The RMSE score.

    Notes
    -----
    This function handles complex numbers in the input arrays.
    """
    y_true = _reshape_given_axis(y_true, y_pred, axis)
    _assert_compatible_shapes(y_true, y_pred, axis)
    return np.sqrt(mse_score(y_true, y_pred, axis=axis))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, axis=0):
    """
    Calculates the generalized R2 score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (true) values.
    y_pred : np.ndarray
        Predicted values.
    axis : int, optional
        Axis along which to calculate the R2 score. Default is 0.

    Returns
    -------
    float
        The R2 score.

    Notes
    -----
    This function handles complex numbers in the input arrays.
    """
    y_true = _reshape_given_axis(y_true, y_pred, axis)
    _assert_compatible_shapes(y_true, y_pred, axis)
    # NOTE: abs is needed to handle complex numbers
    ssr = np.sum(np.abs(y_true - y_pred) ** 2, axis=axis)
    sst = np.sum(np.abs(y_true - np.mean(y_true)) ** 2)
    return 1 - ssr / sst

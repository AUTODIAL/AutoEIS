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


def mape_score(y_true, y_pred):
    """
    Calculates the generalized MAPE (Mean Absolute Percentage Error) score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (true) values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        The MAPE score as a percentage.

    Notes
    -----
    This function handles complex numbers in the input arrays.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mse_score(y_true, y_pred):
    """
    Calculates the generalized MSE (Mean Squared Error) score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (true) values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        The MSE score.

    Notes
    -----
    This function handles complex numbers in the input arrays.
    """
    return np.mean(np.abs(y_true - y_pred) ** 2)


def rmse_score(y_true, y_pred):
    """
    Calculates the generalized RMSE (Root Mean Squared Error) score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (true) values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        The RMSE score.

    Notes
    -----
    This function handles complex numbers in the input arrays.
    """
    return np.sqrt(mse_score(y_true, y_pred))


def r2_score(y_true, y_pred):
    """
    Calculates the generalized R2 score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (true) values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        The R2 score.

    Notes
    -----
    This function handles complex numbers in the input arrays.
    """
    ssr = np.sum(np.abs(y_true - y_pred) ** 2)
    sst = np.sum(np.abs(y_true - np.mean(y_true)) ** 2)
    return 1 - ssr / sst

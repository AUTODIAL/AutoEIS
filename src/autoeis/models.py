"""
Collection of functions to be used as models for Bayesian inference.

.. currentmodule:: autoeis.models

.. autosummary::
   :toctree: generated/

    circuit_regression_complex

"""

from collections import namedtuple
from typing import Callable, Mapping

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import Distribution

Impedance = namedtuple("Impedance", ["real", "imag"])


def circuit_regression_complex(
    freq: np.ndarray[float],
    priors: Mapping[str, Distribution],
    circuit_fn: Callable[[np.ndarray | float, np.ndarray], np.ndarray[complex]],
    Z: np.ndarray[complex] = None,
):
    """NumpyRo model for Bayesian inference of circuit component values.

    This model uses two separate distributions for the real and imaginary
    parts of the impedance.

    Parameters
    ----------
    freq : np.ndarray[float]
        Frequencies to evaluate the posterior predictive distribution at.
    priors : Mapping[str, Distribution]
        Priors for the circuit parameters as a dictionary of parameter names
        and distributions.
    circuit_fn : Callable[[np.ndarray | float, np.ndarray], np.ndarray[complex]]
        Function to compute the circuit impedance, parameterized by frequency
        and the circuit parameters.
    Z : np.ndarray[complex], optional
        Observed impedance data. If ``None``, the model will be used for
        posterior prediction, otherwise it will be used for Bayesian
        inference of circuit parameters.
    """
    Z = Impedance(real=None, imag=None) if Z is None else Z
    # Sample each element of X separately
    p = jnp.array([numpyro.sample(k, v) for k, v in priors.items()])
    # Predict Z using the model
    Z_pred = circuit_fn(freq, p)
    # TODO: Try sampling from the concatenated real and imaginary parts
    # Define observation model for real and imaginary parts of Z
    sigma_real = numpyro.sample("sigma_real", dist.Exponential(rate=1.0))
    numpyro.sample("obs_real", dist.Normal(Z_pred.real, sigma_real), obs=Z.real)
    sigma_imag = numpyro.sample("sigma_imag", dist.Exponential(rate=1.0))
    numpyro.sample("obs_imag", dist.Normal(Z_pred.imag, sigma_imag), obs=Z.imag)

"""
Collection of functions to be used as models for Bayesian inference.

.. currentmodule:: autoeis.models

.. autosummary::
   :toctree: generated/

    circuit_regression
    circuit_regression_wrapped

"""
from collections import namedtuple
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from autoeis import utils

Tin = [Union[np.ndarray, float], np.ndarray]
Tout = np.ndarray[complex]

Impedance = namedtuple("Impedance", ["real", "imag"])


def circuit_regression(
    freq: np.ndarray[float],
    priors: dict[str, dist.Distribution],
    circuit: str,
    Z: np.ndarray[complex],
):
    """NumpyRo model for Bayesian inference of circuit component values."""
    # Sample each element of X separately
    p = jnp.array([numpyro.sample(k, v) for k, v in priors.items()])
    # Predict Z using the model
    circuit_fn = utils.generate_circuit_fn(circuit)
    circuit_fn = jax.jit(circuit_fn)
    Z_pred = circuit_fn(freq, p)
    # Define observation model for real and imaginary parts of Z
    sigma_real = numpyro.sample("sigma_real", dist.Exponential(rate=1.0))
    numpyro.sample("obs_real", dist.Normal(Z_pred.real, sigma_real), obs=Z.real)
    sigma_imag = numpyro.sample("sigma_imag", dist.Exponential(rate=1.0))
    numpyro.sample("obs_imag", dist.Normal(Z_pred.imag, sigma_imag), obs=Z.imag)


# NOTE: To reuse the model for posterior prediction -> optional Z
def circuit_regression_wrapped(
    freq: np.ndarray[float],
    priors: dict[str, dist.Distribution],
    circuit_fn: Callable[Tin, Tout],
    Z: np.ndarray[complex] = None,
):
    """NumpyRo model for Bayesian inference of circuit component values."""
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

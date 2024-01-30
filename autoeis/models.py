"""
Collection of functions to be used as models for Bayesian inference.

.. currentmodule:: autoeis.models

.. autosummary::
   :toctree: generated/

    circuit_component_regression
    circuit_component_regression_fn_wrapped

"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from autoeis import utils


def circuit_component_regression(
    Z: np.ndarray[complex],
    freq: np.ndarray[float],
    priors: dict[str, dist.Distribution],
    circuit: str,
):
    """NumpyRo model for Bayesian inference of circuit component values."""
    # Sample each element of X separately
    X = jnp.array([numpyro.sample(k, v) for k, v in priors.items()])
    # Predict Z using the model
    circuit_fn = utils.generate_circuit_fn(circuit)
    circuit_fn = jax.jit(circuit_fn)
    Z_pred = circuit_fn(X, freq)
    # Define observation model for real and imaginary parts of Z
    sigma_real = numpyro.sample("sigma_real", dist.Exponential(rate=1.0))
    numpyro.sample("obs_real", dist.Normal(Z_pred.real, sigma_real), obs=Z.real)
    sigma_imag = numpyro.sample("sigma_imag", dist.Exponential(rate=1.0))
    numpyro.sample("obs_imag", dist.Normal(Z_pred.imag, sigma_imag), obs=Z.imag)


def circuit_component_regression_fn_wrapped(
    Z: np.ndarray[complex],
    freq: np.ndarray[float],
    priors: dict[str, dist.Distribution],
    circuit_fn: callable,
):
    """NumpyRo model for Bayesian inference of circuit component values."""
    # Sample each element of X separately
    X = jnp.array([numpyro.sample(k, v) for k, v in priors.items()])
    # Predict Z using the model
    Z_pred = circuit_fn(X, freq)
    # Define observation model for real and imaginary parts of Z
    sigma_real = numpyro.sample("sigma_real", dist.Exponential(rate=1.0))
    numpyro.sample("obs_real", dist.Normal(Z_pred.real, sigma_real), obs=Z.real)
    sigma_imag = numpyro.sample("sigma_imag", dist.Exponential(rate=1.0))
    numpyro.sample("obs_imag", dist.Normal(Z_pred.imag, sigma_imag), obs=Z.imag)

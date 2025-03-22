"""
Collection of functions to be used as models for Bayesian inference.

.. currentmodule:: autoeis.models

.. autosummary::
   :toctree: generated/

    circuit_regression_magnitude
    circuit_regression_nyquist
    circuit_regression_bode

Notes
-----

1. For positive variable `y`, there are multiple options to model the observation
to ensure that the predicted value is also positive:

- Sample ``y`` with a shifted ``HalfNormal`` (see pyro-ppl/numpyro/issues/932)
- Sample ``y`` with ``TruncatedNormal`` with 0 as the lower bound
- Sample ``y - y_gt`` with ``Normal(0, sigma)``

2. Using ``Normal`` distribution for positive variables may lead to negative
samples

"""

import copy
from collections import namedtuple
from typing import Callable, Mapping

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import Distribution

Impedance = namedtuple("Impedance", ["real", "imag"])


def circuit_regression_magnitude(
    freq: np.ndarray[float],
    priors: Mapping[str, Distribution],
    circuit_fn: Callable[[np.ndarray | float, np.ndarray], np.ndarray[complex]],
    Z: np.ndarray[complex] = None,
):
    """Inference model for ECM parameters based on |Z|.

    This model infers the circuit parameters by matching the magnitude of the
    impedance measurements. To do this, the MCMC sampling is performed on the
    magnitude of the impedance.

    Parameters
    ----------
    freq : np.ndarray[float]
        Frequencies to evaluate the posterior predictive distribution at.
    priors : Mapping[str, Distribution]
        Priors for the circuit parameters as a dictionary of parameter labels
        and distributions.
    circuit_fn : Callable[[np.ndarray | float, np.ndarray], np.ndarray[complex]]
        Function to compute the circuit impedance, parameterized by frequency
        and circuit parameters, fn(freq, p).
    Z : np.ndarray[complex], optional
        Observed impedance data. If ``None``, the model will be used for
        posterior prediction, otherwise it will be used for Bayesian
        inference of circuit parameters.
    """
    # Make a deep copy of the priors to avoid side effects (pyro-ppl/numpyro/issues/1651)
    priors = copy.deepcopy(priors)
    # Sample each element of X separately
    p = jnp.array([numpyro.sample(k, v) for k, v in priors.items()])
    # Predict Z using the model
    Z_pred = circuit_fn(freq, p)

    # Short-circuit if posterior prediction is requested
    if Z is None:
        _posterior_predictive(Z_pred)
        return

    # Observation model based on the magnitude of the impedance
    error_model = "abs(diff)"
    assert error_model in ["diff(abs)", "abs(diff)"]
    sigma = numpyro.sample("sigma", dist.Exponential(rate=1.0))
    if error_model == "diff(abs)":  # |Z| - |Z_gt|
        error = jnp.abs(Z) - jnp.abs(Z_pred)
        numpyro.sample("obs", dist.Normal(0.0, sigma), obs=error)
    if error_model == "abs(diff)":  # |Z - Z_gt|
        error = jnp.abs(Z - Z_pred)
        numpyro.sample("obs", dist.HalfNormal(sigma), obs=error)


def circuit_regression_nyquist(
    freq: np.ndarray[float],
    priors: Mapping[str, Distribution],
    circuit_fn: Callable[[np.ndarray | float, np.ndarray], np.ndarray[complex]],
    Z: np.ndarray[complex] = None,
):
    """Inference model for ECM parameters based on Nyquist plot.

    This model infers the circuit parameters by matching the impedance
    measurements as plotted on the Nyquist plot. To do this, the MCMC sampling
    is performed on the real and imaginary parts of the impedance.

    Parameters
    ----------
    freq : np.ndarray[float]
        Frequencies to evaluate the posterior predictive distribution at.
    priors : Mapping[str, Distribution]
        Priors for the circuit parameters as a dictionary of parameter labels
        and distributions.
    circuit_fn : Callable[[np.ndarray | float, np.ndarray], np.ndarray[complex]]
        Function to compute the circuit impedance, parameterized by frequency
        and circuit parameters, fn(freq, p).
    Z : np.ndarray[complex], optional
        Observed impedance data. If ``None``, the model will be used for
        posterior prediction, otherwise it will be used for Bayesian
        inference of circuit parameters.
    """
    # Make a deep copy of the priors to avoid side effects (pyro-ppl/numpyro/issues/1651)
    priors = copy.deepcopy(priors)
    # Sample each element of X separately
    p = jnp.array([numpyro.sample(k, v) for k, v in priors.items()])
    # Predict Z using the model
    Z_pred = circuit_fn(freq, p)

    # Short-circuit if posterior prediction is requested
    if Z is None:
        _posterior_predictive(Z_pred)
        return

    # Observation model based on the Nyquist plot
    sigma = {
        "real": numpyro.sample("sigma.real", dist.Exponential(rate=1.0)),
        "imag": numpyro.sample("sigma.imag", dist.Exponential(rate=1.0)),
    }
    numpyro.sample("obs.real", dist.Normal(Z_pred.real, sigma["real"]), obs=Z.real)
    numpyro.sample("obs.imag", dist.Normal(Z_pred.imag, sigma["imag"]), obs=Z.imag)


def circuit_regression_bode(
    freq: np.ndarray[float],
    priors: Mapping[str, Distribution],
    circuit_fn: Callable[[np.ndarray | float, np.ndarray], np.ndarray[complex]],
    Z: np.ndarray[complex] = None,
):
    """Inference model for ECM parameters based on Bode plot.

    This model infers the circuit parameters by matching the impedance
    measurements as plotted on the Bode plot. To do this, the MCMC sampling is
    performed on the magnitude and phase of the impedance.

    Parameters
    ----------
    freq : np.ndarray[float]
        Frequencies to evaluate the posterior predictive distribution at.
    priors : Mapping[str, Distribution]
        Priors for the circuit parameters as a dictionary of parameter labels
        and distributions.
    circuit_fn : Callable[[np.ndarray | float, np.ndarray], np.ndarray[complex]]
        Function to compute the circuit impedance, parameterized by frequency
        and circuit parameters, fn(freq, p).
    Z : np.ndarray[complex], optional
        Observed impedance data. If ``None``, the model will be used for
        posterior prediction, otherwise it will be used for Bayesian
        inference of circuit parameters.
    """
    # Make a deep copy of the priors to avoid side effects (pyro-ppl/numpyro/issues/1651)
    priors = copy.deepcopy(priors)
    # Sample each element of X separately
    p = jnp.array([numpyro.sample(k, v) for k, v in priors.items()])
    # Predict Z using the model
    Z_pred = circuit_fn(freq, p)
    mag, phase = jnp.abs(Z_pred), jnp.angle(Z_pred)

    # Custom observation model based on the Bode plot
    if Z is None:
        mag_gt = phase_gt = None
    else:
        mag_gt, phase_gt = jnp.abs(Z), jnp.angle(Z)
        # Log-transform the magnitude, otherwise low-frequency values dominate
        mag, mag_gt = jnp.log10(mag), jnp.log10(mag_gt)
    sigma_mag = numpyro.sample("sigma.mag", dist.Exponential(rate=1.0))
    sigma_phase = numpyro.sample("sigma.phase", dist.Exponential(rate=1.0))
    dist_mag = dist.TruncatedDistribution(dist.Normal(mag, sigma_mag), low=mag)
    dist_phase = dist.Normal(phase, sigma_phase)
    numpyro.sample("obs.mag", dist_mag, obs=mag_gt)
    numpyro.sample("obs.phase", dist_phase, obs=phase_gt)

    # NOTE: The following code is an alternative to the above code
    # error_mag = jnp.abs(jnp.log10(mag) - jnp.log10(mag_gt))
    # sigma_mag = numpyro.sample("sigma.mag", dist.HalfNormal())
    # numpyro.sample("obs.error.mag", dist.HalfNormal(sigma_mag), obs=error_mag)
    # error_phase = jnp.abs(phase - phase_gt)
    # sigma_phase = numpyro.sample("sigma.phase", dist.HalfNormal())
    # numpyro.sample("obs.error.phase", dist.HalfNormal(sigma_phase), obs=error_phase)


def _posterior_predictive(Z: np.ndarray[complex]):
    """Private helper to compute the posterior predictive distribution."""
    # NOTE: Z is predicted impedance, not observed impedance
    sigma_real = numpyro.sample("sigma.real", dist.Exponential(rate=1.0))
    sigma_imag = numpyro.sample("sigma.imag", dist.Exponential(rate=1.0))
    numpyro.sample("obs.real", dist.Normal(Z.real, sigma_real))
    numpyro.sample("obs.imag", dist.Normal(Z.imag, sigma_imag))

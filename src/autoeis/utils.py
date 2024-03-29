"""
Collection of utility functions used throughout the package.

.. currentmodule:: autoeis.utils

.. autosummary::
   :toctree: generated/

    circuit_complexity
    generate_circuit_fn
    generate_circuit_fn_impedance_backend
    fit_circuit_parameters
    are_circuits_equivalent
    initialize_priors
    initialize_priors_from_posteriors
    validate_circuits_dataframe

"""

import logging
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

import jax  # NOQA: F401
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import psutil
from impedance.models.circuits import CustomCircuit
from impedance.models.circuits.fitting import set_default_bounds
from jax import random
from numpy import pi  # NOQA: F401
from numpyro.distributions import Distribution
from numpyro.infer import MCMC, Predictive
from scipy import stats
from scipy.optimize import curve_fit

import __main__

from . import models, parser

log = logging.getLogger(__name__)


# >>> General utils


def is_notebook():
    """Returns True if the code is running in a Jupyter notebook."""

    # Source: https://discourse.jupyter.org/t/find-out-if-my-code-runs-inside-a-notebook-or-jupyter-lab/6935/21
    def get_runtime():
        """Returns the runtime environment."""
        if "google.colab" in sys.modules:
            return "Google Colab"
        elif "ipykernel" in sys.modules:
            if "jupyter" in sys.modules:
                return "JupyterLab"
            else:
                return "Jupyter Notebook"
        elif "win32" in sys.platform:
            if "CMDEXTVERSION" in os.environ:
                return "Windows Command Prompt"
            else:
                return "Windows PowerShell"
        elif "darwin" in sys.platform:
            return "MacOS Terminal"
        else:
            if hasattr(__main__, "__file__"):
                return "Linux Terminal"
            else:
                return "Interactive Python Shell"

    runtime = get_runtime()
    return runtime in ["Google Colab", "JupyterLab", "Jupyter Notebook"]


@dataclass
class Settings:
    """Settings for the AutoEIS package."""

    loglevel: int = logging.WARNING
    ncores: int = psutil.cpu_count(logical=False)
    notebook: bool = is_notebook()
    progress_bar: bool = True


def flatten(xs: Iterable) -> list:
    """Returns a list of all elements in a nested iterable.

    Parameters
    ----------
    xs: list
        A nested iterable.

    Returns
    -------
    list
        A flattened list.

    Examples
    --------
    >>> flatten([1, 2, [3, 4], [5, [6, 7]]])
    [1, 2, 3, 4, 5, 6, 7]
    """

    def _flatten(xs):
        """Returns a generator that flattens a nested iterable."""
        for x in xs:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from flatten(x)
            else:
                yield x

    return list(_flatten(xs))


def suppress_output_legacy(func: Callable) -> Callable:
    """Suppresses the output of a function.

    Parameters
    ----------
    func: callable
        Input function whose output is to be suppressed.

    Returns
    -------
    callable
        A wrapped function with the output suppressed.

    Notes
    -----
    This approach only works when stdout/stderr are managed by Python.
    """

    class _SuppressOutput:
        def __enter__(self):
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _SuppressOutput():
            return func(*args, **kwargs)

    return wrapped


@contextmanager
def suppress_output():
    """Suppresses the output of a block of code using file descriptors."""
    # NOTE: This approach is more system-level than suppress_output
    # Save the current file descriptors
    original_stderr_fd = os.dup(2)
    original_stdout_fd = os.dup(1)

    try:
        # Use os.devnull to redirect the standard output and standard error
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        # Restore the original file descriptors
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)


# <<< General utils


# >>> Circuit utils


def parse_initial_guess(
    p0: np.ndarray | dict[str, float] | list[float],
    circuit: str,
) -> np.ndarray:
    """Parses the initial guess for circuit parameters from various formats
    and returns an array of parameters.

    Parameters
    ----------
    p0: np.ndarray | dict[str, float] | list[float]
        The initial guess for the circuit parameters.
    circuit: str
        The circuit string.

    Returns
    -------
    np.ndarray
        The array of initial guesses.

    Raises
    ------
    ValueError
        If the initial guess is not not a dict nor array-like.

    Notes
    -----
    If no initial guess is provided, a random array of parameters is returned.
    """
    num_params = parser.count_parameters(circuit)
    if p0 is None:
        return np.random.rand(num_params)
    if isinstance(p0, dict):
        return np.fromiter(p0.values(), dtype=float)
    if isinstance(p0, (list, np.ndarray)):
        return np.array(p0)
    raise ValueError(f"Invalid initial guess: {p0}")


def fit_circuit_parameters_legacy(
    circuit: str,
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    p0: np.ndarray[float] | dict[str, float] = None,
    iters: int = 1,
    maxfev: int = 1000,
) -> dict[str, float]:
    """Fits and returns the parameters of a circuit to impedance data.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    freq : np.ndarray[float]
        Frequencies corresponding to the impedance data.
    Z : np.ndarray[complex]
        Impedance data.
    p0 : np.ndarray[float] | dict[str, float], optional
        Initial guess for the circuit parameters. Default is None.
    iters : int, optional
        Maximum number of iterations for the circuit fitter. Default is 1.
    maxfev : int, optional
        Maximum number of function evaluations for the circuit fitter.
        Default is 1000.

    Returns
    -------
    dict[str, float]
        Fitted parameters as a dictionary of parameter names and values.

    Notes
    -----
    This function uses ``impedance.py`` to fit the circuit parameters.

    """
    # NOTE: Each circuit eval ~ 1 ms, so 1000 evals ~ 1 s
    # Deal with initial guess
    num_params = parser.count_parameters(circuit)
    p0 = parse_initial_guess(p0, circuit)
    assert len(p0) == num_params, "Wrong number of parameters in initial guess."

    # Fit circuit parameters
    circuit_impy = CustomCircuit(
        circuit=parser.convert_to_impedance_format(circuit),
        initial_guess=p0,
    )
    # HACK: Use multiple random initial guesses to avoid local minima
    err_min = np.inf
    for _ in range(iters):
        try:
            circuit_impy.fit(freq, Z, maxfev=maxfev)
        except RuntimeError:
            continue
        err = np.mean(np.abs(circuit_impy.predict(freq) - Z) ** 2)
        if err < err_min:
            err_min = err
            p0 = circuit_impy.parameters_
        circuit_impy.initial_guess = np.random.rand(num_params).tolist()

    if err_min == np.inf:
        raise RuntimeError("Failed to fit the circuit parameters.")

    labels = parser.get_parameter_labels(circuit)
    return dict(zip(labels, p0))


def fit_circuit_parameters(
    circuit: str,
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    p0: np.ndarray[float] | dict[str, float] = None,
    iters: int = 1,
    maxfev: int = 1000,
    ftol: float = 1e-13,
) -> dict[str, float]:
    """Fits and returns the parameters of a circuit to impedance data.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    freq : np.ndarray[float]
        Frequencies corresponding to the impedance data.
    Z : np.ndarray[complex]
        Impedance data.
    p0 : np.ndarray[float] | dict[str, float], optional
        Initial guess for the circuit parameters. Default is None.
    iters : int, optional
        Maximum number of iterations for the circuit fitter. Default is 1.
    maxfev : int, optional
        Maximum number of function evaluations for the circuit fitter.
        Default is 1000.
    ftol : float, optional
        Tolerance for the convergence criterion. Default is 1e-13.

    Returns
    -------
    dict[str, float]
        Fitted parameters as a dictionary of parameter names and values.

    Notes
    -----
    This function uses SciPy's ``curve_fit`` to fit the circuit parameters.
    """
    # Define objective function
    Zc = np.hstack([Z.real, Z.imag])
    fn = generate_circuit_fn(circuit, jit=True, concat=True)
    # Format obj function as f(freq, *p) not f(freq, p) for curve_fit
    obj_fn = lambda freq, *p: fn(freq, p)  # noqa: E731

    # >>> Alternatively, use impedance.py to create the objective function
    # from impedance.models.circuits.fitting import wrapCircuit
    # circuit_impy = parser.convert_to_impedance_format(circuit)
    # obj_fn = wrapCircuit(circuit_impy, constants={})
    # <<<

    # Sanitize initial guess
    num_params = parser.count_parameters(circuit)
    p0 = parse_initial_guess(p0, circuit)
    assert len(p0) == num_params, "Wrong number of parameters in initial guess."

    # Assemble kwargs for curve_fit
    circuit_impy = parser.convert_to_impedance_format(circuit)
    bounds = set_default_bounds(circuit_impy)
    kwargs = {"p0": p0, "bounds": bounds, "maxfev": maxfev, "ftol": ftol}

    # Fit circuit parameters by brute force
    err_min = np.inf
    for _ in range(iters):
        try:
            popt, pcov = curve_fit(obj_fn, freq, Zc, **kwargs)
        except RuntimeError:
            continue
        err = np.mean((obj_fn(freq, *popt) - Zc) ** 2)
        if err < err_min:
            err_min = err
            p0 = popt
        kwargs["p0"] = np.random.rand(num_params)

    if err_min == np.inf:
        raise RuntimeError("Failed to fit the circuit parameters.")

    variables = parser.get_parameter_labels(circuit)
    return dict(zip(variables, p0))


def eval_circuit(circuit: str, f: np.ndarray | float, p: np.ndarray) -> np.ndarray[complex]:
    """Returns the impedance of a circuit at a given frequency and parameters.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    f : np.ndarray | float
        Frequencies at which to evaluate the circuit.
    p : np.ndarray
        Circuit parameters.

    Returns
    -------
    np.ndarray[complex]
        The impedance of the circuit at the given frequency and parameters.
    """
    Z_expr = parser.generate_mathematical_expr(circuit)
    return eval(Z_expr)


def generate_circuit_fn(circuit: str, jit=False, concat=False):
    """Generates a function to compute the circuit impedance, parameterized
    by frequency and the circuit parameters.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    jit : bool, optional
        If True, uses JAX to compile the function. Default is False.
    concat : bool, optional
        If True, the generated function returns concatenated real and
        imaginary parts of the impedance, otherwise it returns the complex
        impedance. Default is False.

    Returns
    -------
    callable
        A function that takes in frequency and the circuit parameters
        and returns the impedance.
    """

    def Z_complex(freq: np.ndarray, p: np.ndarray | float) -> np.ndarray[complex]:
        return eval_circuit(circuit, freq, p)

    def Z_concat(freq: np.ndarray, p: np.ndarray | float) -> np.ndarray[complex]:
        Z = Z_complex(freq, p)
        hstack = jnp.hstack if jit else np.hstack
        return hstack([Z.real, Z.imag])

    fn = Z_concat if concat else Z_complex
    fn = jax.jit(fn) if jit else fn

    return fn


def generate_circuit_fn_impedance_backend(circuit: str):
    """Generates a function to compute the circuit impedance, parameterized
    by frequency and the circuit parameters, using ``impedance.py``.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    callable
        A function that takes in frequency and the circuit parameters
        and returns the impedance.
    """
    num_params = parser.count_parameters(circuit)
    # Convert circuit string to impedance.py format
    circuit = parser.convert_to_impedance_format(circuit)
    # Generate circuit function
    p0 = np.full(num_params, np.nan)
    circuit = CustomCircuit(circuit, initial_guess=p0)

    def func(freq: np.ndarray | float, p: np.ndarray) -> np.ndarray:
        circuit.parameters_ = p
        return circuit.predict(freq)

    return func


def circuit_complexity(circuit: str) -> list[int]:
    """Computes the component complexity of the circuit.

    Component complexity is defined as how deep it is nested in the circuit.

    Parameters
    ----------
    circuit: str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    list[int]
        A list of component complexity values.

    Examples
    --------
    >>> circuit_complexity("R1-[P2,[P3,R4]]-R5")
    [0, 1, 2, 2, 0]
    """

    def increment(arr):
        """Adds one to each element in a nested list."""
        return [increment(e) if isinstance(e, list) else e + 1 for e in arr]

    def depth(arr: list):
        """Computes the depth of each element in a nested list."""
        return [increment(depth(e)) if isinstance(e, list) else 0 for e in arr]

    def split(arr: list, chars: list[str]):
        """Recursively splits comma-separated elements in a nested list."""
        out = []
        sep = "|".join(chars)
        for e in arr:
            if isinstance(e, list):
                out.append(split(e, chars))
            else:
                out.extend(re.split(sep, e))
        return out

    expr = parser.circuit_to_nested_expr(circuit)
    # Split "R1-R2", ["R1,R2"]-like expressions to account for all elements
    expr = split(expr, chars=[",", "-"])
    complexity = depth(expr)
    return flatten(complexity)


def are_circuits_equivalent(circuit1: str, circuit2: str, rtol: float = 1e-5) -> bool:
    """Checks if two circuit strings are equivalent.

    Parameters
    ----------
    circuit1 : str
        The first circuit string in CDC format. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    circuit2 : str
        The second circuit string in CDC format. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    rtol : float, optional
        The relative tolerance for the circuit equivalence check. See the
        Notes section for more details. Default is 1e-5.

    Returns
    -------
    bool
        True if the circuits are equivalent, False otherwise.

    Notes
    -----
    This function uses an approximate heuristic, i.e., it evaluates the
    circuits at a set of random frequencies and checks if the results are
    close enough to be considered equivalent.
    """

    def x0(circuit: str) -> np.ndarray[float]:
        """Custom x0 to test if two circuits are equivalent by comparing Z(x0).

        The idea is that if two circuits are equivalent, then Z(x0) should be
        the same for both circuits, given that x0 corresponds to the same
        component values in both circuits. Since the order of the components
        does not matter, we set the values of the components of the same type
        to be the same.
        """
        values = {"R": 0.85, "C": 0.75, "L": 0.30, "Pw": 0.15, "Pn": 0.6}
        labels = parser.get_parameter_labels(circuit)
        x0 = []
        for label in labels:
            ptype = parser.parse_parameter(label)
            x0.append(values[ptype])
        return np.array(x0)

    freq = np.logspace(-3, 3, 10)
    Z1 = generate_circuit_fn(circuit1)(freq, x0(circuit1))
    Z2 = generate_circuit_fn(circuit2)(freq, x0(circuit2))
    return np.allclose(Z1, Z2)


# <<< Circuit utils


# >>> Statistics utils


# TODO: Remove variables from input arguments
# TODO: Refactor inference functions to strip non-variable keys from MCMC samples
def initialize_priors(
    p0: Mapping[str, float], variables: Iterable[str]
) -> dict[str, Distribution]:
    """Initializes priors for a given circuit.

    Parameters
    ----------
    p0 : Mapping[str, float]
        Initial guess for the circuit parameters as a dictionary of parameter
        names and values.
    variables : Iterable[str]
        List of variable names.

    Returns
    -------
    dict[str, Distribution]
        Priors for the circuit parameters as a dictionary of parameter names
        and distributions.

    Notes
    -----
    This function assigns a uniform distribution for the exponent of CPE
    elements and a log-normal distribution for the rest of the parameters.
    """
    priors = {}
    for var in variables:
        value = p0[var]
        if "n" in var:
            # TODO: use a more informative prior for n, eg truncated normal
            # Exponent of CPE elements is bounded between 0 and 1
            priors[var] = dist.Uniform(0, 1)
        else:
            # Search over a log-normal dist spanning [0.01*u0, 100*u0]
            mean, std_dev = jnp.log(value), jnp.log(10)
            priors[var] = dist.LogNormal(mean, std_dev)
    return priors


def initialize_priors_from_posteriors(
    posterior: Mapping[str, np.ndarray[float]],
    variables: Iterable[str],
    dist_type: str = "lognormal",
) -> dict[str, Distribution]:
    """Creates new priors based on the posterior distributions.

    Parameters
    ----------
    posterior : Mapping[str, np.ndarray[float]]
        Posterior distributions for the circuit parameters as a dictionary
        of parameter names and distributions.
    variables : Iterable[str]
        List of variable names.
    dist_type : str, optional
        Type of prior distribution to use. Default is "lognormal".

    Returns
    -------
    dict[str, Distribution]
        Priors for the circuit parameters as a dictionary of parameter names
        and distributions.

    Notes
    -----
    To create new priors, a log-normal (or as specified) distribution is
    fitteed to the posterior distributions and the fitted parameters
    (e.g., mean, std, etc.) are used to generate the priors.

    For the exponent of CPE elements, a truncated normal distribution is
    used no matter what the ``dist_type`` is.
    """
    priors = {}
    for var in variables:
        samples = posterior[var]
        # Fit data to a truncated normal distribution for exponents of CPE elements
        # HACK: for better convergence (fewer parameters), fit a normal and truncate it
        if "n" in var:
            # Exponent of CPE elements is bounded between 0 and 1
            loc, scale = stats.norm.fit(samples)
            priors[var] = dist.TruncatedNormal(loc=loc, scale=1 * scale, low=0, high=1)
        # Fit data to a log-normal distribution for all other parameters
        else:
            # NOTE: s and scale in scipy.stats -> scale and np.exp(loc) in numpyro
            # NOTE: above conversion is only valid when loc = 0
            if dist_type == "lognormal":
                s, loc, scale = stats.lognorm.fit(samples, floc=0)
                priors[var] = dist.LogNormal(loc=np.log(scale), scale=8 * s)
            elif dist_type == "normal":
                loc, scale = stats.norm.fit(samples)
                priors[var] = dist.TruncatedNormal(
                    loc=loc, scale=1 * scale, low=0, high=np.inf
                )
            elif dist_type == "weibull":
                c, loc, scale = stats.weibull_min.fit(samples, floc=1)
                priors[var] = dist.Weibull(scale=scale, concentration=c)
            elif dist_type == "t":
                df, loc, scale = stats.t.fit(samples)
                priors[var] = dist.StudentT(df=df, loc=loc, scale=scale)
            else:
                raise ValueError(f"Unknown distribution: {dist_type}")
    return priors


def eval_posterior_predictive(
    mcmc: MCMC,
    circuit: str,
    freq: np.ndarray[float],
    priors: Mapping[str, Distribution] = None,
    rng_key: random.PRNGKey = None,
) -> np.ndarray[complex]:
    """Evaluate the posterior predictive distribution of a MCMC run.

    Parameters
    ----------
    mcmc : MCMC
        MCMC object.
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    freq : np.ndarray[float]
        Frequencies to evaluate the posterior predictive distribution at.
    priors : Mapping[str, Distribution], optional
        Priors for the circuit parameters as a dictionary of parameter names
        and distributions. Default is None.
    rng_key : random.PRNGKey, optional
        Random key for the MCMC run. Default is None.

    Returns
    -------
    np.ndarray[complex]
        Posterior predictive distribution of the circuit at the given frequencies.
    """
    rng_key = rng_key or random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    samples = mcmc.get_samples()
    circuit_fn = generate_circuit_fn(circuit, jit=True)

    # Deal with default arguments
    if priors is None:
        variables = parser.get_parameter_labels(circuit)
        p0 = {var: np.median(samples[var]) for var in variables}
        priors = initialize_priors(p0, variables)

    # Create a predictive distribution for the circuit parameters
    model = models.circuit_regression_wrapped
    predictive = Predictive(model, samples)

    # Evaluate the predictive distribution at the given frequency
    kwargs = {"freq": freq, "priors": priors, "circuit_fn": circuit_fn}
    predictions = predictive(rng_key_, **kwargs)
    Z_pred = predictions["obs_real"] + predictions["obs_imag"] * 1j

    return Z_pred


# <<< Statistics utils


# >>> Miscellaneous utils


def validate_circuits_dataframe(circuits: pd.DataFrame):
    """Ensures that the circuits dataframe has the correct format by
    checking the following:

    - Column names are valid (must be "circuitstring", "Parameters")
    - Column types are valid
        - "circuitstring" must be a string
        - "Parameters" must be a dictionary

    Parameters
    ----------
    circuits : pd.DataFrame
        Dataframe containing the circuits.

    Raises
    ------
    ValueError
        If the dataframe does not have the correct format.
    """
    # Check if the dataframe has the required columns
    required_columns = ["circuitstring", "Parameters"]
    missing = set(required_columns).difference(circuits.columns)
    assert not missing, f"Missing columns: {missing}"
    # Check if the circuitstring column contains only strings
    assert (
        circuits["circuitstring"].apply(lambda x: isinstance(x, str)).all()
    ), "circuitstring column must contain only strings."
    # Check if the Parameters column contains only dictionaries
    assert (
        circuits["Parameters"].apply(lambda x: isinstance(x, dict)).all()
    ), "Parameters column must contain only dictionaries."


# <<< Miscellaneous utils

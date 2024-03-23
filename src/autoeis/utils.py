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
import signal
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from typing import Union

import jax  # NOQA: F401
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from impedance.models.circuits import CustomCircuit
from impedance.models.circuits.fitting import set_default_bounds
from numpy import pi  # NOQA: F401
from rich.console import Console
from rich.logging import RichHandler
from scipy import stats
from scipy.optimize import curve_fit

import __main__

from . import parser

# Timeout circuit fitter functions after X seconds
TIMEOUT_AFTER = 15

# >>> Logging utils


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


def is_notebook():
    """Returns True if the code is running in a Jupyter notebook."""
    runtime = get_runtime()
    return runtime in ["Google Colab", "JupyterLab", "Jupyter Notebook"]


def get_logger(name: str, level=logging.WARNING) -> logging.Logger:
    """Returns a logger with the given name."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # If logger has handlers, do not add another to avoid duplicate logs, just set level
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    console = Console(force_jupyter=False)
    handler = RichHandler(
        rich_tracebacks=True, console=console, show_path=not is_notebook()
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(handler)
    return logger


log = get_logger(__name__)

# <<< Logging utils


# >>> General utils


def flatten(xs: list) -> list:
    """Returns a list of all elements in a nested iterable."""

    def _flatten(xs):
        """Returns a generator that flattens a nested iterable."""
        for x in xs:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from flatten(x)
            else:
                yield x

    return list(_flatten(xs))


def find_identical_rows(a: np.ndarray) -> list[list[int]]:
    """Finds identical rows in a 2D array."""
    a = np.asarray(a)
    idx = []
    for i in range(a.shape[0]):
        if i not in flatten(idx):
            idx.append([i])
        for j in range(i + 1, a.shape[0]):
            if np.allclose(a[i, :], a[j, :]):
                idx[-1].append(j)
    return idx


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


def suppress_output_legacy(func):
    """Suppresses the output of a function."""
    # NOTE: This approach only works when stdout/stderr are managed by Python

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


class TimeoutException(Exception):
    pass


def timeout(seconds):
    """Raises a TimeoutException if decorated function doesn't return in time."""

    def decorator(func):
        timeout_msg = f"{func.__name__} didn't converge in time!"

        def _handle_timeout(signum, frame):
            raise TimeoutException(timeout_msg)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutException:
                log.warning(timeout_msg)
                result = None
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


# <<< General utils


# >>> Circuit utils


def parse_initial_guess(
    p0: Union[np.ndarray, dict[str, float], list[float]],
    circuit: str,
) -> np.ndarray:
    """Parses the initial guess for circuit parameters."""
    num_params = parser.count_parameters(circuit)
    if p0 is None:
        return np.random.rand(num_params)
    elif isinstance(p0, dict):
        return np.fromiter(p0.values(), dtype=float)
    elif isinstance(p0, (list, np.ndarray)):
        return np.array(p0)
    raise ValueError(f"Invalid initial guess: {p0}")


def fit_circuit_parameters_legacy(
    circuit: str,
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    p0: Union[np.ndarray[float], dict[str, float]] = None,
    iters: int = 1,
    maxfev: int = 1000,
) -> dict[str, float]:
    """Fits a circuit to impedance data and returns the parameters."""
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
    p0: Union[np.ndarray[float], dict[str, float]] = None,
    iters: int = 1,
    maxfev: int = 1000,
    ftol: float = 1e-13,
) -> dict[str, float]:
    """Fits a circuit to impedance data and returns the parameters."""
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


# FIXME: Timeout logic doesn't work on Windows -> module 'signal' has no attribute 'SIGALRM'.
# if os.name != "nt":
#     fit_circuit_parameters = timeout(TIMEOUT_AFTER)(fit_circuit_parameters)
#     fit_circuit_parameters_legacy = timeout(TIMEOUT_AFTER)(fit_circuit_parameters_legacy)


def eval_circuit(
    circuit: str, f: Union[np.ndarray, float], p: np.ndarray
) -> np.ndarray[complex]:
    """Converts a circuit string to a function of (params, freq) and evaluates it."""
    Z_expr = parser.generate_mathematical_expr(circuit)
    return eval(Z_expr)


def generate_circuit_fn(circuit: str, jit=False, concat=False):
    def Z_complex(freq: np.ndarray, p: Union[np.ndarray, float]) -> np.ndarray[complex]:
        return eval_circuit(circuit, freq, p)

    def Z_concat(freq: np.ndarray, p: Union[np.ndarray, float]) -> np.ndarray:
        Z = Z_complex(freq, p)
        return jnp.hstack([Z.real, Z.imag])

    fn = Z_concat if concat else Z_complex
    fn = jax.jit(fn) if jit else fn

    return fn


def generate_circuit_fn_impedance_backend(circuit: str):
    """Converts a circuit string to a function using impedance.py."""
    num_params = parser.count_parameters(circuit)
    # Convert circuit string to impedance.py format
    circuit = parser.convert_to_impedance_format(circuit)
    # Convert circuit string to function
    p0 = np.full(num_params, np.nan)
    circuit = CustomCircuit(circuit, initial_guess=p0)

    def func(freq: Union[np.ndarray, float], p: np.ndarray) -> np.ndarray:
        circuit.parameters_ = p
        return circuit.predict(freq)

    return func


def circuit_complexity(circuit: str) -> list[int]:
    """Computes the component complexity of the circuit."""

    def increment(arr):
        """Add one to each element in a nested list."""
        return [increment(e) if isinstance(e, list) else e + 1 for e in arr]

    def depth(arr: list):
        """Compute the depth of each element in a nested list."""
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


def are_circuits_equivalent(circuit1: str, circuit2: str) -> bool:
    """Checks if two circuit strings are equivalent."""

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
    p0: dict[str, float], variables: list[str]
) -> dict[str, dist.Distribution]:
    """Initializes priors for a given circuit."""
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
    posterior: dict[str, np.ndarray[float]],
    variables: list[str],
    dist_type: str = "lognormal",
) -> dict[str, dist.Distribution]:
    """Creates new priors based on the posterior distributions."""
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


# <<< Statistics utils


# >>> Miscellaneous utils


def validate_circuits_dataframe(circuits: pd.DataFrame):
    """Validates the circuits dataframe format (columns and dtype)."""
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

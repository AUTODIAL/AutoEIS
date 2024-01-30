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

"""
import logging
import os
import re
import signal
import sys
from collections.abc import Iterable
from functools import wraps
from typing import Union

import jax  # NOQA: F401
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import rich.traceback
from impedance.models.circuits import CustomCircuit
from numpy import pi  # NOQA: F401
from rich.logging import RichHandler
from scipy import stats

# from tensorflow_probability import distributions as tfdist  # NOQA: F401
from autoeis import parser

# >>> Logging utils

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # If logger has handlers, do not add another to avoid duplicate logs
        return logger
    
    logger.setLevel(logging.WARNING)
    handler = RichHandler(rich_tracebacks=True)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(handler)
    return logger


def setup_rich_tracebacks():
    """Set up rich traceback for nicer exception printing."""
    rich.traceback.install()

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
        for j in range(i+1, a.shape[0]):
            if np.allclose(a[i, :], a[j, :]):
                idx[-1].append(j)
    return idx


class _SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def suppress_output(func):
    """Suppresses the output of a function."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        with _SuppressOutput():
            return func(*args, **kwargs)
    return wrapped


class TimeoutException(Exception):
    pass

def timeout(seconds):
    """Raises a TimeoutException if decorated function doesn't return in time."""
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException("Didn't converge in time!")

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutException:
                print("Didn't converge in time!")
                result = None
            finally:
                signal.alarm(0)
            return result

        return wrapper
    return decorator

# <<< General utils


# >>> Circuit utils

def fit_circuit_parameters(
    circuit: str,
    Z: np.ndarray[complex],
    freq: np.ndarray[float],
    p0: Union[np.ndarray[float], dict[str, float]] = None,
    iters: int = 1
) -> dict[str, float]:
    """Fits a circuit to impedance data and returns the parameters."""
    # Deal with initial guess
    num_params = parser.count_parameters(circuit)
    if p0 is None:
        p0 = np.random.rand(num_params)
    elif isinstance(p0, dict):
        p0 = list(p0.values())
    assert len(p0) == num_params, "Wrong number of parameters in initial guess."

    # Fit circuit parameters
    circuit_impy = CustomCircuit(
        circuit=parser.convert_to_impedance_format(circuit),
        initial_guess=p0
    )
    # HACK: Use multiple random initial guesses to avoid local minima
    err_min = np.inf
    for _ in range(iters):
        circuit_impy.fit(freq, Z)
        err = np.mean(np.abs(circuit_impy.predict(freq) - Z)**2)
        if err < err_min:
            err_min = err
            p0 = circuit_impy.parameters_
        circuit_impy.initial_guess = np.random.rand(num_params).tolist()

    labels = parser.get_parameter_labels(circuit)
    return dict(zip(labels, p0))

# FIXME: Timeout logic doesn't work on Windows -> module 'signal' has no attribute 'SIGALRM'.
if os.name != "nt":
    fit_circuit_parameters = timeout(300)(fit_circuit_parameters)


def generate_circuit_fn(
    circuit: str,
    return_str: bool = False,
    label: str = "X",
) -> Union[callable, str]:
    assert isinstance(circuit, str), "Circuit must be a string."
    """Converts a circuit string to a function of (params, freq)"""
    # Apply series-parallel conversion, e.g., [R1,R2] -> (1/R1+1/R2)**(-1)
    circuit_expr = parser.generate_mathematical_expr(circuit)
    # Embed impedance expressions, e.g., C1 -> (1/(2*1j*pi*F*C1))
    circuit_expr = parser.embed_impedance_expr(circuit_expr)
    # Replace variables with array indexing, e.g., R1, P2w, P2n -> X[0], X[1], X[2]
    variables = parser.get_parameter_labels(circuit)
    for i, var in enumerate(variables):
        circuit_expr = circuit_expr.replace(var, f"{label}[{i}]", 1)
    if return_str:
        return circuit_expr
    return lambda X, F: eval(circuit_expr)


def generate_circuit_fn_impedance_backend(circuit: str) -> callable:
    """Converts a circuit string to a function using impedance.py."""
    num_params = parser.count_parameters(circuit)
    # Convert circuit string to impedance.py format
    circuit = parser.convert_to_impedance_format(circuit)
    # Convert circuit string to function
    p0 = np.full(num_params, np.nan)
    circuit = CustomCircuit(circuit, initial_guess=p0)
    def func(params, freq):
        circuit.parameters_ = params
        return circuit.predict(freq)
    return func


def circuit_complexity(circuit: str) -> list[int]:
    """Computes the component complexity of the circuit."""
    def increment(arr):
        """Add one to each element in a nested list."""
        return [increment(e) if isinstance(e, list) else e+1 for e in arr]

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
    def x0(circuit:str) -> np.ndarray[float]:
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
    Z1 = generate_circuit_fn(circuit1)(x0(circuit1), freq)
    Z2 = generate_circuit_fn(circuit2)(x0(circuit2), freq)
    return np.allclose(Z1, Z2)

# <<< Circuit utils


# >>> Statistics utils

def initialize_priors(
    p0: dict[str, float],
    variables: list[str]
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
            priors[var] = dist.TruncatedNormal(loc=loc, scale=1*scale, low=0, high=1)
        # Fit data to a log-normal distribution for all other parameters
        else:
            # NOTE: s and scale in scipy.stats -> scale and np.exp(loc) in numpyro
            # NOTE: above conversion is only valid when loc = 0
            if dist_type == "lognormal":
                s, loc, scale = stats.lognorm.fit(samples, floc=0)
                priors[var] = dist.LogNormal(loc=np.log(scale), scale=8*s)
            elif dist_type == "normal":
                loc, scale = stats.norm.fit(samples)
                priors[var] = dist.TruncatedNormal(loc=loc, scale=1*scale, low=0, high=np.inf)
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

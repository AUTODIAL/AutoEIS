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
    identify_duplicate_circuits
    initialize_priors
    initialize_priors_from_posteriors
    eval_posterior_predictive
    validate_circuits_dataframe
    preprocess_impedance_data
    Settings
    InferenceResult
    ImpedanceData

"""

import copy
import functools
import io
import logging
import os
import re
import sys
import warnings
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
from box import Box
from impedance.models.circuits import CustomCircuit
from impedance.validation import linKK
from jax import random
from mpire import WorkerPool
from numpy import pi  # NOQA: F401
from numpy.linalg import norm
from numpyro.distributions import Distribution
from numpyro.infer import MCMC, Predictive
from scipy import stats
from scipy.optimize import least_squares
from scipy.stats import loguniform
from scipy.stats.mstats import gmean
from tqdm.auto import tqdm

import __main__

from . import metrics, models, parser, visualization

log = logging.getLogger(__name__)


# >>> General utils


class DivergenceError(Exception):
    pass


def flush_streams():
    """Flushes the standard output and error streams."""
    sys.stdout.flush()
    sys.stderr.flush()


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
    """Global settings, e.g., logging, parallelism, etc."""

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


def is_ndfarray_like(xs: Iterable) -> bool:
    """Returns True if the input is an ndarray-like object with float elements."""
    try:
        np.asfarray(xs)
    except (ValueError, TypeError):
        return False
    return True


def is_iterable(xs: Iterable) -> bool:
    """Returns true if the input is an iterable but not a string or bytes.

    Parameters
    ----------
    xs: Iterable
        An iterable.

    Returns
    -------
    bool
        True if the input is an iterable but not a string or bytes, False otherwise.

    Examples
    --------
    >>> is_iterable([1, 2, 3])
    True
    >>> is_iterable("hello")
    False
    """
    return isinstance(xs, Iterable) and not isinstance(xs, (str, bytes))


def is_nested_iterable(xs: Iterable) -> bool:
    """Returns True if all items of an iterable are iterable themselves.

    Parameters
    ----------
    xs: Iterable
        An iterable.

    Returns
    -------
    bool
        True if the iterable is nested, False otherwise.

    Examples
    --------
    >>> is_nested_iterable([1, 2, [3, 4], [5, [6, 7]]])
    False
    >>> is_nested_iterable([1, 2, 3, 4])
    False
    >>> is_nested_iterable([[1, 2], [3, 4], [5, 6]])
    True
    """
    return all(is_iterable(x) for x in xs)


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
    """Suppresses stdout and stderr for both Unix and Windows systems."""
    # Save the original high-level streams
    saved_stderr = sys.stderr
    saved_stdout = sys.stdout

    try:
        # Try to save the current file descriptors
        original_stderr_fd = sys.stderr.fileno()
        original_stdout_fd = sys.stdout.fileno()
        saved_stderr_fd = os.dup(original_stderr_fd)
        saved_stdout_fd = os.dup(original_stdout_fd)

        with open(os.devnull, "wb") as devnull:
            # Redirect the lower-level file descriptors
            os.dup2(devnull.fileno(), original_stderr_fd)
            os.dup2(devnull.fileno(), original_stdout_fd)

        # Redirect the higher-level Python streams
        sys.stderr = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")

    except (io.UnsupportedOperation, AttributeError):
        # If fileno is not supported, just replace the Python streams
        sys.stderr = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")

    try:
        yield
    finally:
        # Restore the high-level Python streams
        sys.stderr.close()
        sys.stdout.close()
        sys.stderr = saved_stderr
        sys.stdout = saved_stdout

        # Restore the original file descriptors if they were saved
        if "saved_stderr_fd" in locals() and "saved_stdout_fd" in locals():
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.close(saved_stderr_fd)
            os.close(saved_stdout_fd)


# <<< General utils


# >>> Circuit utils


def parse_initial_guess(
    p0: Mapping[str, float] | Iterable[float],
) -> np.ndarray:
    """Parses the initial guess for circuit parameters into an ndarray.

    Parameters
    ----------
    p0: Mapping[str, float] | Iterable[float]
        The initial guess for the circuit parameters.

    Returns
    -------
    np.ndarray
        The array of initial guesses.

    Raises
    ------
    ValueError
        If the initial guess is not not a dict nor array-like.
    """
    if isinstance(p0, dict):
        return np.fromiter(p0.values(), dtype=float)
    if is_ndfarray_like(p0):
        return np.array(p0)
    raise ValueError(f"Invalid initial guess: {p0}")


def generate_initial_guess(circuit: str, seed=None, log=True) -> np.ndarray:
    """Generates a random initial guess for the circuit parameters.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    seed : int, optional
        Seed for the random number generator. Default is None.
    log : bool, optional
        If True, samples the parameters in log-space. Default is True.

    Returns
    -------
    np.ndarray
        A random initial guess for the circuit parameters.
    """
    num_params = parser.count_parameters(circuit)
    if seed is not None:
        np.random.seed(seed)
    if not log:
        return np.random.rand(num_params)
    # Sample in log-space for better coverage ~ 1e-9 to 1
    p = loguniform(1e-9, 1).rvs(num_params)
    # Except for Pn, which is sampled from 0 to 1
    ptypes = parser.get_parameter_types(circuit)
    for i, ptype in enumerate(ptypes):
        p[i] = np.random.rand() if ptype == "Pn" else p[i]
    return p


def get_parameter_bounds(circuit: str) -> tuple:
    """Returns a 2-element tuple of lower and upper bounds, to be used in
    SciPy's ``least_squares``.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    tuple
        A 2-element tuple of lower and upper bounds for the circuit parameters.
    """
    # NOTE: Using np.inf actually works better than physical bounds!
    bounds_dict = {
        "R": (0.0, np.inf),  # 1e9
        "C": (0.0, np.inf),  # 10.0
        "Pw": (0.0, np.inf),  # 1e9
        "Pn": (0.0, np.inf),  # 1.0
        "L": (0.0, np.inf),  # 5.0
    }
    types = parser.get_parameter_types(circuit)
    bounds = [bounds_dict[type_] for type_ in types]
    bounds = tuple(zip(*bounds))
    return bounds


def fit_circuit_parameters_legacy(
    circuit: str,
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    p0: Mapping[str, float] | Iterable[float] = None,
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
    p0 : Mapping[str, float] | Iterable[float], optional
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
    p0 = parse_initial_guess(p0) if p0 is not None else generate_initial_guess(circuit)
    num_params = parser.count_parameters(circuit)
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
    p0: Mapping[str, float] | Iterable[float] = None,
    max_iters: int = 50,
    min_iters: int = 25,
    bounds: Iterable[tuple] = None,
    max_nfev: int = None,
    ftol: float = 1e-15,
    xtol: float = 1e-15,
    tol_chi_squared: float = 1e-4,
    method: str = "bode",
    verbose: bool = False,
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
    p0 : Mapping[str, float] | Iterable[float], optional
        Initial guess for the circuit parameters. Default is None.
    max_iters : int, optional
        Maximum number of iterations for the circuit fitter. Default is 10.
    min_iters : int, optional
        Minimum number of iterations for the circuit fitter. Default is 5.
        If ``min_iters`` is reached AND circuit fitter converges, the fitting
        process stops.
    bounds : Iterable[tuple], optional
        List of two tuples, each containing the lower and upper bounds,
        respectively, for the circuit parameters. Default is None. The order
        of the values should match the order of the circuit parameters as
        returned by ``parser.get_parameter_labels``.
    maxfev : int, optional
        Maximum number of function evaluations for the circuit fitter.
        Default is None. See ``scipy.optimize.leastsq`` for details.
    ftol : float, optional
        See ``scipy.optimize.leastsq`` for details. Default is 1e-8.
    xtol : float, optional
        See ``scipy.optimize.leastsq`` for details. Default is 1e-8.
    tol_chi_squared : float, optional
        Tolerance for the chi-squared error. This only gets triggered if
        ``min_iters`` is set. A good chi-squared value is 1e-3 or smaller.
        Default is 1e-3.
    method : str, optional
        Method to use for fitting. Choose from 'chi-squared', 'nyquist',
        'bode', and 'magnitude'. The objective function is different for each
        method:

          * 'chi-squared':
            ``w * ((Re(Zp) - Re(Z)) ** 2 + (Im(Zp) - Im(Z)) ** 2)`` where
            ``w = 1 / (Re(Z)**2 + Im(Z)**2)``
          * 'nyquist': ``[Re(Zp) - Re(Z), Im(Zp) - Im(Z)]``
          * 'bode': ``[log10(mag / mag_gt), phase - phase_gt]``
          * 'magnitude': ``abs(Z - Zp)``

        Default is 'bode'. ``Zp`` is the predicted impedance and ``Z`` is
        ground truth. ``mag`` and ``phase`` are the magnitude and phase of the
        predicted impedance, and finally ``_gt`` denotes ground truth values.

    verbose : bool, optional
        If True, prints the fitting results. Default is False.

    Returns
    -------
    dict[str, float]
        Fitted parameters as a dictionary of parameter names and values.

    Notes
    -----
    This function uses SciPy's ``least_squares`` to fit the circuit parameters.
    """

    def obj_chi_squared(p):
        """Computes ECM error based on residual-based χ2."""
        Zp = fn(freq, p)
        residual = (Zp.real - Z.real) ** 2 + (Zp.imag - Z.imag) ** 2
        weight = 1 / (Z.real**2 + Z.imag**2)
        return residual * weight

    def obj_nyquist(p):
        """Computes ECM error based on the Nyquist plot."""
        Z_pred = fn(freq, p)
        res = jnp.hstack((Z_pred.real - Z.real, Z_pred.imag - Z.imag))
        return res

    def obj_bode(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)
        mag = jnp.abs(Z_pred)
        phase = jnp.angle(Z_pred)
        res = jnp.hstack((jnp.log10(mag / mag_gt), phase - phase_gt))
        # res = jnp.hstack((mag - mag_gt, phase - phase_gt))
        return res

    def obj_magnitude(p):
        """Computes ECM error based on the magnitude of impedance deviation."""
        Z_pred = fn(freq, p)
        res = jnp.abs(Z - Z_pred)
        return res

    def obj_phase(p):
        """Computes ECM error based on the phase of impedance deviation."""
        Z_pred = fn(freq, p)
        res = jnp.abs(jnp.angle(Z) - jnp.angle(Z_pred))
        return res

    msg = f"Invalid method: {method}. Use 'chi-squared', 'nyquist', 'bode', 'magnitude', or 'phase'."
    assert method in ["chi-squared", "nyquist", "bode", "magnitude", "phase"], msg
    assert len(freq) == len(Z), "Length of frequency and impedance data must match."

    variables = parser.get_parameter_labels(circuit)
    fn = generate_circuit_fn(circuit, jit=True)
    obj = {
        "nyquist": obj_nyquist,
        "bode": obj_bode,
        "chi-squared": obj_chi_squared,
        "magnitude": obj_magnitude,
        "phase": obj_phase,
    }[method]

    mag_gt = jnp.abs(Z)
    phase_gt = jnp.angle(Z)

    # Sanitize initial guess
    p0 = parse_initial_guess(p0) if p0 is not None else generate_initial_guess(circuit)
    num_params = parser.count_parameters(circuit)
    assert len(p0) == num_params, "Wrong number of parameters in initial guess."

    # Assemble kwargs for curve_fit
    bounds = get_parameter_bounds(circuit) if bounds is None else bounds
    kwargs = {"x0": p0, "bounds": bounds, "max_nfev": max_nfev, "ftol": ftol, "xtol": xtol}

    # Ensure p0 is not out-of-bounds
    if p0 is not None:
        for i, (lower, upper) in enumerate(zip(*bounds)):
            p0[i] = np.clip(p0[i], lower, upper)

    # Fit circuit parameters by brute force
    min_iters = max_iters if min_iters is None else min_iters
    err_min = np.inf

    for i in tqdm(
        range(max_iters), desc="Fitting ECM Parameters", disable=not verbose, leave=False
    ):
        # HACK: Occasionally, least_squares throws ValueError
        try:
            res = least_squares(obj, verbose=False, **kwargs)
        except ValueError:
            kwargs["x0"] = generate_initial_guess(circuit)
            continue
        if (err := norm(obj(res.x))) < err_min:
            err_min = err
            p0 = res.x
        converged = (X2 := obj_chi_squared(res.x).mean()) < tol_chi_squared
        if i + 1 >= min_iters and converged:
            break
        kwargs["x0"] = generate_initial_guess(circuit)

    converged = True if err_min != np.inf else False

    r2_mag = metrics.r2_score(jnp.abs(Z), jnp.abs(fn(freq, p0)))
    r2_phase = metrics.r2_score(jnp.angle(Z), jnp.angle(fn(freq, p0)))
    X2 = np.mean(obj_chi_squared(p0))
    log.info(
        f"Converged in {i+1} iterations with "
        f"𝛘² = {X2:.3e}, R² (|Z|) = {r2_mag:.4f}, R² (phase) = {r2_phase:.4f}"
    )

    if err_min == np.inf:
        raise DivergenceError(
            "Failed to fit the circuit parameters. Try increasing 'iters' or "
            "'maxfev', or narrow down the search by providing 'bounds'."
        )

    return dict(zip(variables, p0))


def eval_circuit(
    circuit: str, freq: np.ndarray | float, p: np.ndarray, jit: bool = False
) -> np.ndarray[complex]:
    """Returns the impedance of a circuit at a given frequency and parameters.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    freq : np.ndarray | float
        Frequencies at which to evaluate the circuit.
    p : np.ndarray
        Circuit parameters.

    Returns
    -------
    np.ndarray[complex]
        The impedance of the circuit at the given frequency and parameters.
    """
    expr = parser.generate_mathematical_expression(circuit)
    # For frequency-independent circuits, ensure output is the same shape as freq
    if not np.isscalar(freq):
        freq_like_ones = "jnp.ones(len(freq))" if jit else "np.ones(len(freq))"
        expr = f"({expr}) * {freq_like_ones}"
    return eval(expr)


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

    def fn_complex(freq: np.ndarray, p: np.ndarray) -> np.ndarray[complex]:
        atleast_1d = jnp.atleast_1d if jit else np.atleast_1d
        freq, p = atleast_1d(freq), atleast_1d(p)
        msg = f"The parameters: {p} don't match the number of parameters in the circuit: {circuit}."
        assert len(p) == parser.count_parameters(circuit), msg
        return eval_circuit(circuit, freq, p, jit=jit)

    def fn_concat(freq: np.ndarray, p: np.ndarray) -> np.ndarray[complex]:
        Z = fn_complex(freq, p)
        hstack = jnp.hstack if jit else np.hstack
        return hstack([Z.real, Z.imag])

    fn = fn_concat if concat else fn_complex
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

    def increment(arr: Iterable):
        """Adds one to each element in a nested list."""
        return [increment(e) if isinstance(e, list) else e + 1 for e in arr]

    def depth(arr: Iterable):
        """Computes the depth of each element in a nested list."""
        return [increment(depth(e)) if isinstance(e, list) else 0 for e in arr]

    def split(arr: Iterable, chars: Iterable[str]):
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
    # Ensure that the circuits are valid
    parser.validate_circuit(circuit1)
    parser.validate_circuit(circuit2)

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
    return np.allclose(Z1, Z2, rtol=rtol)


def find_duplicate_circuits(circuits: Iterable[str], rtol: float = 1e-5) -> list[list[int]]:
    """Returns the indices of duplicate circuits given a list of circuit strings.

    Parameters
    ----------
    circuits : Iterable[str]
        List of circuit strings in CDC format. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    rtol : float, optional
        The relative tolerance for the circuit equivalence check. Default is 1e-5.

    Returns
    -------
    list[list[int]]
        A list of lists containing the indices of duplicate circuits. The first
        element of each sublist is the index of the circuit that is considered
        unique, and the rest are the indices of the duplicate circuits.
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

    # Validate input circuits
    assert isinstance(circuits, (list, tuple)), "Input must be a list[str] or tuple[str]."

    freq = np.logspace(-3, 3, 10)
    Z = [generate_circuit_fn(circuit)(freq, x0(circuit)) for circuit in circuits]
    y = np.abs(gmean(Z, axis=1))
    y_unique = np.unique(y)
    indices = [np.where(y == v)[0] for v in y_unique]
    indices = sorted(indices, key=lambda elem: elem[0])  # Retain original order
    return indices


# <<< Circuit utils


# >>> Inference utils


def initialize_priors(p0: Mapping[str, float]) -> dict[str, Distribution]:
    """Initializes prior distributions from parameters dictionary.

    Parameters
    ----------
    p0 : Mapping[str, float]
        Initial guess for the circuit parameters as a dictionary of parameter
        names and values.

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
    # Get parameter labels; exclude MCMC-specific parameters, e.g., sigma_real, etc.
    variables = [k for k in p0.keys() if parser.validate_parameter(k, raises=False)]
    for var in variables:
        value = p0[var]
        if "n" in var:
            # TODO: use a more informative prior for n, eg truncated normal
            # Exponent of CPE elements is bounded between 0 and 1
            priors[var] = dist.Uniform(0, 1)
        else:
            # Search over a log-normal dist spanning [0.01*u0, 100*u0]
            mean, std_dev = jnp.log(value), jnp.log(100)
            priors[var] = dist.LogNormal(mean, std_dev)
    return priors


def initialize_priors_from_posteriors(
    posterior: Mapping[str, np.ndarray[float]],
    dist_type: str = "lognormal",
) -> dict[str, Distribution]:
    """Creates new priors based on the posterior distributions.

    Parameters
    ----------
    posterior : Mapping[str, np.ndarray[float]]
        Posterior distributions for the circuit parameters as a dictionary
        of parameter names and distributions.
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
    # Get parameter labels; exclude MCMC-specific parameters, e.g., sigma_real, etc.
    variables = [k for k in posterior.keys() if parser.validate_parameter(k, raises=False)]
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
                priors[var] = dist.LogNormal(loc=np.log(scale), scale=4 * s)
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
    # Make a deep copy of the priors to avoid side effects (pyro-ppl/numpyro/issues/1651)
    return copy.deepcopy(priors)


def eval_posterior_predictive(
    samples: Mapping[str, np.ndarray],
    circuit: str,
    freq: np.ndarray[float],
    priors: Mapping[str, Distribution] = None,
    method: str = "bode",
    rng_key: random.PRNGKey = None,
) -> np.ndarray[complex]:
    """Evaluates the posterior predictive distribution of a MCMC run.

    Parameters
    ----------
    samples: Mapping[str, np.ndarray]
        Samples from the MCMC run as a dictionary of parameter names and arrays.
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    freq : np.ndarray[float]
        Frequencies to evaluate the posterior predictive distribution at.
    priors : Mapping[str, Distribution], optional
        Priors for the circuit parameters as a dictionary of parameter names
        and distributions. Default is None.
    method : str, optional
        Objective function that was used for inference. Default is "bode".
        Options are "bode", "nyquist", "magnitude", and "chi-squared".
    rng_key : random.PRNGKey, optional
        Random key for the MCMC run. Default is None.

    Returns
    -------
    np.ndarray[complex]
        Posterior predictive distribution of the circuit at the given frequencies.
    """
    rng_key = rng_key or random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    circuit_fn = generate_circuit_fn(circuit, jit=True)

    # TODO: priors is a dummy variable for posterior predictive
    # Deal with default arguments
    if priors is None:
        variables = parser.get_parameter_labels(circuit)
        p0 = {var: np.median(samples[var]) for var in variables}
        priors = initialize_priors(p0)

    # Create a predictive distribution for the circuit parameters
    method = method.replace("-", "_")
    model = getattr(models, f"circuit_regression_{method}")
    predictive = Predictive(model, samples)

    # Evaluate the predictive distribution at the given frequency
    kwargs = {"freq": freq, "priors": priors, "circuit_fn": circuit_fn}
    predictions = predictive(rng_key_, **kwargs)

    # Handle inference models where real/imag parts are not explicitly predicted
    if method in ["chi-squared", "magnitude"]:
        raise NotImplementedError(f"'method={method}' is not implemented yet.")
    if method == "bode":
        mag, phase = predictions["obs.mag"], predictions["obs.phase"]
        predictions["obs.real"] = mag * jnp.cos(phase)
        predictions["obs.imag"] = mag * jnp.sin(phase)

    Z_pred = predictions["obs.real"] + predictions["obs.imag"] * 1j

    return Z_pred


class InferenceResult:
    """Container for inference result."""

    def __init__(
        self,
        circuit: str,
        mcmc: MCMC,
        *,
        converged: bool,
        freq: np.ndarray[float],
        Z: np.ndarray[complex],
    ):
        self.circuit = circuit
        self.mcmc = mcmc
        self.converged = converged
        self.freq = freq
        self.Z = Z

    def __repr__(self):
        return f"InferenceResult at {id(self):#x}"

    @functools.cached_property
    def variables(self):
        """Returns the inferred variables, i.e., circuit parameters."""
        return parser.get_parameter_labels(self.circuit)

    @functools.cached_property
    def samples(self):
        """Returns the MCMC samples."""
        return self.mcmc.get_samples()

    @functools.cached_property
    def num_divergences(self):
        """Returns the number of divergences in the MCMC chain."""
        return self.mcmc.get_extra_fields()["diverging"].sum()

    def print_summary(self):
        """Prints a summary of the inference results."""
        self.mcmc.print_summary()


class ImpedanceData:
    """Container for impedance data."""

    def __init__(self, freq: np.ndarray[float], Z: np.ndarray[complex]):
        assert len(freq) == len(Z), "freq and Z data must have the same length."
        self.freq = freq
        self.Z = Z

    def __len__(self):
        return len(self.freq)

    def plot_nyquist(self, ax=None, **kwargs):
        """Plots the Nyquist plot of the impedance data."""
        return visualization.plot_nyquist(self.Z, ax=ax, **kwargs)

    def plot_bode(self, ax=None, **kwargs):
        """Plots the Bode plot of the impedance data."""
        return visualization.plot_bode(self.freq, self.Z, ax=ax, **kwargs)

    def preprocess(
        self, tol_linKK=5e-2, high_freq_threshold=1e3, return_aux=False, inplace=True
    ):
        """Preprocesses the impedance data."""
        out = preprocess_impedance_data(
            self.freq,
            self.Z,
            tol_linKK=tol_linKK,
            high_freq_threshold=high_freq_threshold,
            return_aux=return_aux,
        )
        if inplace:
            self.freq, self.Z = out[:2]
        return out


# <<< Inference utils


# >>> Miscellaneous utils


def preprocess_impedance_data(
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    tol_linKK: float = 5e-2,
    high_freq_threshold: float = 1e3,
    return_aux: bool = False,
) -> tuple[np.ndarray[float], np.ndarray[complex], Box]:
    """Preprocesses/cleans up impedance measurements.

    The preprocessing does the following steps:
        - Discard invalid high frequency measurements (see Notes section)
        - Filter out data with a positive imaginary part in high frequencies
        - Enforce the Kramers-Kronig validation (aka Lin-KK)

    Parameters
    ----------
    freq : np.ndarray[float]
        Frequencies corresponding to impedance measurements.
    Z : np.ndarray[complex]
        Impedance measurements as a complex array.
    tol_linKK : float
        Tolerance for acceptable measurements based on linKK residuals.
    high_freq_threshold : float
        Lower bound for what is considered a high frequency measurement.
    return_aux : bool, optional
        If True, returns the preprocessed data along with auxiliary
        information. Default is False.

    Returns
    -------
    tuple[np.ndarray[float], np.ndarray[complex], Box]
        Tuple containing the preprocessed data with the following elements:
            - freq: Frequencies corresponding to the impedance data.
            - Z: Filtered impedance data.
            - aux: Box containing the Lin-KK validation results with keys:
                - res.real: Residual array for real part of the impedance data.
                - res.imag: Residual array for imaginary part of the impedance data.
                - rmse: Root mean square error of KK validated data vs. measurements.
    """
    log.info("Preprocessing/cleaning up impedance data.")
    n0 = len(freq)

    # Make sure frequency is sorted in descending order (needed in Heuristic 1)
    Z = Z[np.argsort(freq)[::-1]]
    freq = freq[np.argsort(freq)[::-1]]

    # Heuristic 1: @freq->∞: |Z.im|->0 => highest_valid_freq = freq @ np.argmin(|Z.im|)
    if (high_freq := freq > high_freq_threshold).any():
        idx_highest_valid_freq = np.argmin(np.abs(Z.imag[high_freq]))
        # Filter out frequencies above the highest valid frequency
        freq = freq[idx_highest_valid_freq:]
        Z = Z[idx_highest_valid_freq:]

    # Heuristic 2: Remove the data whose Z.imag is positive at high frequencies
    # NOTE: Need to redefine high_freq since freq might have changed in Heuristic 1
    if (high_freq := freq > high_freq_threshold).any():
        positive_im = Z.imag > 0
        mask = high_freq & positive_im
        Z = Z[~mask]
        freq = freq[~mask]

    # Heuristic 3: Kramers-Kronig validation (aka Lin-KK)
    linKK_kwargs = {"c": 0.5, "max_M": 100, "fit_type": "complex", "add_cap": True}
    # TODO: Suppress output until ECSHackWeek/impedance.py/issues/280 is fixed
    linKK_silent = suppress_output_legacy(linKK)
    M, mu, Z_linKK, res_real, res_imag = linKK_silent(freq, Z, **linKK_kwargs)
    rmse = metrics.rmse_score(Z, Z_linKK)

    # NOTE: Attach `freq` in case user wants to plot the residuals (b/c after linKK, freq might change)
    aux = Box(freq=freq, res=Box(real=res_real, imag=res_imag), rmse=rmse)

    mask = (np.abs(res_real) < tol_linKK) & (np.abs(res_imag) < tol_linKK)
    freq = freq[mask]
    Z = Z[mask]

    if (frac_filtered := 1 - len(freq) / n0) > 0.1:
        log.warning(f"{frac_filtered * 100:.0f}% of data filtered out.")

    return (freq, Z, aux) if return_aux else (freq, Z)


# TODO: Add support for kwargs
def distribute_task(
    func,
    *args,
    static: int | Iterable[int] = None,
    iters: int = None,
    n_jobs: int = None,
    progress_bar: bool = True,
    desc: str = "",
    handle_errors: bool = True,
):
    """Distribute workload across multiple processes."""
    static = [static] if isinstance(static, int) else static
    # Infer the number of iterations from *args
    if iters is None:
        iters = [
            len(arg)
            for i, arg in enumerate(args)
            if i not in static and hasattr(arg, "__len__")
        ]
        if np.unique(iters).size == 0:
            raise RuntimeError("Couldn't infer `iters` from args, specify `iters`")
        assert np.unique(iters).size == 1, "All iterable arguments must have the same length"
        iters = iters[0]

    args = list(args)
    for i, arg in enumerate(args):
        args[i] = [arg] * iters if i in static else arg
        assert len(args[i]) == iters, "Make sure 'static' contains all the static args"

    n_jobs = min(iters, n_jobs or psutil.cpu_count(logical=False))
    mpire_kwargs = {
        "progress_bar": progress_bar,
        "progress_bar_style": "notebook" if is_notebook() else "rich",
        "progress_bar_options": {"desc": desc},
        "iterable_len": iters,
        "concatenate_numpy_output": False,
    }

    def _func(*args):
        try:
            return func(*args)
        except Exception as e:
            return e

    _func = _func if handle_errors else func

    with warnings.catch_warnings():
        # JAX doesn't work well with multiprocessing, but "spawn" should be fine
        msg_to_ignore = ".*os\\.fork\\(\\).*"
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=msg_to_ignore)
        with WorkerPool(n_jobs=n_jobs, use_dill=True, start_method="spawn") as pool:
            results = pool.map(_func, zip(*args), **mpire_kwargs)

    return results


def validate_circuits_dataframe(circuits: pd.DataFrame):
    """Ensures that the circuits dataframe if properly formatted/typed.

    Specifically, this function ensures that:
        - column names are valid (must be ``circuitstring``, ``Parameters``),
        - column data types are valid, i.e., ``circuitstring`` and
          ``Parameters`` must contain strings and dictionaries, respectively.

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
    ), "circuitstring column must only contain strings."
    # Check if the Parameters column contains only dictionaries
    assert (
        circuits["Parameters"].apply(lambda x: isinstance(x, dict)).all()
    ), "Parameters column must only contain dictionaries."


# <<< Miscellaneous utils

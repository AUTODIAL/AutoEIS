"""
Collection of utility functions used throughout the package.

.. currentmodule:: autoeis.utils

.. autosummary::
   :toctree: generated/

    get_logger
    setup_rich_tracebacks
    suppress_output

"""
import logging
import os
import re
import sys
from collections.abc import Iterable
from functools import wraps
from typing import Union

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import rich.traceback
from impedance.models.circuits import CustomCircuit
from numpy import pi
from pyparsing import nested_expr
from rich.logging import RichHandler
from scipy.stats import lognorm, norm

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


def setup_rich_tracebacks() -> None:
    """Set up rich traceback for nicer exception printing."""
    rich.traceback.install()

# <<< Logging utils


# >>> Filesystem utils

def flatten(xs):
    """Returns a list of all elements in a nested iterable."""
    def _flatten(xs):
        """Returns a generator that flattens a nested iterable."""
        for x in xs:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from flatten(x)
            else:
                yield x
    return list(_flatten(xs))


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
    @wraps(func)
    def wrapped(*args, **kwargs):
        with _SuppressOutput():
            return func(*args, **kwargs)
    return wrapped

# <<< Filesystem utils


# >>> Circuit utils

def _get_component_labels(circuit: str, component_type: str) -> list[str]:
    """Returns a list of labels for all components of a given type in a circuit string."""
    return re.findall(rf"{component_type}\d+", circuit)


def get_resistors(circuit: str) -> list[str]:
    """Returns a list of labels for all resistors in a circuit string."""
    return _get_component_labels(circuit, "R")


def get_capacitors(circuit: str) -> list[str]:
    """Returns a list of labels for all capacitors in a circuit string."""
    return _get_component_labels(circuit, "C")


def get_inductors(circuit: str) -> list[str]:
    """Returns a list of labels for all inductors in a circuit string."""
    return _get_component_labels(circuit, "L")


def get_cpes(circuit: str) -> list[str]:
    """Returns a list of labels for all CPEs in a circuit string."""
    if "CPE" in circuit:
        return _get_component_labels(circuit, "CPE")
    return _get_component_labels(circuit, "P")


def get_fsws(circuit: str) -> list[str]:
    """Returns a list of labels for all FSWs in a circuit string."""
    return _get_component_labels(circuit, "Wo")


def get_component_labels(circuit: str, types="all") -> list[str]:
    """Returns a list of labels for all components in a circuit string."""
    pattern = r"[A-Za-z]+[0-9]+" if types == "all" else rf"[{types}][0-9]+"
    return re.findall(pattern, circuit)


def get_component_types(circuit: str) -> list[str]:
    """Returns a list of component types in a circuit string."""
    return re.findall(r"[A-Za-z]+", circuit)


def get_parameter_labels(circuit: str, types="RLCP") -> list[str]:
    """Returns a list of labels for all parameters in a circuit string."""
    labels = get_component_labels(circuit, types=types)
    param_labels = []
    for label in labels:
        # CPE elements have two parameters P{i}w and P{i}n
        if label.startswith("P"):
            param_labels.extend([f"{label}w", f"{label}n"])
        else:
            param_labels.append(label)
    return param_labels


def count_params(circuit: str) -> int:
    """Returns the number of parameters that fully describe a circuit string."""
    num_resistors = len(get_resistors(circuit))
    num_capacitors = len(get_capacitors(circuit))
    num_inductors = len(get_inductors(circuit))
    num_cpes = len(get_cpes(circuit))
    num_fsws = len(get_fsws(circuit))
    return num_resistors + num_capacitors + num_inductors + 2 * (num_cpes + num_fsws)


def impedancepy_circuit(circuit: str) -> str:
    """Converts a circuit string the format used by impedance.py."""
    circuit = circuit.replace("P", "CPE")
    circuit = circuit.replace("[", "p(")
    circuit = circuit.replace("]", ")")
    return circuit


def format_parameters(params, labels):
    """Formats a list of parameters and labels to be used in circuits dataframe."""
    # Example: R1 = 1, C1 = 2 -> "(R1 = 1.0, C1 = 2.0)"
    pairs = [f"{label} = {value:.10f}" for label, value in zip(labels, params)]
    pairs = "(" + ", ".join(pairs) + ")"
    return pairs


def parse_circuit_dataframe(circuits: pd.DataFrame) -> pd.DataFrame:
    """Replaces the Parameters column in EquivalentCircuits.jl from a
    stringified list to a proper dict[str, float]: "R1 = 1" -> {"R1": 1}.
    """
    circuit_str = []
    params_dict = []

    for row in circuits.itertuples():
        circuit_str.append(row.circuitstring)
        pstr = row.Parameters
        # Remove parentheses and spaces, then split by comma -> list[var=val, ...]
        pstr = pstr.strip("()").replace(" ", "")
        pstr = pstr.split(",")
        # Extract variable names and values into a dictionary
        pdict = {pair.split("=")[0]: float(pair.split("=")[1]) for pair in pstr}
        params_dict.append(pdict)
    
    return pd.DataFrame({"circuitstring": circuit_str, "Parameters": params_dict})


def fit_circuit_parameters(
    circuit: str,
    Z: np.ndarray[complex],
    freq: np.ndarray[float],
    p0: Union[np.ndarray[float], dict[str, float]] = None,
) -> dict[str, float]:
    """Fits a circuit to impedance data and returns the parameters."""
    num_params = count_params(circuit)
    # Deal with initial guess
    p0 = np.random.rand(num_params) if p0 is None else p0
    p0 = list(p0.values()) if isinstance(p0, dict) else p0
    assert len(p0) == num_params, "Wrong number of parameters in initial guess."
    # Use initial guess if provided, otherwise use random values
    labels = get_parameter_labels(circuit)
    circuit = CustomCircuit(
        circuit=impedancepy_circuit(circuit),
        initial_guess=p0
    )
    circuit.fit(freq, Z)
    params = circuit.parameters_
    return dict(zip(labels, params))


def circuit_to_function_impy(circuit: str):
    """Converts a circuit string to a function using impedance.py."""
    num_params = count_params(circuit)
    # Convert circuit string to impedance.py format
    circuit = impedancepy_circuit(circuit)
    # Convert circuit string to function
    p0 = np.full(num_params, np.nan)
    circuit = CustomCircuit(circuit, initial_guess=p0)
    def func(params, freq):
        circuit.parameters_ = params
        return circuit.predict(freq)
    return func


def circuit_to_nested_expr(circuit: str) -> list:
    """Parses a circuit string to a nested list[str]."""
    # Add brackets to the circuit string to make it a valid nested expression
    circuit = f"[{circuit}]"
    parser = nested_expr(opener="[", closer="]")
    parsed = parser.parse_string(circuit, parse_all=True).as_list()
    # Remove leftover cruft from the parsed expression
    parsed = cleanup_nested_expr(parsed)
    return parsed[0]


def cleanup_nested_expr(lst, chars="-,"):
    """Removes leading/trailing characters ('-' ',') from a nested list[str]."""
    result = []
    for el in lst:
        if isinstance(el, list):
            result.append(cleanup_nested_expr(el))
        else:
            # Don't add empty strings
            if el.strip(chars):
                result.append(el.strip(chars))
    return result


def find_ohmic_resistors(circuit: list) -> list[str]:
    """Finds all ohmic resistors in a nested circuit expression."""
    parsed = circuit_to_nested_expr(circuit)
    series_elements = [el for el in parsed if isinstance(el, str)]
    return re.findall(r"R\d+", str(series_elements))


def validate_circuit_string(circuit: str) -> bool:
    """Checks if a circuit string is valid."""
    # Check for duplicate elements
    params = get_parameter_labels(circuit)
    assert len(params) == len(set(params)), "Duplicate elements found."


def generate_mathematical_expr(circuit:str) -> str:
    """Converts a circuit string to a mathematical expression.
    
    Each variable in the expression is the impedance of the
    corresponding component.
    """
    replacements = {
        "-": "+",
        "[": "((",
        ",": ")**(-1)+(",
        "]": ")**(-1))**(-1)"
    }
    for j, k in replacements.items():
        circuit = circuit.replace(j, k)
    return circuit


def count_component_parameters(component: str) -> int:
    """Count the number of parameters in a component label."""
    count = {"R": 1, "C": 1, "L": 1, "P": 2}
    eltype = re.match(r"[A-Z]+", component).group()
    return count[eltype]


def update_expr(circuit_expr, component, idx, variable="X"):
    """Updates the circuit expression with the impedance of a component."""
    replacement = {
        "R": f"{variable}[{idx}]",
        "C": f"(1/(2*1j*np.pi*F*{variable}[{idx}]))",
        "L": f"(2*1j*np.pi*F*{variable}[{idx}])",
        "P": f"(1/({variable}[{idx}]*(2*1j*np.pi*F)**{variable}[{idx+1}]))"
    }
    # Get the alphabet part of the component label R1 -> R
    eltype = re.match(r"[A-Z]+", component).group()
    circuit_expr = circuit_expr.replace(component, replacement[eltype])
    # Update the index
    idx += count_component_parameters(component)
    return circuit_expr, idx


def generate_circuit_fn(circuit: str, use_jax=False) -> callable:
    """Converts a circuit string to a function of (params, freq)"""
    circuit_expr = generate_mathematical_expr(circuit)
    components = get_component_labels(circuit)
    idx = 0
    for component in components:
        circuit_expr, idx = update_expr(circuit_expr, component, idx)
    if use_jax:
        circuit_expr = circuit_expr.replace("np", "jnp")
    return eval(f"lambda X, F: {circuit_expr}")

# <<< Circuit utils


# >>> Statistics utils

def initialize_priors(p0: dict[str, float], variables: list[str]) -> dict[str, dist.Distribution]:
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
            mean, std_dev = jnp.log(value), jnp.log(100)
            priors[var] = dist.LogNormal(mean, std_dev)
    return priors


def initialize_priors_from_posteriors(posterior, variables):
    """Creates new priors based on the posterior distributions."""
    priors = {}
    for var in variables:
        sample = posterior[var]
        # Fit data to a truncated normal distribution for exponents of CPE elements
        # HACK: for better convergence (fewer parameters), fit a normal and truncate it
        if "n" in var:
            # Exponent of CPE elements is bounded between 0 and 1
            loc, scale = norm.fit(sample)
            priors[var] = dist.TruncatedNormal(loc=loc, scale=2*scale, low=0, high=1)
        # Fit data to a log-normal distribution for all other parameters
        else:
            # NOTE: s and scale in scipy.stats -> scale and np.exp(loc) in numpyro
            # NOTE: above conversion is only valid when loc = 0
            s, loc, scale = lognorm.fit(sample, floc=0)
            priors[var] = dist.LogNormal(loc=np.log(scale), scale=2*s)
    return priors

# <<< Statistics utils


# >>> Metrics utils

def mape_score(y_true, y_pred):
    """Generalized MAPE score that can handle complex numbers."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mse_score(y_true, y_pred):
    """Generalized MSE score that can handle complex numbers."""
    return np.mean(np.abs(y_true - y_pred) ** 2)


def rmse_score(y_true, y_pred):
    """Generalized RMSE score that can handle complex numbers."""
    return np.sqrt(mse_score(y_true, y_pred))


def r2_score(y_true, y_pred):
    """Generalized R2 score that can handle complex numbers."""
    ssr = np.sum(np.abs(y_true - y_pred) ** 2)
    sst = np.sum(np.abs(y_true - np.mean(y_true)) ** 2)
    return 1 - ssr / sst

# <<< Metrics utils

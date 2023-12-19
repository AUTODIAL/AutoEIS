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
import signal
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


def find_identical_rows(a) -> list[list[int]]:
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
    @wraps(func)
    def wrapped(*args, **kwargs):
        with _SuppressOutput():
            return func(*args, **kwargs)
    return wrapped


class TimeoutException(Exception):
    pass

def timeout(seconds):
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

# <<< Filesystem utils


# >>> Circuit utils

def validate_parameter_label(p: str) -> bool:
    """Checks if a parameter label is valid."""
    # Check if parameter label is a string
    assert isinstance(p, str), "Parameter label must be a string."
    # Check if parameter label is not empty
    assert len(p) > 0, "Parameter label is empty."
    # Check if parameter label is valid
    pattern = r"(?:R\d+|C\d+|L\d+|P\d+[wn])"
    assert re.fullmatch(pattern, p), f"Invalid parameter label: {p}"
    

def parse_component(c:str) -> str:
    """Returns the type of a component label, e.g., R1 -> R"""
    return re.match(r"[A-Za-z]+", c).group()


def parse_parameter(p:str, by: str = "type") -> str:
    """Returns the type/component of a parameter label, e.g., P4n -> Pn/P"""
    assert by in ["type", "component"], "by must be either 'type' or 'component'."
    validate_parameter_label(p)
    if p.startswith(("R", "C", "L")):
        ptype, ctype = p[0], p[0]
    elif p.startswith("P") and p.endswith("w"):
        ptype, ctype = "Pw", "P"
    elif p.startswith("P") and p.endswith("n"):
        ptype, ctype = "Pn", "P"
    return ptype if by == "type" else ctype


def get_component_labels(circuit: str, types: list[str] = None) -> list[str]:
    """Returns a list of labels for all components in a circuit string."""
    types = [types] if isinstance(types, str) else types
    types = ["R", "C", "L", "P"] if types is None else types
    assert isinstance(types, list), "types must be a list of strings."
    pattern = rf'\b(?:{"|".join(types)})\d+\b'
    return re.findall(pattern, circuit)


def get_component_types(circuit: str, unique=False) -> list[str]:
    """Returns a list of component types in a circuit string."""
    types = re.findall(r"[A-Za-z]+", circuit)
    return list(set(types)) if unique else types


def get_parameter_types(circuit: str, unique=False) -> list[str]:
    """Returns a list of parameter types in a circuit string."""        
    ptypes = [parse_parameter(p, by="type") for p in get_parameter_labels(circuit)]
    return list(set(ptypes)) if unique else ptypes


def get_parameter_labels(circuit: str, types: list[str] = None) -> list[str]:
    """Returns a list of labels for all parameters in a circuit string."""
    types = [types] if isinstance(types, str) else types
    types = ["R", "C", "L", "P"] if types is None else types
    assert isinstance(types, list), "types must be a list of strings."
    components = get_component_labels(circuit, types=types)
    parameters = []
    for component in components:
        # CPE elements have two parameters P{i}w and P{i}n
        if component.startswith("P"):
            parameters.extend([f"{component}w", f"{component}n"])
        else:
            parameters.append(component)
    return parameters


def group_parameters_by_type(circuit: str) -> dict[str, list[str]]:
    """Groups parameter labels by component type."""
    params = get_parameter_labels(circuit)
    ptypes = get_parameter_types(circuit)
    groups = {ptype: [] for ptype in set(ptypes)}
    for param, ptype in zip(params, ptypes):
        groups[ptype].append(param)
    return groups


def group_parameters_by_component(circuit: str) -> dict[str, list[str]]:
    """Groups parameter labels by component label."""
    ctypes = get_component_types(circuit)
    params_by_component = {ctype: [] for ctype in ctypes}
    params = get_parameter_labels(circuit)
    for param in params:
        ctype = parse_parameter(param, by="component")
        params_by_component[ctype].append(param)
    return params_by_component


def count_params(circuit: str) -> int:
    """Returns the number of parameters that fully describe a circuit string."""
    return len(get_parameter_labels(circuit))


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
    """Replaces the Parameters column in EquivalentCircuits.jl output from a
    stringified list to a proper dict[str, float]: "R1 = 1" -> {"R1": 1}.
    """
    circuits = circuits.copy()

    for row in circuits.itertuples():
        pstr = row.Parameters
        # Remove parentheses and spaces, then split by comma -> list[var=val, ...]
        pstr = pstr.strip("()").replace(" ", "")
        pstr = pstr.split(",")
        # Extract variable names and values into a dictionary
        pdict = {pair.split("=")[0]: float(pair.split("=")[1]) for pair in pstr}
        # Replace the stringified list with a proper dict
        circuits.at[row.Index, "Parameters"] = pdict
    
    return circuits


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

# FIXME: Timeout logic doesn't work on Windows -> module 'signal' has no attribute 'SIGALRM'.
if os.name != "nt":
    fit_circuit_parameters = timeout(20)(fit_circuit_parameters)


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


def find_series_elements(circuit: str) -> list[str]:
    """Extracts the series componenets from a circuit (in the main chain)."""
    parsed = circuit_to_nested_expr(circuit)
    series_elements = [el for el in parsed if isinstance(el, str)]
    series_elements = re.findall(r"[A-Z]+\d+", str(series_elements))
    return series_elements


def find_ohmic_resistors(circuit: list) -> list[str]:
    """Finds all ohmic resistors in a nested circuit expression."""
    series_elements = find_series_elements(circuit)
    return re.findall(r"R\d+", str(series_elements))


def validate_circuit_string(circuit: str) -> bool:
    """Checks if a circuit string is valid."""
    # Check for duplicate elements
    components = get_component_labels(circuit)
    duplicates = [e for e in components if components.count(e) > 1]
    assert not duplicates, f"Duplicate elements found: {set(duplicates)}"
    # Test circuit is not empty
    assert len(circuit) > 0, "Circuit string is empty."
    # Check for valid element names
    valid_types = ["R", "C", "L", "P"]
    types = get_component_types(circuit)
    for t in types:
        assert t in valid_types, f"Invalid element type: {t}"
    

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


def embed_impedance_expr(circuit_expr):
    """Updates the circuit expression with the impedance of a component."""
    def replacement(var):
        eltype = get_component_types(var)[0]
        return {
            "R": f"{var}",
            "C": f"(1/(2*1j*pi*F*{var}))",
            "L": f"(2*1j*pi*F*{var})",
            "P": f"(1/({var}w*(2*1j*pi*F)**{var}n))"
        }[eltype]
    # Get component lables
    components = get_component_labels(circuit_expr)
    # Replace components with impedance expression, e.g., C1 -> (1/(2*1j*np.pi*F*C1))
    for c in components:
        circuit_expr = circuit_expr.replace(c, replacement(c))
    return circuit_expr


def generate_circuit_fn(
    circuit: str,
    return_str=False,
    label="X"
) -> Union[callable, str]:
    """Converts a circuit string to a function of (params, freq)"""
    # Apply series-parallel conversion, e.g., [R1,R2] -> (1/R1+1/R2)**(-1)
    circuit_expr = generate_mathematical_expr(circuit)
    # Embed impedance expressions, e.g., C1 -> (1/(2*1j*np.pi*F*C1))
    circuit_expr = embed_impedance_expr(circuit_expr)
    # Replace variables with array indexing, e.g., R1, P2w, P2n -> X[0], X[1], X[2]
    variables = get_parameter_labels(circuit)
    for i, var in enumerate(variables):
        circuit_expr = circuit_expr.replace(var, f"{label}[{i}]", 1)
    if return_str:
        return circuit_expr
    return lambda X, F: eval(circuit_expr)


def is_equivalent(circuit1: str, circuit2: str) -> bool:
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
        labels = get_parameter_labels(circuit)
        x0 = []
        for label in labels:
            ptype = parse_parameter(label, by="type")
            x0.append(values[ptype])
        return np.array(x0)

    freq = np.logspace(-3, 3, 10)
    Z1 = generate_circuit_fn(circuit1)(x0(circuit1), freq)
    Z2 = generate_circuit_fn(circuit2)(x0(circuit2), freq)
    return np.allclose(Z1, Z2)

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


def circuit_complexity(circuit: str) -> list[int]:
    """Computes the component complexity of the circuit."""
    def add_one(arr):
        """Add one to each element in a nested list."""
        return [add_one(e) if isinstance(e, list) else e+1 for e in arr]

    def how_deep(arr):
        """Compute the depth of each element in a nested list."""
        return [add_one(how_deep(e)) if isinstance(e, list) else 0 for e in arr]

    def split(arr):
        """Recursively splits comma-separated elements in a nested list."""
        out = []
        for e in arr:
            if isinstance(e, list):
                out.append(split(e))
            else:
                out.extend(re.split(",|-", e))
        return out

    parsed = circuit_to_nested_expr(circuit)
    parsed = split(parsed)
    depth = how_deep(parsed)
    return flatten(depth)

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

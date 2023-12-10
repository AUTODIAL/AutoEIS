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

import numpy as np
import pandas as pd
import rich.traceback
from impedance.models.circuits import CustomCircuit
from rich.logging import RichHandler

import autoeis as ae

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

def _get_component_names(circuit_string: str, component_type: str) -> list[str]:
    """Returns a list of labels for all components of a given type in a circuit string."""
    return re.findall(rf"{component_type}\d+", circuit_string)


def get_resistors(circuit_string: str) -> list[str]:
    """Returns a list of labels for all resistors in a circuit string."""
    return _get_component_names(circuit_string, "R")


def get_capacitors(circuit_string: str) -> list[str]:
    """Returns a list of labels for all capacitors in a circuit string."""
    return _get_component_names(circuit_string, "C")


def get_inductors(circuit_string: str) -> list[str]:
    """Returns a list of labels for all inductors in a circuit string."""
    return _get_component_names(circuit_string, "L")


def get_cpes(circuit_string: str) -> list[str]:
    """Returns a list of labels for all CPEs in a circuit string."""
    if "CPE" in circuit_string:
        return _get_component_names(circuit_string, "CPE")
    return _get_component_names(circuit_string, "P")


def get_component_labels(circuit_string: str, types="RLCP") -> list[str]:
    """Returns a list of labels for all components in a circuit string."""
    return re.findall(r'[A-Za-z]+[0-9]+', circuit_string)


def get_parameter_labels(circuit_string: str, types="RLCP") -> list[str]:
    """Returns a list of labels for all parameters in a circuit string."""
    labels = get_component_labels(circuit_string, types=types)
    param_labels = []
    for label in labels:
        # CPE elements have two parameters P{i}w and P{i}n
        if label.startswith("P"):
            param_labels.extend([f"{label}w", f"{label}n"])
        else:
            param_labels.append(label)
    return param_labels


def count_params(circuit_string: str) -> int:
    """Returns the number of parameters that fully describe a circuit string."""
    num_resistors = len(get_resistors(circuit_string))
    num_capacitors = len(get_capacitors(circuit_string))
    num_inductors = len(get_inductors(circuit_string))
    num_cpes = len(get_cpes(circuit_string))
    return num_resistors + num_capacitors + num_inductors + 2 * num_cpes


def impedancepy_circuit(circuit_string: str) -> str:
    """Converts a circuit string the format used by impedance.py."""
    circuit_string = circuit_string.replace("P", "CPE")
    circuit_string = circuit_string.replace("[", "p(")
    circuit_string = circuit_string.replace("]", ")")
    return circuit_string


def format_parameters(params, labels):
    """Formats a list of parameters and labels to be used in circuits dataframe."""
    # Example: R1 = 1, C1 = 2 -> "(R1 = 1.0, C1 = 2.0)"
    pairs = [f"{label} = {value:.10f}" for label, value in zip(labels, params)]
    pairs = "(" + ", ".join(pairs) + ")"
    return pairs


def parse_circuit_dataframe(circuit: pd.DataFrame):
    """Parses a circuit dataframe into a circuit string and parameter dictionary."""
    circuit_string = circuit["circuitstring"].item()
    pattern = r"(\b\w+\b)\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    params_dict = dict(re.findall(pattern, circuit["Parameters"].item()))
    params_dict = {k: float(v) for k, v in params_dict.items()}
    return circuit_string, params_dict


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

# <<< Circuit utils


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

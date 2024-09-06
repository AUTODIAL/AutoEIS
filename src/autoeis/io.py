"""
Collection of functions for importing and exporting EIS data/results.

.. currentmodule:: autoeis.io

.. autosummary::
   :toctree: generated/

    get_assets_path
    load_battery_dataset
    load_test_dataset
    load_test_circuits
    parse_ec_output

"""

import logging
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

import autoeis as ae

log = logging.getLogger(__name__)


def get_assets_path() -> Path:
    """Returns the path to the assets folder."""
    PATH = Path(ae.__file__).parent / "assets"
    return PATH


def load_battery_dataset(
    preprocess: bool = False,
    noise: float = 0,
) -> list[tuple[np.ndarray[float], np.ndarray[complex]]]:
    """Loads EIS data of a battery cell during cycling (at discharged state).

    Parameters
    ----------
    preprocess: bool, optional
        If True, the impedance data is preprocessed using
        :func:`autoeis.core.preprocess_impedance_data`. Default is False.
    noise: float, optional
        If greater than zero, uniform noise with prescribed amplitude is added
        to the impedance data. Default is 0.

    Returns
    -------
    list[tuple[np.ndarray[float], np.ndarray[complex]]]
        List of tuples of frequency and impedance arrays for each cycle.
    """
    PATH = get_assets_path()
    fpath = os.path.join(PATH, "battery_data.npy")
    data = np.load(fpath)
    # Data are stored as complex, convert frequency to float
    data = [(freq.real, Z) for freq, Z in data]
    if preprocess:
        data = [ae.utils.preprocess_impedance_data(freq, Z) for freq, Z in data]
    if noise:
        # Only add noise to impedance data (opinionated!)
        noise_real = [np.random.rand(len(Z)) * Z.real * noise for freq, Z in data]
        noise_imag = [np.random.rand(len(Z)) * Z.imag * noise for freq, Z in data]
        data = [
            (freq, Z + noise_real[i] + noise_imag[i] * 1j) for i, (freq, Z) in enumerate(data)
        ]
    return data


def load_test_dataset(
    preprocess: bool = False, noise: float = 0
) -> tuple[np.ndarray[float], np.ndarray[complex]]:
    """Returns a test dataset as a tuple of frequency and impedance arrays.

    Parameters
    ----------
    preprocess: bool, optional
        If True, the impedance data is preprocessed using
        :func:`autoeis.core.preprocess_impedance_data`. Default is False.
    noise: float, optional
        If greater than zero, uniform noise with prescribed amplitude is added
        to the impedance data. Default is 0.

    Returns
    -------
    tuple[np.ndarray[float], np.ndarray[complex]]
        Tuple of frequency and impedance data.
    """
    PATH = get_assets_path()
    fpath = os.path.join(PATH, "test_data.txt")
    freq, Zreal, Zimag = np.loadtxt(fpath, skiprows=1, unpack=True, usecols=(0, 1, 2))
    # Convert to complex impedance (the file contains -Im(Z) hence the minus sign)
    Z = Zreal - 1j * Zimag
    if preprocess:
        freq, Z = ae.utils.preprocess_impedance_data(freq, Z)
    if noise:
        # Only add noise to impedance data (opinionated!)
        noise_real = np.random.rand(len(Z)) * Z.real * noise
        noise_imag = np.random.rand(len(Z)) * Z.imag * noise
        Z += noise_real + noise_imag * 1j
    return freq, Z


def load_test_circuits(filtered: bool = False) -> pd.DataFrame:
    """Returns candidate ECMs fitted to test dataset for testing.

    Parameters
    ----------
    filtered: bool, optional
        If True, only physically plausible circuits are returned. Default is False.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the ECMs.

    """
    PATH = get_assets_path()
    fname = "circuits_filtered.csv" if filtered else "circuits_unfiltered.csv"
    fpath = os.path.join(PATH, fname)
    circuits = pd.read_csv(fpath)
    # Convert stringified list to proper Python objects
    circuits["Parameters"] = circuits["Parameters"].apply(eval)
    return circuits


def parse_ec_output(
    circuits: Iterable[str] | str, ignore_invalid_inputs: bool = True
) -> pd.DataFrame:
    """Parses the output of EquivalentCircuits.jl's ``circuit_evolution``.

    Parameters
    ----------
    circuits: Iterable[str] | str
        List of stringified output of EquivalentCircuits.jl's
        ``circuit_evolution``. A valid input should be in the following format:
        'EquivalentCircuit("R1", (R1 = 1.0,))', or a list of such strings.

    Returns
    -------
    pd.DataFrame
        Dataframe containing ECMs (cols: "circuitstring" and "Parameters")
    """

    def _validate_input(ec_output: str, raise_error: bool = True):
        """Ensures the input string can be parsed into circuit and parameters."""
        ec_output = ec_output.replace(" ", "")
        try:
            assert len(ec_output.split('",(')) == 2
            assert "EquivalentCircuit" in ec_output
        except AssertionError:
            if raise_error:
                raise ValueError(f"Invalid EC output format: {ec_output}.")
            return False
        return True

    def _split_labels_and_values(ec_output: str) -> tuple[str, dict[str, float]]:
        """Splits the circuit string and parameters from the input string."""
        ec_output = ec_output.removeprefix("EquivalentCircuit(").removesuffix(")")
        ec_output = ec_output.replace(" ", "")
        cstr, pstr = ec_output.split('",(')
        cstr = cstr.replace('"', "")
        # NOTE: Trailing comma in one-element tuples needs to also be removed
        pstr = pstr.replace(")", "").replace("(", "").replace('"', "").rstrip(",").split(",")
        # Convert parameters substring to a dict[str, float]
        pdict = dict(pair.split("=") for pair in pstr)  # dict[str, str]
        pdict = {p.split("=")[0]: float(p.split("=")[1]) for p in pstr}  # dict[str, float]
        return cstr, pdict

    circuits = [circuits] if isinstance(circuits, str) else circuits
    parsed = []

    header = "circuitstring,Parameters"

    for circuit in circuits:
        # Validate input format
        circuit = circuit.replace(" ", "")
        # Skip header if present
        if circuit == header:
            continue
        # Skip invalid inputs if flag is set
        if not _validate_input(circuit, raise_error=not ignore_invalid_inputs):
            continue
        # Finally, parse the circuit and parameters
        cstr, pdict = _split_labels_and_values(circuit)
        parsed.append([cstr, pdict])

    return pd.DataFrame(parsed, columns=["circuitstring", "Parameters"])

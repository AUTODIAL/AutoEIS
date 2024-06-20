"""
Collection of functions for importing and exporting EIS data/results.

.. currentmodule:: autoeis.io

.. autosummary::
   :toctree: generated/

    get_assets_path
    load_test_dataset
    load_test_circuits
    parse_ec_output

"""

import logging
import os
import re
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


def parse_ec_output(circuits: Iterable[str] | str) -> pd.DataFrame:
    """Parses the output of EquivalentCircuits.jl's ``circuit_evolution``.

    Parameters
    ----------
    circuits: Iterable[str] | str
        List of stringified output of EquivalentCircuits.jl's ``circuit_evolution``.

    Returns
    -------
    pd.DataFrame
        Dataframe containing ECMs (cols: "circuitstring" and "Parameters")
    """
    # Example: 'EquivalentCircuit("R1", (R1 = 1.0,))' -> ('R1', {'R1': 1.0})
    circuits = [circuits] if isinstance(circuits, str) else circuits
    parsed = []

    for circuit in circuits:
        circuit = circuit.removeprefix("EquivalentCircuit(").removesuffix(")")
        cstr = re.findall(r"\"(.*?)\"", circuit)[0]
        pstr = re.findall(r"\((.*?)\)", circuit)[0]
        # NOTE: rstrip(",") to account for the trailing comma in one-element tuples
        pstr = pstr.replace(" ", "").rstrip(",").split(",")
        pdict = dict(pair.split("=") for pair in pstr)
        pdict = {p.split("=")[0]: float(p.split("=")[1]) for p in pstr}
        parsed.append([cstr, pdict])

    return pd.DataFrame(parsed, columns=["circuitstring", "Parameters"])

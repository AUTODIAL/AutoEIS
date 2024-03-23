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
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import autoeis as ae
import autoeis.utils as utils

log = utils.get_logger(__name__)


def get_assets_path() -> Path:
    """Returns the path to the assets folder."""
    PATH = Path(ae.__file__).parent / "assets"
    return PATH


def load_test_dataset() -> tuple[np.ndarray[complex], np.ndarray[float]]:
    """Returns a test dataset as a tuple of impedance and frequency arrays."""
    PATH = get_assets_path()
    fpath = os.path.join(PATH, "test_data.txt")
    freq, Zreal, Zimag = np.loadtxt(fpath, skiprows=1, unpack=True, usecols=(0, 1, 2))
    # Convert to complex impedance (the file contains -Im(Z) hence the minus sign)
    Z = Zreal - 1j * Zimag
    return Z, freq


def load_test_circuits(filtered=False) -> pd.DataFrame:
    """Returns equivalent circuits fitted to test dataset for testing."""
    PATH = get_assets_path()
    fname = "circuits_filtered.csv" if filtered else "circuits_unfiltered.csv"
    fpath = os.path.join(PATH, fname)
    circuits = pd.read_csv(fpath)
    # Convert stringified list to proper Python objects
    circuits["Parameters"] = circuits["Parameters"].apply(eval)
    return circuits


def parse_ec_output(circuits: list[str]) -> list[tuple[str, dict[str, float]]]:
    """Parses the output of EquivalentCircuits.jl's ``circuit_evolution``."""
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

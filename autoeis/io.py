"""
Collection of functions for importing and exporting EIS data/results.

.. currentmodule:: autoeis.io

.. autosummary::
   :toctree: generated/

    get_assets_path
    load_test_dataset
    parse_ec_output

"""
import os
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
    Z = Zreal - 1j*Zimag
    return Z, freq


def parse_ec_output(circuits: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the output of EquivalentCircuits.jl's ``circuit_evolution``
    and returns a dataframe with circuit parameters stored as dict[str, float].
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

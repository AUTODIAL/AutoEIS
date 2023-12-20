"""
Collection of functions for importing and exporting EIS data/results.

.. currentmodule:: autoeis.io

.. autosummary::
   :toctree: generated/

    get_assets_path
    load_test_dataset

"""
import os
from pathlib import Path

import numpy as np

import autoeis as ae
import autoeis.utils as utils

log = utils.get_logger(__name__)


def get_assets_path() -> Path:
    """Returns the path to the assets folder."""
    PATH = Path(ae.__file__).parent / "assets"
    return PATH


def load_test_dataset() -> tuple[np.ndarray[complex], np.ndarray[float]]:
    """Loads a test dataset from package assets folder."""
    PATH = get_assets_path()
    fpath = os.path.join(PATH, "test_data.txt")
    freq, Zreal, Zimag = np.loadtxt(fpath, skiprows=1, unpack=True, usecols=(0, 1, 2))
    # Convert to complex impedance (the file contains -Im(Z) hence the minus sign)
    Z = Zreal - 1j*Zimag
    return Z, freq

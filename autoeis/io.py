"""
Collection of functions for importing and exporting EIS data/results.

.. currentmodule:: autoeis.io

.. autosummary::
   :toctree: generated/

    get_assets_path
    load_test_dataset

"""
import os

import numpy as np

import autoeis as ae
import autoeis.utils as utils

log = utils.get_logger(__name__)


def get_assets_path():
    """Returns the path to the assets folder."""
    return ae.__path__[0] + "/assets"


def load_test_dataset():
    """Loads a test dataset from package assets folder."""
    PATH = get_assets_path()
    fpath = os.path.join(PATH, "test_data.txt")
    freq, Zreal, Zimag = np.loadtxt(fpath, skiprows=1, unpack=True, usecols=(0, 1, 2))
    # Convert to complex impedance (the file contains -Im(Z) hence the minus sign)
    Z = Zreal - 1j*Zimag
    return Z, freq

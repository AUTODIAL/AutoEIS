from pathlib import Path

import numpy as np
import pytest

import autoeis as ae

ASSETS_DIR = Path(__file__).parent.parent / 'assets'
FILE_PATH = ASSETS_DIR / 'file.txt'


def test_load_eis_data():
    # Throw error for non-existent files
    with pytest.raises(Exception):
        ae.load_eis_data("nonexistentfile.csv")

    # Load a valid txt file of EIS data
    fpath = ASSETS_DIR / 'testdata.txt'
    df = ae.load_eis_data(fpath)
    frequencies = np.array(df["freq/Hz"]).astype(float)
    reals = np.array(df["Re(Z)/Ohm"]).astype(float)
    imags = -np.array(df["-Im(Z)/Ohm"]).astype(float)

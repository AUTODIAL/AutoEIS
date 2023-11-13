"""
Collection of functions for importing and exporting EIS data/results.

.. currentmodule:: autoeis.io

.. autosummary::
   :toctree: generated/

    load_eis_data
    load_results_dataframe

"""

import json
import pickle
from pathlib import Path

import pandas as pd

import autoeis.utils as utils

log = utils.get_logger(__name__)


# TODO: We're not really providing any value here.
def load_eis_data(fname: str, column_indices=[0, 1, 2]) -> pd.DataFrame:
    """Loads electrochemical impedance spectroscopy data from a file.

    Parameters
    ----------
    fname : str
        Path to the EIS data file.
    column_indices : list[int]
        Indices of the columns containing Re(Z), and Im(Z), and frequencies.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns "freq", "Zreal", and "Zimag".

    Raises
    ------
    ValueError:
        If the file format is not supported.
    FileNotFoundError:
        If the file does not exist.
    """
    path = Path(fname)

    csv_args = {
        "header": 0 if _includes_header(fname) else "infer",
        "names": ["Zreal", "Zimag", "freq"],
        "usecols": column_indices,
    }

    loaders = {
        ".json": lambda: pd.DataFrame(json.loads(path.read_text())),
        ".csv": lambda: pd.read_csv(path, **csv_args),
        ".txt": lambda: pd.read_csv(path, sep="\t", **csv_args),
        ".xlsx": lambda: pd.read_excel(path),
        ".pkl": lambda: pd.DataFrame(pickle.loads(path.read_bytes()))
    }

    loader = loaders.get(path.suffix)
    if loader:
        return loader()
    else:
        log.error("Unsupported file format.")
        raise ValueError("Unsupported file format.")


# TODO: We're not really providing any value here.
def load_results_dataframe(fname: str) -> pd.DataFrame:
    """Loads AutoEIS results CSV file and converts it to a dataframe.

    Parameters
    ----------
    fname: str
        Path of the CSV file containing AutoEIS results.

    Returns
    -------
    df_circuits: pd.DataFrame
        Dataframe containing ECMs (2 columns)
    """
    df_circuits = pd.read_csv(fname)
    return df_circuits


def _includes_header(fpath):
    """Checks if a CSV file includes a header."""
    df = pd.read_csv(fpath)
    try:
        df.columns.astype(float)
    except ValueError:
        return True
    return False

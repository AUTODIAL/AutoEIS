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
def load_eis_data(fname: str) -> pd.DataFrame:
    """Load EIS (Electrochemical Impedance Spectroscopy) data from a file.

    Parameters
    ----------
    fname : str
        Path to the EIS data file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing impedance and frequency data.

    Raises
    ------
    ValueError:
        If the file format is not supported.
    FileNotFoundError:
        If the file does not exist.
    """
    path = Path(fname)

    if not path.exists():
        log.error(f"No such file or directory: {path}")
        raise FileNotFoundError(f"No such file or directory: {path}")

    loaders = {
        ".json": lambda: pd.DataFrame(json.loads(path.read_text())),
        ".csv": lambda: pd.read_csv(path),
        ".txt": lambda: pd.read_csv(path, sep="\t"),
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
    """Load AutoEIS results CSV file and convert it to a dataframe.

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

"""
Collection of functions for importing and exporting EIS data/results.

.. currentmodule:: autoeis.io

.. autosummary::
   :toctree: generated/

    load_eis_data
    load_results_dataframe

"""
import pandas as pd

import autoeis.utils as utils

log = utils.get_logger(__name__)


# TODO: this function does nothing, needs to be removed
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
    """Checks if a csv/txt file includes a header."""
    df = pd.read_csv(fpath)
    try:
        df.columns.astype(float)
    except ValueError:
        return True
    return False

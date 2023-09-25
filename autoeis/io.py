import json
import logging
import pickle
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


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


def load_results(fname: str) -> pd.DataFrame:
    """Load the generated ECMs and convert to a dataframe.

    Parameters
    ----------
    fname: str
        Path of the file containing the generated ECMs

    Returns:
    --------
    df_circuits: pd.DataFrame
        Dataframe containing ECMs (2 columns)
    """
    df_circuits = pd.read_csv(fname)
    if len(df_circuits) == 0:
        log.error("No plausible ECMs found. Consider increasing the iterations.")
    return df_circuits

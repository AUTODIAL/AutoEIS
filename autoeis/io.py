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

"""
Collection of utility functions used throughout the package.

.. currentmodule:: autoeis.utils

.. autosummary::
   :toctree: generated/

    get_logger
    setup_rich_tracebacks
    make_dir
    get_dirname
    get_filename
    get_fileext
    reset_storage
    suppress_output

"""

import logging
import os
import sys
from functools import wraps
from pathlib import Path

import rich.traceback
from rich.logging import RichHandler

# --- Logging utils --- # BEGIN

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # If logger has handlers, do not add another to avoid duplicate logs
        return logger
    
    logger.setLevel(logging.WARNING)  # Set the logging level to INFO for this logger.
    handler = RichHandler(rich_tracebacks=True)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(handler)
    return logger


def setup_rich_tracebacks() -> None:
    """Set up rich traceback for nicer exception printing."""
    rich.traceback.install()

# --- Logging utils --- # END


# --- Filesystem utils --- # BEGIN

class _SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def suppress_output(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with _SuppressOutput():
            return func(*args, **kwargs)
    return wrapped

# --- Filesystem utils --- # END

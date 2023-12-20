import builtins

from . import core, io, metrics, utils, visualization  # noqa: F401
from .core import *
from .version import __equivalent_circuits_jl_version__, __version__  # noqa: F401

# Set up rich traceback for nicer exception printing
utils.setup_rich_tracebacks()

# Override print() with rich's print() for nicer printing
builtins.print = visualization.rich_print

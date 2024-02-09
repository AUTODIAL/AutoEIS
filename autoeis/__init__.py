from .utils import get_logger  # noqa: F401

log = get_logger(__name__)
log.warning(
    "If you use the Python distribution by Conda, the initialization "
    "of the package may take 1-2 minutes. Be patient :)"
)

from . import core, io, metrics, parser, utils, visualization  # noqa: F401, E402
from .core import *  # noqa: E402
from .version import __equivalent_circuits_jl_version__, __version__  # noqa: F401, E402
from .visualization import rich_print  # noqa: F401, E402

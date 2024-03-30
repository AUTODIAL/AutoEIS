import logging

from rich.console import Console
from rich.logging import RichHandler

from . import core, io, metrics, parser, utils, visualization  # noqa: F401, E402
from .core import *  # noqa: E402
from .version import __equivalent_circuits_jl_version__, __version__  # noqa: F401, E402
from .visualization import rich_print  # noqa: F401, E402

settings = utils.Settings()


def _setup_logger_rich():
    """Sets up logging using ``rich``."""
    log = logging.getLogger("autoeis")

    if log.hasHandlers():
        log.critical("Logging already set up.")
        return

    log.setLevel(logging.WARNING)
    console = Console(force_jupyter=False)
    handler = RichHandler(
        rich_tracebacks=True, console=console, show_path=not settings.notebook
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    log.addHandler(handler)


_setup_logger_rich()

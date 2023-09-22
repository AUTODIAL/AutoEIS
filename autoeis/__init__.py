import logging

import rich.traceback
from rich.logging import RichHandler

# Enable rich traceback for nicer exception printing
rich.traceback.install()

# Set up logging using rich for nicer logging
FORMAT = "%(message)s"
logging.basicConfig(
    level="WARNING",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

from ._funcs import *

__version__ = "0.0.15"
__description__ = "A demo package to assist EIS analysis by automatically proposing statistically plausible ECM "
__author__ = "Runze Zhang, Robert Black, Jason Hattrick-Simpers"
__contact_information = "runzee.zhang@mail.utoronto.ca"


import logging

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="WARNING", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'An Automated EIS analysis packgae'
LONG_DESCRIPTION = 'A package for fully automated EIS analysis based on ECMs generated by GEP-GA'

# Setting up
setup(
    name="AutoEis",
    version= VERSION,
    author="Runze zhang, Robert Black, Parisa Karimi, Jason Hattrick-Simpers",
    author_email="runzee.zhang@mail.utoronto.ca",
    description= DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/kbknudsen/PyEIS",
    packages = find_packages(),
    install_requires = [],
    keywords=['python', 'Electrochemical impedance spectroscopy', 'Fully-automated', 'equivalent circuit models', 'gene-expression programming', 'genetic algorithm'],
    classifiers=[
        "Programming Language :: Python ::3",
        "License :: MIT",
        "Operating System :: Windows 10",
    ],
)

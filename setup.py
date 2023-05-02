from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.13'
DESCRIPTION = 'A demo package to assist EIS analysis by automatically proposing statistically plausible ECM'
LONG_DESCRIPTION = 'AutoEIS is a novel tool designed to aid EIS analysis by automatically prioritizing statistically optimal ECM by combining evolutionary algorithms and Bayesian inference.'

# Setting up
setup(
    name="AutoEis",
    version= VERSION,
    author="Runze zhang, Robert Black, Jason Hattrick-Simpers*",
    author_email="runzee.zhang@mail.utoronto.ca",
    description= DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="",
    packages = find_packages(),
    install_requires = ['matplotlib>=3.3.2',
                        'numpy>=1.20.3',
                        'pandas>=1.1.3',
                        'impedance>=1.4.0',
                        'regex>=2.2.1',
                        'arviz>=0.12.0',
                        'numpyro>=0.9.1',
                        'dill>=0.3.4',
                        'julia>=0.5.7',
                        'IPython>=7.19.0',
                        'jax>=0.3.9'],
    keywords=['Electrochemical impedance spectroscopy', 'equivalent circuit models', 'Bayesian inference', 'evolutionary algorithm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
    ],
)

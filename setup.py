from pathlib import Path

from setuptools import find_packages, setup

version = "0.0.16"
description = "A tool for automated EIS analysis by proposing statistically plausible ECMs."
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="autoeis",
    version=version,
    author="Runze Zhang, Robert Black, Jason Hattrick-Simpers*",
    author_email="runzee.zhang@mail.utoronto.ca",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.3.2",
        "numpy>=1.20.3",
        "pandas>=1.1.3",
        "impedance>=1.4.0",
        "regex>=2.2.1",
        "arviz==0.12.0",
        "numpyro==0.10.1",
        "dill>=0.3.4",
        "julia==0.5.7",
        "IPython>=7.19.0",
        "jax==0.3.9",
    ],
    keywords=[
        "electrochemical impedance spectroscopy",
        "equivalent circuit models",
        "bayesian inference",
        "evolutionary algorithm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

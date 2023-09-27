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
        "matplotlib",
        "numpy",
        "pandas",
        "impedance",
        "arviz",
        "numpyro",
        "dill",
        "julia",
        "ipython",
        "jax",
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

# AutoEIS Input Data Format Guide

As demonstrated in the [example notebook](https://github.com/AUTODIAL/AutoEIS/blob/main/examples/demos/basic_workflow.ipynb), AutoEIS requires access to two specific features from the measured EIS data:
- **Impedance Data (Complex Values)**: Denoted as *Z* in the sample file.
- **Measurement Frequencies**: Denoted as *freq*.

Due to the variety of devices and settings used for EIS analysis, the collected data often comes in different formats, presenting challenges for developing a universal data loading function. To address this, we suggest users customize their data loading process as described below.

## General Formats
For common EIS data formats, you can load them using the following methods:
- **CSV (.csv)**: `pandas.read_csv`
- **JSON (.json)**: `pandas.read_json` or `json.load()`
- **Spreadsheets (.xls, .xlsx)**: `pandas.read_excel`
- **Text file (.txt)**: `numpy.loadtxt`

## Device-Specific Formats
For specific formats provided by measurement devices, consider using additional packages such as [pyimpspec](https://vyrjana.github.io/pyimpspec/index.html):
- **BioLogic**: (.mpt)
- **Eco Chemie**: (.dfr)
- **Gamry**: (.dta)
- **Ivium**: (.idf) and (.ids)

## The `pyimpspec` Package
[`pyimpspec`](https://vyrjana.github.io/pyimpspec/index.html) is a Python library designed to handle and process spectroscopic data. It supports loading EIS data from various proprietary formats and converting them into a format compatible with AutoEIS, facilitating easy data access.

## Example Usage
Below is an example demonstrating how to use `pyimpspec` to load a proprietary EIS data format and access the impedance data.

```python
# !pip install pyimpspec

from pyimpspec import DataSet, parse_data

# Load EIS data from a proprietary format
data: DataSet
for eis_data in parse_data("path/to/your/data.dta"):
    loaded_eis = eis_data.to_dataframe()

    # Extract frequency and complex impedance data
    freq = loaded_eis["f (Hz)"].values
    Z = loaded_eis["Re(Z) (ohm)"].values + loaded_eis["Im(Z) (ohm)"].values * 1j

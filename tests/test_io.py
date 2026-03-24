import os

import autoeis as ae
import numpy as np
import pytest


def test_get_assets_path():
    path = ae.io.get_assets_path()
    assert os.path.exists(path)


def test_load_test_dataset():
    freq, Z = ae.io.load_test_dataset()
    assert Z.shape == freq.shape
    assert len(Z) == len(freq) > 0


def test_load_dataset_preprocess():
    freq0, Z0 = ae.io.load_test_dataset()
    freq, Z = ae.io.load_test_dataset(preprocess=True)
    assert Z.shape == freq.shape
    assert len(Z) == len(freq) > 0
    assert len(Z) < len(Z0)


def test_load_dataset_noise():
    freq0, Z0 = ae.io.load_test_dataset()
    freq, Z = ae.io.load_test_dataset(noise=0.1)
    assert Z.shape == freq.shape
    assert len(Z) == len(freq) > 0
    assert np.allclose(freq, freq0)  # No noised added to frequency
    assert not np.allclose(Z, Z0)  # Noise added to impedance


def test_load_test_circuits_no_filter():
    circuits = ae.io.load_test_circuits()
    assert len(circuits) == 118
    assert "circuitstring" in circuits.columns.tolist()
    assert "Parameters" in circuits.columns.tolist()


def test_load_test_circuits_filter():
    circuits = ae.io.load_test_circuits(filtered=True)
    assert len(circuits) == 15
    assert "circuitstring" in circuits.columns.tolist()
    assert "Parameters" in circuits.columns.tolist()


def test_parse_ec_output_single():
    circuits = 'EquivalentCircuit("R1", (R1 = 1.0,))'
    circuits = ae.io.parse_ec_output(circuits)
    assert len(circuits) == 1
    assert circuits.columns.tolist() == ["circuitstring", "Parameters"]
    assert circuits["circuitstring"][0] == "R1"
    assert circuits["Parameters"][0] == {"R1": 1.0}


def test_parse_ec_output_multiple():
    circuits = [
        'EquivalentCircuit("R1", (R1 = 1.0,))',
        'EquivalentCircuit("R1-[C2,P3]", (R1 = 1.0, C2 = 1.5, P3w = 1.25e3, P3n = 0.75))',
    ]
    circuits = ae.io.parse_ec_output(circuits)
    assert len(circuits) == 2
    assert circuits.columns.tolist() == ["circuitstring", "Parameters"]
    assert circuits["circuitstring"][0] == "R1"
    assert circuits["Parameters"][0] == {"R1": 1.0}
    assert circuits["circuitstring"][1] == "R1-[C2,P3]"
    assert circuits["Parameters"][1] == {"R1": 1.0, "C2": 1.5, "P3w": 1.25e3, "P3n": 0.75}


# --- load_eis_data tests ---

ASSETS = ae.io.get_assets_path()


def _assert_valid_eis(freq, Z):
    """Checks that freq and Z are valid EIS arrays."""
    assert freq.shape == Z.shape
    assert len(freq) > 0
    assert np.issubdtype(freq.dtype, np.floating)
    assert np.issubdtype(Z.dtype, np.complexfloating)


@pytest.mark.parametrize("instrument,filename", [
    ("gamry", "exampleDataGamry.DTA"),
    ("autolab", "exampleDataAutolab.txt"),
    ("biologic", "exampleDataBioLogic.mpt"),
    ("parstat", "exampleDataParstat.txt"),
    ("zplot", "exampleDataZPlot.z"),
    ("versastudio", "exampleDataVersaStudio.par"),
    ("powersuite", "exampleDataPowersuite.txt"),
    ("chinstruments", "exampleDataCHInstruments.txt"),
])
def test_load_eis_data_vendored_formats(instrument, filename):
    freq, Z = ae.io.load_eis_data(ASSETS / filename, instrument=instrument)
    _assert_valid_eis(freq, Z)


@pytest.mark.parametrize("instrument,filename", [
    ("ecochemie", "data.dfr"),
    ("ivium", "data.idf"),
    ("ivium", "data.ids"),
    ("palmsens", "data.pssession"),
    ("spreadsheet", "data.xlsx"),
    ("spreadsheet", "data.ods"),
])
def test_load_eis_data_pyimpspec_formats(instrument, filename):
    freq, Z = ae.io.load_eis_data(ASSETS / filename, instrument=instrument)
    _assert_valid_eis(freq, Z)


def test_load_eis_data_auto_detect():
    freq, Z = ae.io.load_eis_data(ASSETS / "exampleDataGamry.DTA")
    _assert_valid_eis(freq, Z)


def test_load_eis_data_auto_detect_csv():
    freq, Z = ae.io.load_eis_data(ASSETS / "exampleData.csv")
    _assert_valid_eis(freq, Z)


def test_load_eis_data_invalid_instrument():
    with pytest.raises(ValueError, match="Unsupported instrument"):
        ae.io.load_eis_data(ASSETS / "exampleData.csv", instrument="nonexistent")


def test_load_eis_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        ae.io.load_eis_data(ASSETS / "nonexistent.dta", instrument="gamry")

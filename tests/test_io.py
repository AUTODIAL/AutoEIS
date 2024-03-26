import os

import autoeis as ae


def test_get_assets_path():
    path = ae.io.get_assets_path()
    assert os.path.exists(path)


def test_load_test_dataset():
    Z, freq = ae.io.load_test_dataset()
    assert Z.shape == freq.shape
    assert len(Z) == len(freq) > 0


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

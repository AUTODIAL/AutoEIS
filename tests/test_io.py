import os

import autoeis as ae


def test_get_assets_path():
    path = ae.io.get_assets_path()
    assert os.path.exists(path)


def test_load_test_dataset():
    freq, Z = ae.io.load_test_dataset()
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

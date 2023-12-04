import re

import numpy as np
import pandas as pd
import pytest
from impedance.models.circuits import CustomCircuit

import autoeis as ae


def test_generate_mathematical_expression():
    circuit = "R0-C1-[P2,R3]-[P4-[P5,C6],[L7,R8]]"

    resistors = re.findall(r"R\d+", circuit)
    capacitors = re.findall(r"C\d+", circuit)
    inductors = re.findall(r"L\d+", circuit)
    cpes = re.findall(r"P\d+", circuit)

    num_params = len(resistors) + len(capacitors) + len(inductors) + 2*len(cpes)
    freq = np.array([1, 10, 100])
    p = np.random.rand(num_params)

    circuit_df = pd.DataFrame([circuit], columns=["circuitstring"])
    func_str = ae.core.generate_mathematical_expression(circuit_df)
    func_str = func_str["Mathematical expressions"][0]
    def func(X, F): return eval(func_str)
    Z_py = func(p, freq)

    ec = ae.julia_helpers.import_backend()
    Z_jl = np.array([ec.get_target_impedance(circuit, p, f) for f in freq])

    np.testing.assert_allclose(Z_py, Z_jl)


def test_find_ohmic_resistance():
    circuit_string = "R1-[P2,P3-R4]"
    circuit_string = ae.utils.impedancepy_circuit(circuit_string)
    R1 = 250
    parameters = np.array([R1, 1e-3, 0.1, 5e-5, 0.8, 10])
    circuit = CustomCircuit(circuit_string, initial_guess=parameters)
    freq = np.logspace(-3, 3, 1000)
    Z = circuit.predict(freq)
    R = ae.core.find_ohmic_resistance(Z, freq)
    assert np.isclose(R, R1, rtol=0.15)


# FIXME: We don't have a robust way to handle missing high frequency data
def test_find_ohmic_resistance_missing_high_freq():
    circuit_string = "R1-[P2,P3-R4]"
    circuit_string = ae.utils.impedancepy_circuit(circuit_string)
    R1 = 250
    parameters = np.array([R1, 1e-3, 0.1, 5e-5, 0.8, 10])
    circuit = CustomCircuit(circuit_string, initial_guess=parameters)
    freq = np.logspace(-3, 1, 1000)
    Z = circuit.predict(freq)
    # with pytest.raises(ValueError):
    #     R = ae.core.find_ohmic_resistance(Z, freq)

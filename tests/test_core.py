import numpy as np
from impedance.models.circuits import CustomCircuit

import autoeis as ae
from autoeis import julia_helpers, parser, utils


def test_generate_circuit_fn():
    circuit = "R0-C1-[P2,R3]-[P4-[P5,C6],[L7,R8]]"
    num_params = parser.count_parameters(circuit)
    freq = np.array([1, 10, 100])
    p = np.random.rand(num_params)
    circuit_fn = utils.generate_circuit_fn(circuit)
    Z_py = circuit_fn(p, freq)
    ec = julia_helpers.import_backend()
    Z_jl = np.array([ec.get_target_impedance(circuit, p, f) for f in freq])
    np.testing.assert_allclose(Z_py, Z_jl)


def test_compute_ohmic_resistance():
    circuit_string = "R1-[P2,P3-R4]"
    circuit_string = ae.parser.impedancepy_circuit(circuit_string)
    R1 = 250
    parameters = np.array([R1, 1e-3, 0.1, 5e-5, 0.8, 10])
    circuit = CustomCircuit(circuit_string, initial_guess=parameters)
    circuit.parameters_ = parameters
    freq = np.logspace(-3, 3, 1000)
    Z = circuit.predict(freq)
    R = ae.core.compute_ohmic_resistance(Z, freq)
    np.testing.assert_allclose(R, R1, rtol=0.15)


def test_compute_ohmic_resistance_missing_high_freq():
    circuit_string = "R1-[P2,P3-R4]"
    circuit_string = ae.parser.impedancepy_circuit(circuit_string)
    R1 = 250
    parameters = np.array([R1, 1e-3, 0.1, 5e-5, 0.8, 10])
    circuit = CustomCircuit(circuit_string, initial_guess=parameters)
    circuit.parameters_ = parameters
    freq = np.logspace(-3, 0, 1000)
    Z = circuit.predict(freq)
    R = ae.core.compute_ohmic_resistance(Z, freq)
    Zreal_at_high_freq = Z.real[-1]
    np.testing.assert_allclose(R, Zreal_at_high_freq)

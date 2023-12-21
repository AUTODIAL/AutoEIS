import numpy as np

from autoeis import core, utils


def test_compute_ohmic_resistance():
    circuit_string = "R1-[P2,P3-R4]"
    circuit_fn = utils.generate_circuit_fn_impedance_backend(circuit_string)
    R1 = 250
    parameters = np.array([R1, 1e-3, 0.1, 5e-5, 0.8, 10])
    freq = np.logspace(-3, 3, 1000)
    Z = circuit_fn(parameters, freq)
    R = core.compute_ohmic_resistance(Z, freq)
    np.testing.assert_allclose(R, R1, rtol=0.15)


def test_compute_ohmic_resistance_missing_high_freq():
    circuit_string = "R1-[P2,P3-R4]"
    circuit_fn = utils.generate_circuit_fn_impedance_backend(circuit_string)
    R1 = 250
    parameters = np.array([R1, 1e-3, 0.1, 5e-5, 0.8, 10])
    freq = np.logspace(-3, 0, 1000)
    Z = circuit_fn(parameters, freq)
    R = core.compute_ohmic_resistance(Z, freq)
    # When high frequency measurements are missing, Re(Z) @ max(freq) is good approximation
    Zreal_at_high_freq = Z.real[np.argmax(freq)]
    np.testing.assert_allclose(R, Zreal_at_high_freq)

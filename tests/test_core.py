import numpy as np
import pandas as pd

from autoeis import core, io, utils


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


def test_gep_serial():
    Z, freq = io.load_test_dataset()
    Z, freq, rmse = core.preprocess_impedance_data(Z, freq, threshold=5e-2)
    kwargs = {
        "iters": 2,
        "complexity": 12,
        "population_size": 10,
        "generations": 5,
        "tol": 1e-2,
        "parallel": False,
    }
    circuits = core.generate_equivalent_circuits(Z, freq, **kwargs)
    assert isinstance(circuits, pd.DataFrame)


def test_gep_parallel():
    Z, freq = io.load_test_dataset()
    Z, freq, rmse = core.preprocess_impedance_data(Z, freq, threshold=5e-2)
    kwargs = {
        "iters": 2,
        "complexity": 12,
        "population_size": 10,
        "generations": 5,
        "tol": 1e-2,
        "parallel": True,
    }
    circuits = core.generate_equivalent_circuits(Z, freq, **kwargs)
    assert isinstance(circuits, pd.DataFrame)

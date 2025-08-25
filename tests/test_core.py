import autoeis as ae
import numpy as np
import pandas as pd
import pytest


def test_compute_ohmic_resistance():
    circuit_string = "R1-[P2,P3-R4]"
    circuit_fn = ae.utils.generate_circuit_fn(circuit_string, backend="impedance")
    R1 = 250
    parameters = np.array([R1, 1e-3, 0.1, 5e-5, 0.8, 10])
    freq = np.logspace(-3, 3, 1000)
    Z = circuit_fn(freq, parameters)
    R = ae.core.compute_ohmic_resistance(freq, Z)
    np.testing.assert_allclose(R, R1, rtol=0.15)


def test_compute_ohmic_resistance_missing_high_freq():
    circuit_string = "R1-[P2,P3-R4]"
    circuit_fn = ae.utils.generate_circuit_fn(circuit_string, backend="impedance")
    R1 = 250
    parameters = np.array([R1, 1e-3, 0.1, 5e-5, 0.8, 10])
    freq = np.logspace(-3, 0, 1000)
    Z = circuit_fn(freq, parameters)
    R = ae.core.compute_ohmic_resistance(freq, Z)
    # When high frequency measurements are missing, Re(Z) @ max(freq) is good approximation
    Zreal_at_high_freq = Z.real[np.argmax(freq)]
    np.testing.assert_allclose(R, Zreal_at_high_freq)


def test_gep():
    def test_gep_serial():
        freq, Z = ae.io.load_test_dataset()
        freq, Z = ae.utils.preprocess_impedance_data(freq, Z, tol_linKK=5e-2)
        kwargs = {
            "iters": 2,
            "complexity": 2,
            "population_size": 5,
            "generations": 2,
            "tol": 1e10,
            "parallel": False,
        }
        circuits = ae.core.generate_equivalent_circuits(freq, Z, **kwargs)
        assert len(circuits) == kwargs["iters"]
        assert isinstance(circuits, pd.DataFrame)

    def test_gep_parallel():
        freq, Z = ae.io.load_test_dataset(preprocess=True)
        kwargs = {
            "iters": 2,
            "complexity": 2,
            "population_size": 5,
            "generations": 2,
            "tol": 1e10,
            "parallel": True,
        }
        circuits = ae.core.generate_equivalent_circuits(freq, Z, **kwargs)
        assert len(circuits) == kwargs["iters"]
        assert isinstance(circuits, pd.DataFrame)

    test_gep_serial()
    test_gep_parallel()


def test_filter_implausible_circuits():
    circuits_unfiltered = ae.io.load_test_circuits()
    N1 = len(circuits_unfiltered)
    circuits = ae.core.filter_implausible_circuits(circuits_unfiltered)
    N2 = len(circuits)
    assert N2 < N1


def test_bayesian_inference_single_circuit_single_data():
    freq, Z = ae.io.load_test_dataset(preprocess=True)
    circuits = ae.io.load_test_circuits(filtered=True)
    circuit = circuits.iloc[0].circuitstring
    p0 = circuits.iloc[0].Parameters
    kwargs_mcmc = {"num_warmup": 25, "num_samples": 10, "progress_bar": False}
    result = ae.core.perform_bayesian_inference(circuit, freq, Z, p0, **kwargs_mcmc)[0]
    assert isinstance(result, ae.utils.InferenceResult)
    assert result.converged


def test_bayesian_inference_single_circuit_multiple_data():
    # Load test dataset N times with noise to simulate multiple datasets
    n_datasets = 2
    freq, Z = zip(
        *[ae.io.load_test_dataset(preprocess=True, noise=0.1) for _ in range(n_datasets)]
    )
    circuits = ae.io.load_test_circuits(filtered=True)
    circuit = circuits.iloc[0].circuitstring
    p0 = circuits.iloc[0].Parameters
    kwargs_mcmc = {"num_warmup": 25, "num_samples": 10, "progress_bar": False}
    results = ae.core.perform_bayesian_inference(circuit, freq, Z, p0, **kwargs_mcmc)
    assert len(results) == n_datasets
    for result in results:
        assert isinstance(result, ae.utils.InferenceResult)
        assert result.converged


def test_bayesian_inference_multiple_circuits_single_data():
    freq, Z = ae.io.load_test_dataset(preprocess=True)
    circuits = ae.io.load_test_circuits(filtered=True)
    circuits = circuits.iloc[:2]  # Test the first two circuits to save CI time
    kwargs_mcmc = {"num_warmup": 25, "num_samples": 10, "progress_bar": False}
    results = ae.core.perform_bayesian_inference(circuits, freq, Z, **kwargs_mcmc)
    assert len(results) == len(circuits)
    for result in results:
        assert isinstance(result, ae.utils.InferenceResult)
        assert result.converged


# TODO: Use a simple circuit to generate test data so this test doesn't take too long
@pytest.mark.skip(reason="This test is too slow!")
def test_perform_full_analysis():
    freq, Z = ae.io.load_test_dataset()
    results = ae.core.perform_full_analysis(freq, Z)
    required_columns = ["circuitstring", "Parameters", "MCMC", "success", "divergences"]
    assert all(col in results.columns for col in required_columns)

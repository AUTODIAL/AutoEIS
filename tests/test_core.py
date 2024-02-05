import numpy as np
import numpyro
import pandas as pd
import pytest

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


def test_filter_implausible_circuits():
    Z, freq = io.load_test_dataset()
    circuits_unfiltered = io.load_test_circuits()
    N1 = len(circuits_unfiltered)
    circuits = core.filter_implausible_circuits(circuits_unfiltered)
    N2 = len(circuits)
    assert N2 < N1


def test_bayesian_inference_single():
    Z, freq = io.load_test_dataset()
    circuits = io.load_test_circuits(filtered=True)
    circuit = circuits.iloc[0].circuitstring
    p0 = circuits.iloc[0].Parameters
    kwargs_mcmc = {
        "num_warmup": 2500,
        "num_samples": 1000,
        "progress_bar": False,
    }
    mcmc, exist_code = core._perform_bayesian_inference(
        circuit, Z, freq, p0, **kwargs_mcmc
    )
    assert exist_code in [-1, 0]
    assert isinstance(mcmc, numpyro.infer.mcmc.MCMC)


def test_bayesian_inference_batch():
    Z, freq = io.load_test_dataset()
    circuits = io.load_test_circuits(filtered=True)
    mcmc_results = core.perform_bayesian_inference(circuits, Z, freq)
    assert len(mcmc_results) == len(circuits)
    for mcmc, exist_code in mcmc_results:
        assert exist_code in [-1, 0]
        assert isinstance(mcmc, numpyro.infer.mcmc.MCMC)


@pytest.mark.skip(reason="This test is too slow!")
def test_perform_full_analysis():
    Z, freq = io.load_test_dataset()
    results = core.perform_full_analysis(Z, freq)
    required_columns = [
        "circuitstring",
        "Parameters",
        "MCMC (chains)",
        "MCMC (status)",
    ]
    assert all(col in results.columns for col in required_columns)

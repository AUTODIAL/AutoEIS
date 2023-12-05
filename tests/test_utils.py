import numpy as np
import sklearn.metrics as skmetrics
from impedance.models.circuits import CustomCircuit

from autoeis import utils

# Real numbers
x1 = np.random.rand(10)
x2 = np.random.rand(10)
# Complex numbers
y1 = x1 + np.zeros(10) * 1j
y2 = x2 + np.zeros(10) * 1j

# Simulated EIS data
circuit_string = "R1-[P2,R3]"
p0_dict = {"R1": 250, "P2w": 1e-3, "P2n": 0.5, "R3": 10}
p0_vals = list(p0_dict.values())
circuit = CustomCircuit(
    utils.impedancepy_circuit(circuit_string),
    initial_guess=p0_vals,
)
circuit.parameters_ = p0_vals
freq = np.logspace(-3, 3, 1000)
Z = circuit.predict(freq)


def test_mse_score_real():
    mse = utils.mse_score(x1, x2)
    mse_gt = skmetrics.mean_squared_error(x1, x2)
    assert np.isclose(mse, mse_gt)


def test_mse_score_complex():
    mse = utils.mse_score(y1, y2)
    mse_gt = skmetrics.mean_squared_error(x1, x2)
    assert np.isclose(mse, mse_gt)


def test_rmse_score_real():
    rmse = utils.rmse_score(x1, x2)
    rmse_gt = skmetrics.mean_squared_error(x1, x2, squared=False)
    assert np.isclose(rmse, rmse_gt)


def test_rmse_score_complex():
    rmse = utils.rmse_score(y1, y2)
    rmse_gt = skmetrics.mean_squared_error(x1, x2, squared=False)
    assert np.isclose(rmse, rmse_gt)


def test_mape_score_real():
    mape = utils.mape_score(x1, x2)
    mape_gt = skmetrics.mean_absolute_percentage_error(x1, x2) * 100
    assert np.isclose(mape, mape_gt)


def test_mape_score_complex():
    mape = utils.mape_score(y1, y2)
    mape_gt = skmetrics.mean_absolute_percentage_error(x1, x2) * 100
    assert np.isclose(mape, mape_gt)


def test_r2_score_real():
    r2 = utils.r2_score(x1, x2)
    r2_gt = skmetrics.r2_score(x1, x2)
    assert np.isclose(r2, r2_gt)


def test_r2_score_complex():
    r2 = utils.r2_score(y1, y2)
    r2_gt = skmetrics.r2_score(x1, x2)
    assert np.isclose(r2, r2_gt)
    

def test_fit_circuit_parameters_no_initial_guess():
    p_dict = utils.fit_circuit_parameters(circuit_string, Z, freq)
    p_fit = list(p_dict.values())
    assert np.allclose(p_fit, p0_vals, rtol=0.01)


def test_fit_circuit_parameters_with_initial_guess():
    # Add some noise to the initial guess to test robustness
    p0 = p0_vals + np.random.rand(len(p0_vals)) * p0_vals * 0.5
    p_dict = utils.fit_circuit_parameters(circuit_string, Z, freq, p0)
    p_fit = list(p_dict.values())
    assert np.allclose(p_fit, p0_vals, rtol=0.01)

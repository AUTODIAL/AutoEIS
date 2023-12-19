import numpy as np
import pytest
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


def test_get_component_labels():
    circuit = "[R1,R2-P12]-L2-R22-[R6,C7-[L8,R5],L9]-R3"
    # Pass default types = None (all components)
    components = utils.get_component_labels(circuit)
    components_gt = ["R1", "R2", "P12", "L2", "R22", "R6", "C7", "L8", "R5", "L9", "R3"]
    assert components == components_gt
    # Pass string as types
    components = utils.get_component_labels(circuit, types="P")
    components_gt = ["P12"]
    assert components == components_gt
    # Pass list as types
    components = utils.get_component_labels(circuit, types=["L", "P"])
    components_gt = ["P12", "L2", "L8", "L9"]
    assert components == components_gt


def test_get_component_types():
    circuit = "[R1,R2-P12]-L2-R22-[R6,C7-[L8,R5],L9]-R3"
    types_gt = ["R", "R", "P", "L", "R", "R", "C", "L", "R", "L", "R"]
    types = utils.get_component_types(circuit)
    assert types == types_gt


def test_get_parameter_labels():
    circuit = "[R1,R2-P12]-L2-R22-[R6,C7-[L8,R5],L9]-R3"
    # Use default types = None (all parameters)
    variables = utils.get_parameter_labels(circuit)
    variables_gt = ["R1", "R2", "P12w", "P12n", "L2", "R22", "R6", "C7", "L8", "R5", "L9", "R3"]
    assert variables == variables_gt
    # Pass string as types
    variables = utils.get_parameter_labels(circuit, types="R")
    variables_gt = ["R1", "R2", "R22", "R6", "R5", "R3"]
    assert variables == variables_gt
    # Pass list as types
    variables = utils.get_parameter_labels(circuit, types=["R", "P"])
    variables_gt = ["R1", "R2", "P12w", "P12n", "R22", "R6", "R5", "R3"]
    assert variables == variables_gt


def test_count_params():
    circuit = "[R1,R2-P12]-L2-R22-[R6,C7-[L8,R5],L9]-R3"
    num_params_gt = 12
    num_params = utils.count_params(circuit)
    assert num_params == num_params_gt
    circuit = ""
    num_params_gt = 0
    num_params = utils.count_params(circuit)
    assert num_params == num_params_gt
    circuit = "[P1,P2]-P3-R4"
    num_params_gt = 7
    num_params = utils.count_params(circuit)
    assert num_params == num_params_gt    


def test_validate_circuit_string():
    # Valid circuit
    circuit = "[R1,R2-P12]-L2-R22-[R6,C7-[L8,R5],L9]-R3"
    utils.validate_circuit_string(circuit)
    # Empty circuit
    circuit = ""
    with pytest.raises(AssertionError):
        utils.validate_circuit_string(circuit)
    # Duplicate component
    circuit = "[R1,R2]-R1"
    with pytest.raises(AssertionError):
        utils.validate_circuit_string(circuit)


def test_find_ohmic_resistors():
    # No ohmic resistors
    circuit = "[R1,R2-P12]-L2-[R6,C7-[L8,R5],L9]-P3"
    ohmic_gt = []
    ohmic = utils.find_ohmic_resistors(circuit)
    assert ohmic == ohmic_gt
    # Single ohmic resistor
    circuit = "[R1,R2-P12]-L2-[R6,C7-[L8,R5],L9]-R3"
    ohmic_gt = ["R3"]
    ohmic = utils.find_ohmic_resistors(circuit)
    assert ohmic == ohmic_gt
    # Multiple ohmic resistors
    circuit = "[R1,R2-P12]-L2-R9-[R6,C7-[L8,R5],L9]-R8"
    ohmic_gt = ["R9", "R8"]
    ohmic = utils.find_ohmic_resistors(circuit)
    assert ohmic == ohmic_gt


def test_find_series_elements():
    # Valid circuit with series elements
    circuit = "[R1,R2-P12]-L2-[R6,C7-[L8,R5],L9]-P3"
    series_gt = ["L2", "P3"]
    series = utils.find_series_elements(circuit)
    assert series == series_gt
    # Valid circuit with N > 1 series elements in a row
    circuit = "[R1,R2-P12]-L2-[R6,C7-[L8,R5],L9]-P3-P4"
    series_gt = ["L2", "P3", "P4"]
    series = utils.find_series_elements(circuit)
    assert series == series_gt
    # Empty circuit
    circuit = ""
    series_gt = []
    series = utils.find_series_elements(circuit)
    assert series == series_gt
    # No series elements
    circuit = "[P1,P2]-[P3,P4]"
    series_gt = []
    series = utils.find_series_elements(circuit)
    assert series == series_gt


def test_get_parameter_types():
    circuit = "[R1,R2-P12]-L2-[R6,C7-[L8,R5],L9]-P3"
    types_gt = ["R", "R", "Pw", "Pn", "L", "R", "C", "L", "R", "L", "Pw", "Pn"]
    types_gt_unique = list(set(types_gt))
    types = utils.get_parameter_types(circuit, unique=False)
    types_unique = utils.get_parameter_types(circuit, unique=True)
    assert types == types_gt
    assert types_unique == types_gt_unique

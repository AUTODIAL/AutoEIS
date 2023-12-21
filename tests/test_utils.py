import numpy as np
from impedance.models.circuits import CustomCircuit

from autoeis import parser, utils

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
    parser.impedancepy_circuit(circuit_string),
    initial_guess=p0_vals,
)
circuit.parameters_ = p0_vals
freq = np.logspace(-3, 3, 1000)
Z = circuit.predict(freq)


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


def test_find_ohmic_resistors():
    # No ohmic resistors
    circuit = "[R1,R2-P12]-L2-[R6,C7-[L8,R5],L9]-P3"
    ohmic_gt = []
    ohmic = parser.find_ohmic_resistors(circuit)
    assert ohmic == ohmic_gt
    # Single ohmic resistor
    circuit = "[R1,R2-P12]-L2-[R6,C7-[L8,R5],L9]-R3"
    ohmic_gt = ["R3"]
    ohmic = parser.find_ohmic_resistors(circuit)
    assert ohmic == ohmic_gt
    # Multiple ohmic resistors
    circuit = "[R1,R2-P12]-L2-R9-[R6,C7-[L8,R5],L9]-R8"
    ohmic_gt = ["R9", "R8"]
    ohmic = parser.find_ohmic_resistors(circuit)
    assert ohmic == ohmic_gt


def test_circuit_complexity():
    circuit_complexity_dict = {
        "R1": [0],
        "[R1,R2]": [1, 1],
        "R1-C2": [0, 0],
        "R1-[C2,L3]": [0, 1, 1],
        "[R1,R2-R3]-[C4,L5]-P6": [1, 1, 1, 1, 1, 0],
        "R1-[R2,R3]-[[C4,L5]-P6]-[R7,[R8,[C9,L10]]]": [0, 1, 1, 2, 2, 1, 1, 2, 3, 3],
    }
    for cstr, cc in circuit_complexity_dict.items():
        assert utils.circuit_complexity(cstr) == cc

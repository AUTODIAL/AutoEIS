import pytest

from autoeis import parser


def test_get_component_labels():
    circuit = "[R1,R2-P12]-L2-R22-[R6,C7-[L8,R5],L9]-R3"
    # Pass default types = None (all components)
    components = parser.get_component_labels(circuit)
    components_gt = ["R1", "R2", "P12", "L2", "R22", "R6", "C7", "L8", "R5", "L9", "R3"]
    assert components == components_gt
    # Pass string as types
    components = parser.get_component_labels(circuit, types="P")
    components_gt = ["P12"]
    assert components == components_gt
    # Pass list as types
    components = parser.get_component_labels(circuit, types=["L", "P"])
    components_gt = ["P12", "L2", "L8", "L9"]
    assert components == components_gt


def test_get_parameter_labels():
    circuit = "[R1,R2-P12]-L2-R22-[R6,C7-[L8,R5],L9]-R3"
    # Use default types = None (all parameters)
    variables = parser.get_parameter_labels(circuit)
    variables_gt = ["R1", "R2", "P12w", "P12n", "L2", "R22", "R6", "C7", "L8", "R5", "L9", "R3"]
    assert variables == variables_gt
    # Pass string as types
    variables = parser.get_parameter_labels(circuit, types="R")
    variables_gt = ["R1", "R2", "R22", "R6", "R5", "R3"]
    assert variables == variables_gt
    # Pass list as types
    variables = parser.get_parameter_labels(circuit, types=["R", "P"])
    variables_gt = ["R1", "R2", "P12w", "P12n", "R22", "R6", "R5", "R3"]
    assert variables == variables_gt


def test_get_component_types():
    circuit = "[R1,R2-P12]-L2-R22-[R6,C7-[L8,R5],L9]-R3"
    types_gt = ["R", "R", "P", "L", "R", "R", "C", "L", "R", "L", "R"]
    types = parser.get_component_types(circuit)
    assert types == types_gt


def test_get_parameter_types():
    circuit = "[R1,R2-P12]-L2-[R6,C7-[L8,R5],L9]-P3"
    types_gt = ["R", "R", "Pw", "Pn", "L", "R", "C", "L", "R", "L", "Pw", "Pn"]
    types_gt_unique = list(set(types_gt))
    types = parser.get_parameter_types(circuit, unique=False)
    types_unique = parser.get_parameter_types(circuit, unique=True)
    assert types == types_gt
    assert types_unique == types_gt_unique


def test_count_parameters():
    circuit = "[R1,R2-P12]-L2-R22-[R6,C7-[L8,R5],L9]-R3"
    num_params_gt = 12
    num_params = parser.count_parameters(circuit)
    assert num_params == num_params_gt
    circuit = ""
    num_params_gt = 0
    num_params = parser.count_parameters(circuit)
    assert num_params == num_params_gt
    circuit = "[P1,P2]-P3-R4"
    num_params_gt = 7
    num_params = parser.count_parameters(circuit)
    assert num_params == num_params_gt    


def test_validate_circuit():
    # Valid circuit
    circuit = "[R1,R2-P12]-L2-R22-[R6,C7-[L8,R5],L9]-R3"
    parser.validate_circuit(circuit)
    # Empty circuit
    circuit = ""
    with pytest.raises(AssertionError):
        parser.validate_circuit(circuit)
    # Duplicate component
    circuit = "[R1,R2]-R1"
    with pytest.raises(AssertionError):
        parser.validate_circuit(circuit)


def test_find_series_elements():
    # Valid circuit with series elements
    circuit = "[R1,R2-P12]-L2-[R6,C7-[L8,R5],L9]-P3"
    series_gt = ["L2", "P3"]
    series = parser.find_series_elements(circuit)
    assert series == series_gt
    # Valid circuit with N > 1 series elements in a row
    circuit = "[R1,R2-P12]-L2-[R6,C7-[L8,R5],L9]-P3-P4"
    series_gt = ["L2", "P3", "P4"]
    series = parser.find_series_elements(circuit)
    assert series == series_gt
    # Empty circuit
    circuit = ""
    series_gt = []
    series = parser.find_series_elements(circuit)
    assert series == series_gt
    # No series elements
    circuit = "[P1,P2]-[P3,P4]"
    series_gt = []
    series = parser.find_series_elements(circuit)
    assert series == series_gt

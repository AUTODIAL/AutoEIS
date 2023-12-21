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


def test_group_parameters_by_component():
    circuit = "[R1,R2-P3]-L4-L5-[P6,R7]"
    g = {
        "R": ["R1", "R2", "R7"],
        "P": ["P3w", "P3n", "P6w", "P6n"],
        "L": ["L4", "L5"]
    }
    assert parser.group_parameters_by_component(circuit) == g


def test_group_parameters_by_type():
    circuit = "[R1,R2-P3]-L4-L5-[P6,R7]"
    g = {
        "R": ["R1", "R2", "R7"],
        "Pw": ["P3w", "P6w"],
        "Pn": ["P3n", "P6n"],
        "L": ["L4", "L5"]
    }
    assert parser.group_parameters_by_type(circuit) == g


def test_count_parameters():
    d = {
        "[R1,R2-P12]-L2-R22-[R6,C7-[L8,R5],L9]-R3": 12,
        "": 0,
        "[P1,P2]-P3-R4": 7
    }
    for circuit, num_params_gt in d.items():
        assert parser.count_parameters(circuit) == num_params_gt


def test_validate_circuit():
    # Valid circuits
    circuits_valid = [
        "[R1,R2-P12]-L2-R22-[R6,C7-[L8,R5],L9]-R3"
    ]
    for circuit in circuits_valid:
        parser.validate_circuit(circuit)
    # Invalid circuits
    circuits_invalid = [
        "",
        "[R1,R2]-R1"
    ]
    for circuit in circuits_invalid:
        with pytest.raises(AssertionError):
            parser.validate_circuit(circuit)


def test_validate_parameter():
    # Valid parameters
    parameters_valid = ["R1", "C5", "L12", "P1n", "P22w"]
    for p in parameters_valid:
        parser.validate_parameter(p)
    # Invalid parameters
    parameter_invalid = ["R1w", "C", "L2r", "Pn", "P2s"]
    for p in parameter_invalid:
        with pytest.raises(AssertionError):
            parser.validate_parameter(p)


def test_parse_component():
    d = {
        "R2": "R",
        "C5": "C",
        "L12": "L",
        "P1n": "P",
        "P22w": "P",
        "P4": "P"
    }
    for k, v in d.items():
        assert parser.parse_component(k) == v


def test_parse_parameter():
    d = {
        "R2": "R",
        "C5": "C",
        "L12": "L",
        "P1n": "Pn",
        "P22w": "Pw",
    }
    for k, v in d.items():
        assert parser.parse_parameter(k) == v

def test_find_series_elements():
    d = {
        "[R1,R2-P12]-L2-[R6,C7-[L8,R5],L9]-P3": ["L2", "P3"],
        "[R1,R2-P12]-L2-[R6,C7-[L8,R5],L9]-P3-P4": ["L2", "P3", "P4"],
        "": [],
        "[P1,P2]-[P3,P4]": []
    }
    for k, v in d.items():
        assert parser.find_series_elements(k) == v


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
    

def test_convert_to_impedance_format():
    circuit = "R1-[R2,P1-[R5,L8]]-P5"
    circuit_impy = parser.convert_to_impedance_format(circuit)
    circuit_gt = "R1-p(R2,CPE1-p(R5,L8))-CPE5"
    assert circuit_impy == circuit_gt

"""
Collection of functions for parsing circuit strings.

.. currentmodule:: autoeis.parser

.. autosummary::
   :toctree: generated/

    validate_circuit
    validate_parameter
    parse_component
    parse_parameter
    get_component_labels
    get_component_types
    get_parameter_types
    get_parameter_labels
    group_parameters_by_type
    group_parameters_by_component
    count_parameters
    convert_to_impedance_format
    find_ohmic_resistors

"""
import re

from numpy import pi  # noqa: F401
from pyparsing import nested_expr


def validate_circuit(circuit: str) -> bool:
    """Checks if a circuit string is valid."""
    # Check for duplicate elements
    components = get_component_labels(circuit)
    duplicates = [e for e in components if components.count(e) > 1]
    assert not duplicates, f"Duplicate elements found: {set(duplicates)}"
    # Test circuit is not empty
    assert len(circuit) > 0, "Circuit string is empty."
    # Check for valid element names
    valid_types = ["R", "C", "L", "P"]
    types = get_component_types(circuit)
    for t in types:
        assert t in valid_types, f"Invalid element type: {t}"
    # TODO: Check for parallel elements with < 2 elements
    # TODO: Check for duplicate "-" or "," symbols
    # TODO: Check for disconnected elements, eg R1R2 or R1[R2,R3]
    return True  # If all checks pass, the circuit is considered valid


def validate_parameter(p: str) -> bool:
    """Checks if a parameter label is valid."""
    # Check if parameter label is a string
    assert isinstance(p, str), "Parameter label must be a string."
    # Check if parameter label is not empty
    assert len(p) > 0, "Parameter label is empty."
    # Check if parameter label is valid
    pattern = r"(?:R\d+|C\d+|L\d+|P\d+[wn])"
    assert re.fullmatch(pattern, p), f"Invalid parameter label: {p}"
    return True  # If all checks pass, the parameter label is considered valid


def parse_component(c: str) -> str:
    """Returns the type of a component label."""
    return re.match(r"[A-Za-z]+", c).group()


def parse_parameter(p: str) -> str:
    """Returns the type of a parameter label."""
    validate_parameter(p)
    if p.startswith(("R", "C", "L")):
        ptype = p[0]
    elif p.startswith("P") and p.endswith("w"):
        ptype = "Pw"
    elif p.startswith("P") and p.endswith("n"):
        ptype = "Pn"
    return ptype


def get_component_labels(circuit: str, types: list[str] = None) -> list[str]:
    """Returns a list of labels for all components in a circuit string."""
    types = [types] if isinstance(types, str) else types
    types = ["R", "C", "L", "P"] if types is None else types
    assert isinstance(types, list), "types must be a list of strings."
    pattern = rf'\b(?:{"|".join(types)})\d+\b'
    return re.findall(pattern, circuit)


def get_component_types(circuit: str, unique: bool = False) -> list[str]:
    """Returns a list of component types in a circuit string."""
    types = re.findall(r"[A-Za-z]+", circuit)
    return list(set(types)) if unique else types


def get_parameter_labels(circuit: str, types: list[str] = None) -> list[str]:
    """Returns a list of labels for all parameters in a circuit string."""
    types = [types] if isinstance(types, str) else types
    types = ["R", "C", "L", "P"] if types is None else types
    assert isinstance(types, list), "types must be a list of strings."
    components = get_component_labels(circuit, types=types)
    parameters = []
    for component in components:
        # CPE elements have two parameters P{i}w and P{i}n
        if component.startswith("P"):
            parameters.extend([f"{component}w", f"{component}n"])
        else:
            parameters.append(component)
    return parameters


def get_parameter_types(circuit: str, unique: bool = False) -> list[str]:
    """Returns a list of parameter types in a circuit string."""
    ptypes = [parse_parameter(p) for p in get_parameter_labels(circuit)]
    return list(set(ptypes)) if unique else ptypes


def group_parameters_by_type(circuit: str) -> dict[str, list[str]]:
    """Groups parameter labels by component type."""
    params = get_parameter_labels(circuit)
    ptypes = get_parameter_types(circuit)
    groups = {ptype: [] for ptype in set(ptypes)}
    for param, ptype in zip(params, ptypes):
        groups[ptype].append(param)
    return groups


def group_parameters_by_component(circuit: str) -> dict[str, list[str]]:
    """Groups parameter labels by component label."""
    ctypes = get_component_types(circuit)
    params_by_component = {ctype: [] for ctype in ctypes}
    params = get_parameter_labels(circuit)
    for param in params:
        ctype = parse_component(param)
        params_by_component[ctype].append(param)
    return params_by_component


def count_parameters(circuit: str) -> int:
    """Returns the number of parameters that fully describe a circuit string."""
    return len(get_parameter_labels(circuit))


def convert_to_impedance_format(circuit: str) -> str:
    """Converts a circuit string the format used by impedance.py."""
    circuit = circuit.replace("P", "CPE")
    circuit = circuit.replace("[", "p(")
    circuit = circuit.replace("]", ")")
    return circuit


def circuit_to_nested_expr(circuit: str) -> list:
    """Parses a circuit string to a nested list[str]."""

    def cleanup(lst: list, chars: list[str]):
        """Removes leading/trailing chars from a nested list[str]."""
        chars = "".join(chars)
        result = []
        for el in lst:
            if isinstance(el, list):
                result.append(cleanup(el, chars))
            else:
                # Don't add empty strings
                if el.strip(chars):
                    result.append(el.strip(chars))
        return result

    def parse(circuit: str):
        # Enclose circuit with brackets to make it a valid nested expression
        circuit = f"[{circuit}]"
        parser = nested_expr(opener="[", closer="]")
        parsed = parser.parse_string(circuit, parse_all=True).as_list()
        return parsed

    expr = parse(circuit)
    expr = cleanup(expr, chars=[",", "-"])
    return expr[0]


def find_series_elements(circuit: str) -> list[str]:
    """Extracts the series componenets from a circuit (in the main chain)."""
    parsed = circuit_to_nested_expr(circuit)
    series_elements = [el for el in parsed if isinstance(el, str)]
    series_elements = re.findall(r"[A-Z]+\d+", str(series_elements))
    return series_elements


def find_ohmic_resistors(circuit: list) -> list[str]:
    """Finds all ohmic resistors in a nested circuit expression."""
    series_elements = find_series_elements(circuit)
    return re.findall(r"R\d+", str(series_elements))


def generate_mathematical_expr(circuit: str) -> str:
    """Converts a circuit string to a mathematical expression for impedance.

    The returned string can be evaluated assuming 'p' is an array of
    component values and 'f' is the frequency (scalar/array).
    """
    # Apply series-parallel conversion, e.g., [R1,R2] -> (1/R1+1/R2)**(-1)
    replacements = {
        "-": "+",
        "[": "((",
        ",": ")**(-1)+(",
        "]": ")**(-1))**(-1)"
    }  # fmt: off
    expr = circuit
    for j, k in replacements.items():
        expr = expr.replace(j, k)

    # Embed impedance expressions, e.g., C1 -> (1/(2*1j*pi*f*C1))
    expr = replace_components_with_impedance(expr)

    # Replace parameters with array indexing, e.g., R1, P2w, P2n -> x[0], x[1], x[2]
    parameters = get_parameter_labels(circuit)
    for i, var in enumerate(parameters):
        expr = expr.replace(var, f"p[{i}]", 1)

    return expr


def replace_components_with_impedance(expr: str) -> str:
    """Updates the circuit expression with the impedance of the components.

    Notes
    -----
    The impedance expressions are described 'parameter' labels, not 'component'
    labels, although the two are often the same (except for components defined
    by more than one parameter). For example, a CPE element P1 has impedance
    expression as a function of of P1w and P1n.
    """

    def replacement(var):
        eltype = get_component_types(var)[0]
        return {
            "R": f"{var}",
            "C": f"(1/(2*1j*pi*f*{var}))",
            "L": f"(2*1j*pi*f*{var})",
            "P": f"(1/({var}w*(2*1j*pi*f)**{var}n))",
        }[eltype]

    # Get component lables
    components = get_component_labels(expr)
    # Replace components with impedance expression, e.g., C1 -> (1/(2*1j*pi*f*C1))
    for c in components:
        expr = expr.replace(c, replacement(c))
    return expr

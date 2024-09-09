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
    """Checks if a circuit string is valid.

    This function ensures that the circuit string:
        - is not empty,
        - contains valid element names (``R``, ``C``, ``L``, ``P``),
        - and, doesn't contain duplicate elements, e.g., ``R1-[P2,P2]``.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    bool
        True if the circuit string is valid, False otherwise.
    """
    # TODO: Check for parallel elements with < 2 elements
    # TODO: Check for duplicate "-" or "," symbols
    # TODO: Check for disconnected elements, eg R1R2 or R1[R2,R3]
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
    return True  # If all checks pass, the circuit is considered valid


def validate_parameter(p: str, raises: bool = True) -> bool:
    """Checks if a parameter label is valid.

    Valid parameter labels: {R,C,L,Pw,Pn}{N} where N is a number, e.g., P1n.

    Parameters
    ----------
    p : str
        String representation of the parameter label.
    raises : bool, optional
        If True, raises an AssertionError on invalid parameter labels.
        Default is True.

    Returns
    -------
    bool
        True if the parameter label is valid, False otherwise.
    """
    # Check if parameter label is a string
    if not isinstance(p, str):
        if raises:
            raise AssertionError("Parameter label must be a string.")
        return False
    # Check if parameter label is not empty
    if not p:
        if raises:
            raise AssertionError("Parameter label is empty.")
        return False
    # Check if parameter label is valid
    pattern = r"(?:R\d+|C\d+|L\d+|P\d+[wn])"
    if not re.fullmatch(pattern, p):
        if raises:
            raise AssertionError(f"Invalid parameter label: {p}")
        return False
    return True


def parse_component(c: str) -> str:
    """Returns the component type of a component/parameter label.

    Parameters
    ----------
    c : str
        String representation of a component/parameter label.

    Returns
    -------
    str
        The type of the component label from the set {R,C,L,P}.

    Examples
    --------
    >>> parse_component("R1")
    'R'
    >>> parse_component("P2n")
    'P'
    """
    return re.match(r"[A-Za-z]+", c).group()


def parse_parameter(p: str) -> str:
    """Returns the type of a parameter label.

    Parameters
    ----------
    p : str
        String representation of the parameter label.

    Returns
    -------
    str
        The type of the parameter label from the set {R,C,L,Pn,Pw}.

    Examples
    --------
    >>> parse_parameter("R1")
    'R'
    >>> parse_parameter("P2n")
    'Pn'
    """
    validate_parameter(p)
    if p.startswith(("R", "C", "L")):
        ptype = p[0]
    elif p.startswith("P") and p.endswith("w"):
        ptype = "Pw"
    elif p.startswith("P") and p.endswith("n"):
        ptype = "Pn"
    return ptype


def get_component_labels(circuit: str, types: list[str] = None) -> list[str]:
    """Returns a list of labels for all components in a circuit string.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    types : list[str], optional
        List of component types to filter by. Default is None.

    Returns
    -------
    list[str]
        A list of component labels.

    Examples
    --------
    >>> get_component_labels("R1-[R2,P4]")
    ['R1', 'R2', 'P4']
    >>> get_component_labels("R1-[R2,P4]", types=["R"])
    ['R1', 'R2']
    """
    types = [types] if isinstance(types, str) else types
    types = ["R", "C", "L", "P"] if types is None else types
    assert isinstance(types, list), "types must be a list of strings."
    pattern = rf'\b(?:{"|".join(types)})\d+\b'
    return re.findall(pattern, circuit)


def get_component_types(circuit: str, unique: bool = False) -> list[str]:
    """Returns a list of component types in a circuit string.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    unique : bool, optional
        If True, returns a list of unique component types. Default is False.

    Returns
    -------
    list[str]
        A list of component types.

    Examples
    --------
    >>> get_component_types("R1-[R2,P4]")
    ['R', 'R', 'P']
    >>> get_component_types("R1-[R2,P4]", unique=True)
    ['P', 'R']
    """
    types = re.findall(r"[A-Za-z]+", circuit)
    return list(set(types)) if unique else types


def get_parameter_labels(circuit: str, types: list[str] = None) -> list[str]:
    """Returns a list of labels for all parameters in a circuit string.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    types : list[str], optional
        List of parameter types to filter by. Default is None.

    Returns
    -------
    list[str]
        A list of parameter labels.

    Examples
    --------
    >>> get_parameter_labels("R1-[R2,P4]")
    ['R1', 'R2', 'P4w', 'P4n']
    >>> get_parameter_labels("R1-[R2,P4]", types=["R"])
    ['R1', 'R2']
    """
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
    """Returns a list of parameter types in a circuit string.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    unique : bool, optional
        If True, returns a list of unique parameter types. Default is False.

    Returns
    -------
    list[str]
        A list of parameter types.

    Examples
    --------
    >>> get_parameter_types("R1-[R2,P4]")
    ['R', 'R', 'Pw', 'Pn']
    >>> get_parameter_types("R1-[R2,P4]", unique=True)
    ['Pn', 'Pw', 'R']
    """
    ptypes = [parse_parameter(p) for p in get_parameter_labels(circuit)]
    return list(set(ptypes)) if unique else ptypes


def group_parameters_by_type(circuit: str) -> dict[str, list[str]]:
    """Groups parameter labels by component type.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    dict[str, list[str]]
        A dictionary of parameter labels grouped by component type.

    Examples
    --------
    >>> group_parameters_by_type("R1-[R2,P4]")
    {'Pn': ['P4n'], 'Pw': ['P4w'], 'R': ['R1', 'R2']}
    """
    params = get_parameter_labels(circuit)
    ptypes = get_parameter_types(circuit)
    groups = {ptype: [] for ptype in set(ptypes)}
    for param, ptype in zip(params, ptypes):
        groups[ptype].append(param)
    return groups


def group_parameters_by_component(circuit: str) -> dict[str, list[str]]:
    """Groups parameter labels by component label.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    dict[str, list[str]]
        A dictionary of parameter labels grouped by component label.

    Examples
    --------
    >>> group_parameters_by_component("R1-[R2,P4]")
    {'R': ['R1', 'R2'], 'P': ['P4w', 'P4n']}
    """
    ctypes = get_component_types(circuit)
    params_by_component = {ctype: [] for ctype in ctypes}
    params = get_parameter_labels(circuit)
    for param in params:
        ctype = parse_component(param)
        params_by_component[ctype].append(param)
    return params_by_component


def count_parameters(circuit: str) -> int:
    """Returns the number of parameters present in a circuit string.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    int
        The number of parameters present in the circuit.

    Examples
    --------
    >>> count_parameters("R1-[R2,P4]")
    4
    """
    return len(get_parameter_labels(circuit))


def convert_to_impedance_format(circuit: str) -> str:
    """Converts a circuit string the format used by impedance.py.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    str
        The circuit string in the format used by impedance.py.

    Examples
    --------
    >>> convert_to_impedance_format("R1-[R2,P4]")
    'R1-p(R2,CPE4)'
    """
    circuit = circuit.replace("P", "CPE")
    circuit = circuit.replace("[", "p(")
    circuit = circuit.replace("]", ")")
    return circuit


def circuit_to_nested_expr(circuit: str) -> list:
    """Parses a circuit string to a nested list[str].

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    list
        A nested list of component labels.

    Examples
    --------
    >>> circuit_to_nested_expr("R1-[R2,P4]")
    ['R1', ['R2,P4']]
    """

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
    """Extracts the series componenets from a circuit (in the main chain).

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    list[str]
        A list of series component labels.

    Examples
    --------
    >>> find_series_elements("R1-[R2,P4]-P5")
    ['R1', 'P5]
    """
    parsed = circuit_to_nested_expr(circuit)
    series_elements = [el for el in parsed if isinstance(el, str)]
    series_elements = re.findall(r"[A-Z]+\d+", str(series_elements))
    return series_elements


def find_ohmic_resistors(circuit: str) -> list[str]:
    """Finds all ohmic resistors in a circuit (only in the main chain).

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    list[str]
        A list of ohmic resistor labels.

    Examples
    --------
    >>> find_ohmic_resistors("R1-[R2,P4]-R5")
    ['R1', 'R5']
    """
    series_elements = find_series_elements(circuit)
    return re.findall(r"R\d+", str(series_elements))


def generate_mathematical_expression(circuit: str) -> str:
    """Converts a circuit string to a mathematical expression, parameterized
    by frequency and the circuit parameters, i.e., func(freq, p).

    The returned string can be evaluated assuming 'p' is an array of
    parameter values and 'freq' is the frequency (scalar/array).

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    str
        The mathematical expression for impedance.

    Examples
    --------
    >>> generate_mathematical_expr("R1-R2")
    'p[0]+(1/(p[1]*(2*1j*pi*freq)**p[2]))'
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
        # Negative look-ahead to avoid replacing R10 when dealing with R1
        expr = re.sub(rf"{var}(?!\d)", f"p[{i}]", expr)

    return expr


def replace_components_with_impedance(expr: str) -> str:
    """Expands the circuit expression with the impedance of the components.

    The circuit expression describes describes the impedance of a circuit,
    parameterized by component impedance values, e.g., '[R1,P2]' ->
    '1 / (1/R1 + 1/P2)'. This function parameterizes the impedence terms using
    the actual component values, e.g., R1 -> R1, P2 -> 1/(P2w*(2*1j*pi*freq)**P2n)

    Parameters
    ----------
    expr : str
        The circuit expression to be expanded.

    Returns
    -------
    str
        The expanded circuit expression.

    Examples
    --------
    >>> replace_components_with_impedance("R1")
    'R1'
    >>> replace_components_with_impedance("P1")
    '(1/(P1w*(2*1j*pi*freq)**P1n))'
    >>> replace_components_with_impedance("R1+P2")
    'R1+(1/(P2w*(2*1j*pi*freq)**P2n))'
    """

    def replacement(var):
        eltype = get_component_types(var)[0]
        return {
            "R": f"{var}",
            "C": f"(1/(2*1j*pi*freq*{var}))",
            "L": f"(2*1j*pi*freq*{var})",
            "P": f"(1/({var}w*(2*1j*pi*freq)**{var}n))",
        }[eltype]

    # Get component lables
    components = get_component_labels(expr)
    # Replace components with impedance expression, e.g., C1 -> (1/(2*1j*pi*freq*C1))
    for c in components:
        # Negative look-ahead to avoid replacing R10 when dealing with R1
        expr = re.sub(rf"{c}(?!\d)", replacement(c), expr)

    return expr

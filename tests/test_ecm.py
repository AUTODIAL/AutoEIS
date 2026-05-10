import numpy as np
import pandas as pd

import autoeis as ae


def test_parse_normalize_and_compare_ecm_topology():
    tree = ae.ecm.parse_circuit_tree("R1-[P2,R3]", preserve_labels=False)

    assert tree == ("series", ["R", ("parallel", ["P", "R"])])
    assert ae.ecm.compare_ecm_topology("R1-C2-L3", "L9-R8-C7")
    assert not ae.ecm.compare_ecm_topology("R1-C2-L3", "R1-C2-P3")


def test_simplify_ecm_preserves_representative_labels():
    simplified = ae.ecm.simplify_ecm("R1-R2-[C1-C2,R3]-R4")

    assert simplified == "R1-[C1,R3]"


def test_deduplicate_ecms_preserves_first_occurrence():
    circuits = ["R1-C2", "C9-R8", "R1-[P2,R3]", "R4-[R5,P6]"]

    unique = ae.ecm.deduplicate_ecms(circuits)

    assert unique == ["R1-C2", "R1-[P2,R3]"]


def test_ecm_vectorization_roundtrip():
    vector = ae.ecm.ecm_to_vector("R1-[P2,R3]", simplify=False)

    assert vector == [0, 2, 1, 5, 2, -3, -2]
    assert ae.ecm.vector_to_ecm(vector) == "R1-[P1,R2]"


def test_eis_vectorization_and_value_replacement():
    vector = ae.ecm.eis_to_vector("10-[2.5,30]-1e-6")

    assert vector == [0, 2, 10.0, 1, 2, 2.5, 2, 30.0, -3, 2, 1e-6, -2]
    assert ae.ecm.extract_eis_values(vector) == [10.0, 2.5, 30.0, 1e-6]
    replaced = ae.ecm.replace_eis_values(vector, [1, 2, 3, 4])
    assert ae.ecm.extract_eis_values(replaced) == [1.0, 2.0, 3.0, 4.0]


def test_simplify_matching_ecm_and_eis_pair():
    ecm_tree, eis_tree = ae.ecm.simplify_ecm_eis_pair("R1-R2-[C1,C2]", "10-20-[2,3]")

    assert ae.ecm.tree_to_circuit(ecm_tree) == "R1-C1"
    assert ae.ecm.tree_to_circuit(eis_tree) == "30-5"


def test_ecm_to_eis_string_and_parameter_scalar_ordering():
    parameter_values = {"R1": 10, "P2": 0.5, "R3": 30}

    eis_string = ae.ecm.ecm_to_eis_string("R1-[P2,R3]", parameter_values)

    assert eis_string == "10-[0.5,30]"


def test_compress_and_order_parameter_values():
    f_ref = 1 / (2 * np.pi)
    params = {"R1": 10, "P2w": 4, "P2n": 1, "C3": 1e-6}

    compressed = ae.ecm.compress_parameter_values(params, f_ref)
    ordered, vector = ae.ecm.order_parameters_by_topology("R1-[P2,C3]", compressed)

    assert compressed["P2"] == 4
    assert list(ordered) == ["R1", "P2", "C3"]
    assert vector == [10.0, 4.0, 1e-6]


def test_add_ecm_vectors_to_dataframe():
    circuits = pd.DataFrame({"circuitstring": ["R1-[P2,R3]"]})

    out = ae.ecm.add_ecm_vectors(circuits, simplify=False)

    assert out["ECM Vector"].iloc[0] == [0, 2, 1, 5, 2, -3, -2]

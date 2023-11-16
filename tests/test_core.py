import re

import numpy as np
import pandas as pd

import autoeis as ae


def test_generate_mathematical_expression():
    circuit = "R0-C1-[P2,R3]-[P4-[P5,C6],[L7,R8]]"

    resistors = re.findall(r"R\d+", circuit)
    capacitors = re.findall(r"C\d+", circuit)
    inductors = re.findall(r"L\d+", circuit)
    cpes = re.findall(r"P\d+", circuit)

    num_params = len(resistors) + len(capacitors) + len(inductors) + 2*len(cpes)
    freq = np.array([1, 10, 100])
    p = np.random.rand(num_params)

    circuit_df = pd.DataFrame([circuit], columns=["circuitstring"])
    func_str = ae.core.generate_mathematical_expression(circuit_df)
    func_str = func_str["Mathematical expressions"][0]
    def func(X, F): return eval(func_str)
    Z_py = func(p, freq)

    ec = ae.julia_helpers.import_backend()
    Z_jl = np.array([ec.get_target_impedance(circuit, p, f) for f in freq])

    np.testing.assert_allclose(Z_py, Z_jl)

import autoeis as ae
import numpy as np

# Real numbers
x1 = np.random.rand(10)
x2 = np.random.rand(10)
# Complex numbers
y1 = x1 + np.zeros(10) * 1j
y2 = x2 + np.zeros(10) * 1j

# Simulated EIS data
circuit_string = "R1-[P2,R3]"
p0_dict = {"R1": 250, "P2w": 1e-3, "P2n": 0.5, "R3": 10.0}
p0_vals = list(p0_dict.values())
circuit_fn_gt = ae.utils.generate_circuit_fn_impedance_backend(circuit_string)
freq = np.logspace(-3, 3, 1000)
Z = circuit_fn_gt(freq, p0_vals)


def test_fit_circuit_parameters_without_x0():
    p_dict = ae.utils.fit_circuit_parameters(circuit_string, freq, Z, iters=10)
    p_fit = list(p_dict.values())
    assert np.allclose(p_fit, p0_vals, rtol=0.01)


def test_fit_circuit_parameters_with_x0():
    # Add some noise to the initial guess to test robustness
    p0 = p0_vals + np.random.rand(len(p0_vals)) * p0_vals * 0.5
    p_dict = ae.utils.fit_circuit_parameters(circuit_string, freq, Z, p0)
    p_fit = list(p_dict.values())
    assert np.allclose(p_fit, p0_vals, rtol=0.01)


def test_generate_circuit_fn():
    circuit = "R0-C1-[P2,R3]-[P4-[P5,C6],[L7,R8]]"
    num_params = ae.parser.count_parameters(circuit)
    freq = np.array([1, 10, 100])
    p = np.random.rand(num_params)
    circuit_fn = ae.utils.generate_circuit_fn(circuit)
    Z_py = circuit_fn(freq, p)
    Main = ae.julia_helpers.init_julia()
    ec = ae.julia_helpers.import_backend(Main)
    Z_jl = np.array([ec.get_target_impedance(circuit, p, f) for f in freq])
    np.testing.assert_allclose(Z_py, Z_jl)


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
        assert ae.utils.circuit_complexity(cstr) == cc


def test_are_circuits_equivalent():
    testset = [
        ["R1-C2-L3-R4", "C1-R2-R5-L8", True],
        ["R1-[C2,R3-L4]-P5", "[L1-R2,C5]-P10-R20", True],
        ["R1-[[[C2,P5],R3],P2]", "[P1,[R1,[P5,C5]]]-R4", True],
        ["R1-C2-L3", "R1-C2-P3", False],
        ["[R1,C2]-[P1,C3]", "[R1,C3]-[P1,R2]", False],
    ]
    for row in testset:
        c1, c2, eq = row
        assert ae.utils.are_circuits_equivalent(c1, c2) == eq


def test_eval_posterior_predictive():
    # Load test dataset
    freq, Z = ae.io.load_test_dataset()
    circuits = ae.io.load_test_circuits(filtered=True)
    circuit = circuits.iloc[0].circuitstring
    p0 = circuits.iloc[0].Parameters

    # Perform Bayesian inference on a single ECM
    kwargs_mcmc = {"num_warmup": 2500, "num_samples": 1000, "progress_bar": False}
    result = ae.core.perform_bayesian_inference(circuit, freq, Z, p0, **kwargs_mcmc)

    # Evaluate the posterior predictive distribution with priors
    priors = ae.utils.initialize_priors(p0, variables=p0.keys())
    Z_pred = ae.utils.eval_posterior_predictive(result.mcmc, circuit, freq, priors)
    assert Z_pred.shape == (1000, len(freq))

    # Evaluate the posterior predictive distribution without priors
    Z_pred = ae.utils.eval_posterior_predictive(result.mcmc, circuit, freq)
    assert Z_pred.shape == (1000, len(freq))

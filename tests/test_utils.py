import numpy as np
import pytest

import autoeis as ae

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
circuit_fn_gt = ae.utils.generate_circuit_fn(circuit_string, backend="impedance")
freq = np.logspace(-3, 3, 1000)
Z = circuit_fn_gt(freq, p0_vals)


def test_preprocess_impedance_data():
    freq, Z = ae.io.load_test_dataset()
    # Test various tolerances for linKK validation
    freq_prep, Z_prep = ae.utils.preprocess_impedance_data(freq, Z, tol_linKK=5e-2)
    assert len(Z_prep) == len(freq_prep)
    assert len(Z_prep) == 60
    freq_prep, Z_prep = ae.utils.preprocess_impedance_data(freq, Z, tol_linKK=5e-3)
    assert len(Z_prep) == len(freq_prep)
    assert len(Z_prep) == 50
    # Test return_aux=True
    _, _, aux = ae.utils.preprocess_impedance_data(freq, Z, return_aux=True)
    assert set(aux.keys()) == {"res", "rmse", "freq"}
    assert set(aux["res"].keys()) == {"real", "imag"}


def test_preprocess_impedance_data_dtype_consistency():
    # This is to ensure AUTODIAL/AutoEIS/#140 is fixed
    freq, Z = ae.io.load_test_dataset()
    freq = freq.astype("float32")  # Used to break workflow
    # Test various tolerances for linKK validation
    ae.utils.preprocess_impedance_data(freq, Z, tol_linKK=5e-2)


def test_preprocess_impedance_data_no_high_freq():
    # This is to ensure AUTODIAL/AutoEIS/#122 is fixed
    freq, Z = ae.io.load_test_dataset()
    # Pass high_freq_threshold=1e10 to simulate missing high frequency data
    ae.utils.preprocess_impedance_data(freq, Z, high_freq_threshold=1e10)


def test_fit_circuit_parameters_without_x0():
    p_dict = ae.utils.fit_circuit_parameters(circuit_string, freq, Z, max_iters=100)
    p_fit = list(p_dict.values())
    assert np.allclose(p_fit, p0_vals, rtol=0.01)


def test_fit_circuit_parameters_with_x0():
    # Add some noise to the initial guess to test robustness
    p0 = p0_vals + np.random.rand(len(p0_vals)) * p0_vals * 0.5
    p_dict = ae.utils.fit_circuit_parameters(circuit_string, freq, Z, p0)
    p_fit = list(p_dict.values())
    assert np.allclose(p_fit, p0_vals, rtol=0.01)


@pytest.mark.skip(reason="We're catching Exceptions in the function")
def test_fit_circuit_parameters_with_bounds():
    # Pass incorrect bounds to ensure bounds are being used (Exception should be raised)
    bounds = [(0, 0, 0, 0), (1e-6, 1e-6, 1e-6, 1e-6)]
    with pytest.raises(Exception):
        ae.utils.fit_circuit_parameters(circuit_string, freq, Z, max_iters=25, bounds=bounds)


def test_generate_circuit_fn():
    circuit = "R0-C1-[P2,R3]-[P4-[P5,C6],[L7,R8]]"
    circuit_fn = ae.utils.generate_circuit_fn(circuit)
    num_params = ae.parser.count_parameters(circuit)
    p = np.random.rand(num_params)
    freq = np.array([1, 10, 100])
    Z_py = circuit_fn(freq, p)
    Main = ae.julia_helpers.init_julia()
    ec = ae.julia_helpers.import_backend(Main)
    Z_jl = np.array([ec.get_target_impedance(circuit, p, f) for f in freq])
    np.testing.assert_allclose(Z_py, Z_jl)


@pytest.mark.parametrize("backend", ["numpy", "jax", "numexpr", "impedance"])
def test_generate_circuit_fn_frequency_independent_ecm(backend):
    """
    Frequency-independent ECMs used to return a scalar, rather than an
    array of size len(freq). This unit test checks for this behavior.
    """
    circuit = "R1-R2"
    circuit_fn = ae.utils.generate_circuit_fn(circuit, backend=backend)
    num_params = ae.parser.count_parameters(circuit)
    p = np.random.rand(num_params)
    # Input: frequency array, output: array of size len(freq)
    freq = np.array([1, 10, 100])
    assert len(circuit_fn(freq, p)) == len(freq)
    # Input: scalar frequency, output: still array, but of size 1!
    freq = 10
    assert len(circuit_fn(freq, p)) == 1


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
    result = ae.core.perform_bayesian_inference(circuit, freq, Z, p0, **kwargs_mcmc)[0]

    # Evaluate the posterior predictive distribution with priors
    priors = ae.utils.initialize_priors(p0)
    Z_pred = ae.utils.eval_posterior_predictive(result.samples, circuit, freq, priors)
    assert Z_pred.shape == (1000, len(freq))

    # Evaluate the posterior predictive distribution without priors
    Z_pred = ae.utils.eval_posterior_predictive(result.samples, circuit, freq)
    assert Z_pred.shape == (1000, len(freq))


def test_find_duplicate_circuits():
    circuits = [
        "R0",
        "R1-C2-L3",
        "[R1,[R2,P3]]",
        "R4-C5-[R1,P2]-R0",
        "L2-R5-C6",
        "R1-[P2,R3]-C4",
        "R4-[R1,R2]-R3",
    ]
    # No simplification
    duplicates = ae.utils.find_duplicate_circuits(circuits)
    expected = [[0], [1, 4], [2], [3], [5], [6]]
    match = [np.allclose(a, b) for a, b in zip(duplicates, expected)]
    assert all(match), f"Expected {expected}, got {duplicates}"
    # With simplification
    duplicates = ae.utils.find_duplicate_circuits(circuits, simplify=True)
    expected = [[0, 6], [1, 4], [2], [3, 5]]
    match = [np.allclose(a, b) for a, b in zip(duplicates, expected)]
    assert all(match), f"Expected {expected}, got {duplicates}"


def test_sample_circuit_parameters_basic():
    circuit = "R1-[P2,C3]"

    # Test single sample (backward compatibility)
    p_single = ae.utils.sample_circuit_parameters(circuit, seed=42)
    assert isinstance(p_single, np.ndarray)
    assert p_single.shape == (4,)
    assert len(p_single) == ae.parser.count_parameters(circuit)

    # Test multiple samples
    p_multi = ae.utils.sample_circuit_parameters(circuit, num_samples=10, seed=42)
    assert p_multi.shape == (10, 4)

    # Test reproducibility
    p1 = ae.utils.sample_circuit_parameters(circuit, seed=42)
    p2 = ae.utils.sample_circuit_parameters(circuit, seed=42)
    np.testing.assert_array_equal(p1, p2)

    # Test different circuits
    circuits = ["R1", "R1-C2", "[R1,C2]-P3"]
    for test_circuit in circuits:
        expected_params = ae.parser.count_parameters(test_circuit)
        p = ae.utils.sample_circuit_parameters(test_circuit, seed=42)
        assert p.shape == (expected_params,)


def test_sample_circuit_parameters_bounds():
    circuit = "R1-P2-C3"

    # Test default bounds enforcement
    p = ae.utils.sample_circuit_parameters(circuit, num_samples=50, seed=42)

    # Pn values should be [0,1]
    pn_values = p[:, 2]  # P2n
    assert np.all(pn_values >= 0.0) and np.all(pn_values <= 1.0)

    # Other parameters should be positive
    assert np.all(p[:, [0, 1, 3]] > 0)  # R, Pw, C

    # Test custom bounds
    bounds = {"R": (1e-2, 1e2), "C": (1e-9, 1e-6)}
    p_custom = ae.utils.sample_circuit_parameters(
        "R1-C2", bounds=bounds, num_samples=20, seed=42
    )

    assert np.all(p_custom[:, 0] >= bounds["R"][0])  # R min
    assert np.all(p_custom[:, 0] <= bounds["R"][1])  # R max
    assert np.all(p_custom[:, 1] >= bounds["C"][0])  # C min
    assert np.all(p_custom[:, 1] <= bounds["C"][1])  # C max


def test_sample_circuit_parameters_sampling_modes():
    circuit = "P1"  # CPE with Pw and Pn

    # Test log vs linear produces different results
    p_log = ae.utils.sample_circuit_parameters(circuit, num_samples=50, log=True, seed=42)
    p_linear = ae.utils.sample_circuit_parameters(circuit, num_samples=50, log=False, seed=42)

    # Pw should be different between log/linear
    assert not np.array_equal(p_log[:, 0], p_linear[:, 0])

    # Pn should be uniformly distributed in both cases
    pn_log = p_log[:, 1]
    pn_linear = p_linear[:, 1]

    assert np.all(pn_log >= 0.0) and np.all(pn_log <= 1.0)
    assert np.all(pn_linear >= 0.0) and np.all(pn_linear <= 1.0)


@pytest.mark.parametrize("backend", ["numpy", "jax", "numexpr", "impedance"])
@pytest.mark.parametrize("concat", [False, True])
def test_generate_circuit_fn_backends(backend, concat):
    """Test basic functionality across all backends."""
    circuits = ["R1-C2", "R1-[P2,C3]"]
    freq = np.logspace(-1, 3, 5)  # Smaller array for faster tests

    for circuit in circuits:
        fn = ae.utils.generate_circuit_fn(circuit, backend=backend, concat=concat)
        p = ae.utils.sample_circuit_parameters(circuit, seed=42)
        Z = fn(freq, p)

        # Check output shape and type
        expected_shape = (2 * len(freq),) if concat else (len(freq),)
        expected_dtype = np.float64 if concat else np.complex128

        msg = f"Backend {backend}, concat={concat}: Shape mismatch."
        assert Z.shape == expected_shape, msg
        msg = f"Backend {backend}, concat={concat}: Dtype mismatch."
        assert Z.dtype == expected_dtype, msg

        # Test scalar frequency
        Z_scalar = fn(10.0, p)
        if concat:
            assert Z_scalar.shape == (2,)
        else:
            assert Z_scalar.shape == (1,)


@pytest.mark.parametrize("backend", ["numpy", "jax", "numexpr", "impedance"])
def test_generate_circuit_fn_consistency(backend):
    """Test that all backends produce consistent results."""
    circuit = "R1-[P2,C3]"
    freq = np.logspace(-1, 3, 10)
    p = ae.utils.sample_circuit_parameters(circuit, seed=42)

    # Get reference result from numpy backend
    fn_numpy = ae.utils.generate_circuit_fn(circuit, backend="numpy", concat=False)
    Z_ref = fn_numpy(freq, p)

    # Test backend against reference
    fn_backend = ae.utils.generate_circuit_fn(circuit, backend=backend, concat=False)
    Z_test = fn_backend(freq, p)

    # Allow slightly looser tolerance for different backends
    tolerance = 1e-10 if backend in ["numpy", "jax"] else 1e-8
    np.testing.assert_allclose(
        Z_test,
        Z_ref,
        rtol=tolerance,
        err_msg=f"Backend {backend} inconsistent with numpy reference",
    )


@pytest.mark.parametrize("backend", ["numpy", "jax", "numexpr"])
@pytest.mark.parametrize("concat", [False, True])
def test_generate_circuit_fn_batch_evaluation(backend, concat):
    """Test batch evaluation for backends that support it."""
    circuit = "R1-[P2,C3]"
    freq = np.logspace(-1, 3, 10)
    n_samples = 5
    p_batch = ae.utils.sample_circuit_parameters(circuit, num_samples=n_samples, seed=42)

    # Test batch capabilities
    fn_batch = ae.utils.generate_circuit_fn(circuit, backend=backend, concat=concat)
    Z_batch = fn_batch(freq, p_batch)

    # Compare with individual evaluations
    fn_single = ae.utils.generate_circuit_fn(circuit, backend="numpy", concat=concat)
    Z_individual = np.array([fn_single(freq, p) for p in p_batch])

    # Check shapes and results
    assert Z_batch.shape == Z_individual.shape
    np.testing.assert_allclose(Z_batch, Z_individual, rtol=1e-10)


def test_generate_circuit_fn_batch_impedance_error():
    """Test that impedance backend raises error for batch evaluation."""
    circuit = "R1-C2"
    freq = np.logspace(-1, 3, 5)
    n_samples = 3
    p_batch = ae.utils.sample_circuit_parameters(circuit, num_samples=n_samples, seed=42)

    fn_impedance = ae.utils.generate_circuit_fn(circuit, backend="impedance")

    with pytest.raises(RuntimeError):
        fn_impedance(freq, p_batch)


def test_generate_circuit_fn_edge_cases():
    """Test edge cases and error handling."""

    # Test frequency-independent circuit
    freq_indep_circuit = "R1-R2"
    freq = np.array([1, 10, 100])
    p = ae.utils.sample_circuit_parameters(freq_indep_circuit, seed=42)

    for backend in ["numpy", "jax", "numexpr", "impedance"]:
        fn = ae.utils.generate_circuit_fn(freq_indep_circuit, backend=backend)
        Z = fn(freq, p)
        msg = f"Backend {backend} failed frequency-independent circuit test"
        assert Z.shape == (len(freq),), msg

        # Test scalar frequency
        Z_scalar = fn(10.0, p)
        assert Z_scalar.shape == (1,), f"Backend {backend} failed scalar frequency test"

    # Test invalid backend
    with pytest.raises(ValueError, match="Invalid backend"):
        ae.utils.generate_circuit_fn("R1", backend="invalid")

    # Test single component circuits
    for component in ["R1", "C1", "L1"]:
        p_single = ae.utils.sample_circuit_parameters(component, seed=42)
        fn = ae.utils.generate_circuit_fn(component, backend="numpy")
        Z = fn(freq, p_single)
        assert len(Z) == len(freq)

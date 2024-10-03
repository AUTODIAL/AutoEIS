"""
Collection of functions core to AutoEIS functionality.

.. currentmodule:: autoeis.core

.. autosummary::
   :toctree: generated/

    perform_full_analysis
    generate_equivalent_circuits
    filter_implausible_circuits
    perform_bayesian_inference

"""

import logging
import os
import time
import warnings
from collections.abc import Iterable, Mapping
from copy import deepcopy

import arviz as az
import jax
import jax.numpy as jnp  # noqa: F401
import numpy as np
import numpyro
import pandas as pd
import psutil
from deprecated import deprecated
from jax import config
from mpire import WorkerPool
from numpyro.distributions import Distribution
from numpyro.infer import MCMC, NUTS
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

from autoeis import io, julia_helpers, metrics, models, parser, utils

from .utils import InferenceResult

# FP32 gradients not sufficient for ECM fitting
numpyro.enable_x64()
# GPU is actually slower for typical EIS dataset sizes
numpyro.set_platform("cpu")
# Get rid of the JAX warning about GPU might be available
config.update("jax_platforms", "cpu")

warnings.filterwarnings("ignore", category=Warning, module="arviz.*")
log = logging.getLogger(__name__)

# Initialize Julia runtime
os.environ["PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION"] = "no"
julia_helpers.ensure_julia_deps_ready(quiet=True)
jl = julia_helpers.init_julia(quiet=True)
ec = julia_helpers.import_backend(jl)

__all__ = [
    "perform_full_analysis",
    "generate_equivalent_circuits",
    "filter_implausible_circuits",
    "perform_bayesian_inference",
    "compute_fitness_metrics",
]


# TODO: Breaks when data is noisy -> use curve_fit to extrapolate R0
def compute_ohmic_resistance(freq: np.ndarray[float], Z: np.ndarray[complex]) -> float:
    """Extracts the ohmic resistance from impedance data.

    Parameters
    ----------
    Z : np.ndarray[complex]
        Impedance measurements.
    freq : np.ndarray[float]
        Frequencies corresponding to impedance measurements.

    Returns
    -------
    ohmic_resistance: float
        The ohmic resistance of impedance data.

    Raises
    ------
    ValueError
        If the ohmic resistance cannot be reliably extracted.
    """
    # Sort impedance data by descending frequency
    mask = np.argsort(freq)[::-1]
    Z = Z[mask]
    freq = freq[mask]

    # Curve fit to a saturating function: (ax + b) / (x + c)
    # NOTE: x -> inf, f(x) -> a === 1/R0 => set a > 0 as lower bound
    # NOTE: To put more weight on high frequency data, use 1/Z.real as y
    x = freq[:]
    y = 1 / Z.real[:]

    def func(x, a, b, c):
        return (a * x + b) / (x + c)

    fallback = False
    bounds = ([0, -np.inf, -np.inf], np.inf)

    try:
        (a, b, c), pcov = curve_fit(func, x, y, bounds=bounds)
        R = 1 / a
    except RuntimeError:
        fallback = True
    if R < 0 or R < 0.2 * Z.real[0]:
        fallback = True
    if fallback:
        R = Z.real[0]
        log.warning("Failed to fit ohmic resistance, returning Re(Z) @ max(freq).")
    return R


def generate_equivalent_circuits(
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    iters: int = 100,
    complexity: int = 12,
    tol: float = 1e-3,
    parallel: bool = True,
    generations: int = 30,
    population_size: int = 100,
    terminals: str = "RLP",
    seed: int = None,
) -> pd.DataFrame:
    """Generates candidate circuits that fit the impedance data using
    evolutionary algorithms.

    Parameters
    ----------
    freq : np.ndarray[float]
        Frequencies corresponding to impedance measurements.
    Z : np.ndarray[complex]
        Impedance measurements as a complex array.
    iters : int, optional
        Number of ECM generation iterations (default is 100).
    complexity : int, optional
        Complexity of the ECM search space (default is 12).
    tol : float, optional
        Convergence threshold for the ECM search (default is 1e-3).
    parallel : bool, optional
        If True, the ECM search will be performed in parallel (default is True).
    generations : int, optional
        Number of generations for the ECM search (default is 30).
    population_size : int, optional
        Number of ECMs to generate per generation (default is 100).
    terminals : str, optional
        Circuit components to consider (default is "RLP"). R: resistor,
        L: inductor, C: capacitor, P: constant-phase element.
    seed : int, optional
        Random seed for reproducibility (default is None).

    Returns
    -------
    pd.DataFrame or None
        Dataframe containing circuits.
    """
    log.info("Generating equivalent circuits via evolutionary algorithms.")

    # Set the seed for reproducibility (if not set, use current time in nanoseconds)
    seed = seed or time.time_ns() % 2**32

    ec_kwargs = {
        "head": complexity,
        "terminals": terminals,
        "convergence_threshold": tol,
        "generations": generations,
        "population_size": population_size,
    }

    ecm_generator = _generate_ecm_parallel_julia if parallel else _generate_ecm_serial
    circuits = ecm_generator(freq, Z, iters, ec_kwargs, seed)

    # Convert output to DataFrame with columns ("circuitstring", "Parameters")
    circuits = io.parse_ec_output(circuits)

    if not len(circuits):
        log.warning("No plausible circuits found. Increase 'tol', 'iters', or both!")

    return circuits


def _generate_ecm_serial(
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    iters: int,
    ec_kwargs: dict,
    seed: int,
) -> list[str]:
    """Generates candidate circuits that fit the impedance data, in serial."""
    # Set random seed for reproducibility
    jl.seval(f"import Random; Random.seed!({seed})")

    circuits = []
    for _ in tqdm(range(iters), desc="Generating Candidate ECMs", leave=False):
        utils.flush_streams()
        try:
            circuit = ec.circuit_evolution(Z, freq, **ec_kwargs, quiet=True)
        except Exception as e:
            log.error(f"Error generating circuit: {e}")
            continue
        circuits.append(circuit)
    else:
        utils.flush_streams()

    circuits = [str(c) for c in circuits if c is not None]
    return circuits


def _generate_ecm_parallel_julia(
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    iters: int,
    ec_kwargs: dict,
    seed: int,
):
    """Generates candidate circuits that fit the impedance data, in parallel
    via Julia multiprocessing.
    """
    # Set random seed for reproducibility (Python and Julia)
    # FIXME: This doesn't work when multiprocessing, use @everywhere instead
    jl.seval(f"import Random; Random.seed!({seed})")

    # HACK: To get a progress bar, chunk the iterations -> call Julia repeatedly
    nprocs = psutil.cpu_count(logical=False)
    # Double the number of workers to buffer for those with slow convergence
    # (don't do this for small iters, otherwise progress bar becomes pointless)
    nprocs = 2 * nprocs if (iters // nprocs) > 10 else nprocs
    # NOTE: e.g., iters = 11, nprocs = 4 -> iters_chunked = [4, 4, 3]
    iters_chunked = [nprocs] * (iters // nprocs)
    if iters % nprocs:
        iters_chunked.append(iters % nprocs)

    # Perform parallelized GEP in chunks
    circuits = []

    with tqdm(total=iters, desc="Generating Candidate ECMs", miniters=1, leave=False) as pbar:
        utils.flush_streams()
        for iters_ in iters_chunked:
            try:
                circuits_ = ec.circuit_evolution_batch(
                    Z, freq, **ec_kwargs, iters=iters_, quiet=True
                )
            except Exception as e:
                log.error(f"Error generating circuits: {e}")
                circuits_ = []
            circuits += circuits_
            pbar.update(iters_)
            utils.flush_streams()

    circuits = [str(c) for c in circuits if c is not None]
    return circuits


@deprecated(reason="This function is deprecated, use _generate_ecm_parallel_julia instead")
def _generate_ecm_parallel_mpire(
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    iters: int,
    ec_kwargs: dict,
    seed: int,
):
    """Generates candidate circuits that fit the impedance data, in parallel
    via Python multiprocessing.
    """

    def circuit_evolution(seed: int):
        """Closure to generate a single circuit to be used with multiprocessing."""
        jl = julia_helpers.init_julia()
        ec = julia_helpers.import_backend(jl)
        # Set random seed for reproducibility
        jl.seval(f"import Random; Random.seed!({seed})")
        try:
            circuit = ec.circuit_evolution(Z, freq, **ec_kwargs)
        except Exception as e:
            log.error(f"Error generating circuit: {e}")
            return None
        return circuit

    nproc = os.cpu_count()
    mpire_kwargs = {
        "iterable_len": iters,
        "progress_bar": True,
        "progress_bar_style": "notebook",
        "progress_bar_options": {"desc": "Circuit Evolution"},
    }

    # Julia cannot be initialized in the main process -> guard against this
    runtime_error = False

    # Set a different seed for each process
    seed = [seed + i for i in range(iters)]

    with WorkerPool(n_jobs=nproc) as pool:
        try:
            circuits = pool.map(circuit_evolution, seed, **mpire_kwargs)
        except RuntimeError:
            runtime_error = True

    if runtime_error:
        raise RuntimeError("Julia must not be manually initialized, restart the kernel.")

    circuits = [str(c) for c in circuits if c is not None]
    return circuits


def split_components(circuits: pd.DataFrame) -> pd.DataFrame:
    """Augments the circuits dataframe with additional columns for each component."""
    # Initialize lists to populate the component columns later
    components = {"R": [], "C": [], "L": [], "P": []}
    labels = {"R": "Resistors", "C": "Capacitors", "L": "Inductors", "P": "CPEs"}

    for row in circuits.itertuples():
        circuit = row.circuitstring
        # Find components of each kind
        pgroups = parser.group_parameters_by_component(circuit)
        for ctype in components.keys():
            components[ctype].append(pgroups.get(ctype, []))

    # Add component columns to the dataframe
    for ctype, params in components.items():
        column_label = labels[ctype]
        circuits[column_label] = params

    return circuits


def capacitance_filter(circuits: pd.DataFrame) -> pd.DataFrame:
    """Excludes ideal capacitors from the circuits dataframe."""
    circuits = circuits.copy(deep=True)

    for row in circuits.itertuples():
        variables = row.Parameters.keys()
        contains_capacitor = any("C" in var for var in variables)
        if contains_capacitor:
            circuits.drop(row.Index, inplace=True)
    circuits.reset_index(drop=True, inplace=True)

    return circuits


def ohmic_resistance_filter(circuits: pd.DataFrame) -> pd.DataFrame:
    """Excludes circuits without an ohmic resistance from the circuits dataframe."""
    circuits = circuits.copy(deep=True)

    for row in circuits.itertuples():
        circuit = row.circuitstring
        resistors = parser.find_ohmic_resistors(circuit)
        if not resistors:
            circuits.drop(row.Index, inplace=True)

    circuits.reset_index(drop=True, inplace=True)
    return circuits


def series_filter(circuits: pd.DataFrame) -> pd.DataFrame:
    """Excludes circuits with series-only components from the circuits dataframe."""
    circuits = circuits.copy(deep=True)

    for row in circuits.itertuples():
        circuit = row.circuitstring
        contains_parallel_route = "[" in circuit
        if not contains_parallel_route:
            circuits.drop(row.Index, inplace=True)

    circuits.reset_index(drop=True, inplace=True)
    return circuits


def merge_identical_circuits(circuits: "pd.DataFrame") -> "pd.DataFrame":
    """Merges identical circuits from the circuits dataframe."""
    circuits = circuits.copy(deep=True)

    for i, row_i in circuits.iterrows():
        circuit_i = row_i.circuitstring
        for j, row_j in circuits.loc[i + 1 :].iterrows():
            if utils.are_circuits_equivalent(circuit_i, row_j.circuitstring):
                circuits.drop(j, inplace=True)

    circuits.reset_index(drop=True, inplace=True)
    return circuits


def compute_fitness_metrics(
    circuits: pd.DataFrame, freq: np.ndarray[float], Z: np.ndarray[complex]
) -> pd.DataFrame:
    """Computes various fitness metrics and returns an augmented dataframe.

    Parameters
    ----------
    circuits : pd.DataFrame
        Circuits dataframe with inference results
    freq : np.ndarray[float]
        Frequencies corresponding to the impedance data
    Z : np.ndarray[complex]
        Complex impedance data

    Returns
    -------
    circuits : pd.DataFrame
        Circuits dataframe with fitness metrics
    """
    circuits = circuits.copy(deep=True)

    results = circuits["InferenceResult"]
    mcmcs = [result.mcmc for result in results]
    circuits["converged"] = [result.converged for result in results]
    circuits["divergences"] = [result.num_divergences for result in results]

    # Compute the posterior predictive and add to the dataframe
    Z_pred = [utils.eval_posterior_predictive(r.samples, r.circuit, freq) for r in results]
    circuits["Z_pred"] = Z_pred

    # FIXME: This function doesn't work with Bode as objective function
    # TODO: Extract log-likelihood of mag/phase, and turn it into that for real/imag

    # Compute WAIC and add to the dataframe
    if "sigma.real" in mcmcs[0].get_samples().keys():
        waic_re = [az.waic(x, var_name="obs.real", scale="deviance") for x in mcmcs]
        waic_im = [az.waic(x, var_name="obs.imag", scale="deviance") for x in mcmcs]
        circuits["WAIC (real)"] = [x["elpd_waic"] + 2 * x["p_waic"] for x in waic_re]
        circuits["WAIC (imag)"] = [x["elpd_waic"] + 2 * x["p_waic"] for x in waic_im]
    else:
        waic_mag = [az.waic(x, var_name="obs.mag", scale="deviance") for x in mcmcs]
        waic_phase = [az.waic(x, var_name="obs.phase", scale="deviance") for x in mcmcs]
        circuits["WAIC (mag)"] = [x["elpd_waic"] + 2 * x["p_waic"] for x in waic_mag]
        circuits["WAIC (phase)"] = [x["elpd_waic"] + 2 * x["p_waic"] for x in waic_phase]

    # Compute R^2 and add to the dataframe
    fn = lambda row: metrics.r2_score(Z.real, row.Z_pred.real, axis=1)
    circuits["R^2 (real)"] = circuits.apply(fn, axis=1)
    fn = lambda row: metrics.r2_score(Z.imag, row.Z_pred.imag, axis=1)
    circuits["R^2 (imag)"] = circuits.apply(fn, axis=1)
    # Since R^2 is a vector (num_samples), also compute its mean for convenience
    circuits["R^2 (ravg)"] = circuits["R^2 (real)"].apply(np.mean)
    circuits["R^2 (iavg)"] = circuits["R^2 (imag)"].apply(np.mean)

    # Compute MAPE and add to the dataframe
    fn = lambda row: metrics.mape_score(Z.real, row.Z_pred.real, axis=1)
    circuits["MAPE (real)"] = circuits.apply(fn, axis=1)
    fn = lambda row: metrics.mape_score(Z.imag, row.Z_pred.imag, axis=1)
    circuits["MAPE (imag)"] = circuits.apply(fn, axis=1)
    # Since MAPE is a vector (num_samples), also compute its mean for convenience
    circuits["MAPE (ravg)"] = circuits["MAPE (real)"].apply(np.mean)
    circuits["MAPE (iavg)"] = circuits["MAPE (imag)"].apply(np.mean)

    # Add number of parameters to the dataframe
    circuits["n_params"] = circuits.apply(lambda row: len(row.Parameters), axis=1)

    return circuits


def _validate_circuit(circuits):
    """Validates the circuits input."""
    # Dataframe
    if isinstance(circuits, pd.DataFrame):
        utils.validate_circuits_dataframe(circuits)
    # List
    elif isinstance(circuits, list):
        for c in circuits:
            assert isinstance(c, str), f"Invalid circuit: {c}"
    # String
    elif isinstance(circuits, str):
        pass
    else:
        raise ValueError("`circuits` must be a DataFrame, list, or string.")
    return circuits


def _validate_impedance_data(freq, Z):
    """Validates the impedance data input."""
    # Shared validations (single/multiple datasets)
    assert type(freq) is type(Z), "freq and Z must be of the same type."
    assert utils.is_iterable(freq), "freq must be an ndarray or list/tuple[ndarray]."
    assert utils.is_iterable(Z), "Z must be an ndarray or list/tuple[ndarray]."
    assert len(freq) == len(Z), "freq and Z must have the same length."
    # Standardize freq/Z to be an np.ndarray (freq/Z[i] for multiple datasets)
    if utils.is_nested_iterable(freq):
        for i, (f, z) in enumerate(zip(freq, Z)):
            assert isinstance(f, np.ndarray), f"freq must be a numpy array (dataset #{i})."
            assert isinstance(z, np.ndarray), f"Z must be a numpy array (dataset #{i})."
            assert f.size == z.size, f"freq and Z must have the same length (dataset #{i})."
    else:
        assert isinstance(freq, np.ndarray), "freq must be a numpy array."
        assert isinstance(Z, np.ndarray), "Z must be a numpy array."
    return freq, Z


def _validate_p0(p0, broadcast_to: int = 1):
    """Validates the initial guess for ECM parameters, and broadcasts to a list."""
    # Validate input data type
    if not (isinstance(p0, (dict, list)) or utils.is_ndfarray_like(p0) or p0 is None):
        raise ValueError("'p0' must be an ndarray/dict or list[ndarray/dict].")

    # If single p0 is provided, broadcast
    if p0 is None or utils.is_ndfarray_like(p0) or isinstance(p0, dict):
        p0 = [deepcopy(p0) for _ in range(broadcast_to)]

    # If a list of p0 is provided, validate
    assert isinstance(p0, list), "Something went wrong. Please report!"
    for p0_ in p0:
        msg = f"Invalid p0: {p0_}"
        assert utils.is_ndfarray_like(p0_) or isinstance(p0_, dict), msg

    return p0


def _validate_priors(priors, broadcast_to: int = 1):
    """Validates the priors for the ECM parameters, and broadcasts to a list."""
    # Validate input data type
    if not isinstance(priors, (Mapping, list, type(None))):
        raise ValueError("'priors' must be a dict or list[dict].")

    # If single priors is provided, broadcast
    if priors is None or isinstance(priors, Mapping):
        priors = [deepcopy(priors) for _ in range(broadcast_to)]

    # Ensure all priors are valid
    assert isinstance(priors, list), "Something went wrong. Please report!"
    for priors_ in priors:
        msg = "'priors' must be a dict of NumPyro Distribution objects."
        assert isinstance(priors_, Mapping), msg
        assert all(isinstance(v, Distribution) for v in priors_.values()), msg

    return priors


def _validate_seed(seed, num_splits=1) -> list[jax.Array] | jax.Array:
    """Validates the random seed to be used in ``MCMC.run()``."""
    # If seed is not set, generate a new seed using time
    if seed is None:
        seed = time.time_ns() % 2**32
    # If a jax.Array is already provided, use it otherwise generate one
    if isinstance(seed, jax.Array):
        assert seed.size == 2, "'seed' must be a 2-element jax.Array."
        subkey = seed
    elif isinstance(seed, int):
        key = jax.random.PRNGKey(seed)
        key, *subkey = jax.random.split(key, num_splits + 1)
        subkey = subkey[0] if num_splits == 1 else subkey
    else:
        raise ValueError("'seed' must be an int or a jax.Array.")
    return subkey


def _refine_p0(p0, circuit, datasets, progress_bar):
    """"""
    if p0 is None:
        p0 = [None] * len(datasets)
    if len(p0) != len(circuit) != len(datasets):
        raise ValueError("Length of 'p0', 'circuit', and 'datasets' must be the same.")

    max_iters = 10
    p0 = utils.distribute_task(
        utils.fit_circuit_parameters,
        circuit,
        [dataset.freq for dataset in datasets],
        [dataset.Z for dataset in datasets],
        p0,
        max_iters,
        static=(4),  # static args = (circuit, freq, Z, max_iters)
        progress_bar=progress_bar,
        desc="Refining Initial Guess",
    )

    # Handle failed refinements
    for i, elem in enumerate(p0):
        if isinstance(elem, Exception):
            raise RuntimeError(f"Failed to refine p0 for circuit {circuit[i]}: {elem}")

    return p0


def perform_bayesian_inference(
    circuit: str | Iterable[str] | pd.DataFrame,
    freq: np.ndarray[float] | Iterable[float],
    Z: np.ndarray[complex] | Iterable[complex],
    p0: Iterable[float]
    | Mapping[str, float]
    | Iterable[Iterable[float]]
    | Iterable[Mapping[str, float]] = None,
    priors: Mapping[str, Distribution] = None,
    num_warmup: int = 2500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int | jax.Array = None,
    method: str = "bode",
    progress_bar: bool = True,
    refine_p0: bool = True,
    parallel: bool = True,
) -> InferenceResult | list[InferenceResult]:
    """Performs Bayesian inference on the circuits based on impedance data.

    Parameters
    ----------
    circuit : str | Iterable[str] | pd.DataFrame
        Dataframe containing circuits, list of circuit strings, or a single
        circuit string.
    freq: np.ndarray[float] | Iterable[float]
        Frequency array corresponding to the impedance data. If a list of
        frequencies is provided, separate inferences will be performed for
        each frequency array.
    Z : np.ndarray[complex] | Iterable[np.ndarray[complex]]
        Impedance data as a complex array. If a list of impedance arrays is
        provided, separate inferences will be performed for each impedance
        array.
    p0 : Iterable[float] \
            | Mapping[str, float] \
            | Iterable[Iterable[float]] \
            | Iterable[Mapping[str, float]], optional
        Initial guess for the circuit parameters (default is None).
    priors : Mapping[str, Distribution], optional
        Priors for the circuit parameters (default is None).
    num_warmup : int, optional
        Number of warmup samples for the MCMC (default is 2500).
    num_samples : int, optional
        Number of samples for the MCMC (default is 1000).
    num_chains : int, optional
        Number of MCMC chains (default is 1).
    seed : int, optional
        Random seed for reproducibility (default is None).
    method : str, optional
        Objective function to minimize (default is "bode"). Choose from "bode",
        "nyquist", "magnitude", or "chi-squared".
    progress_bar : bool, optional
        If True, a progress bar will be displayed (default is True).
    refine_p0 : bool, optional
        If True, the initial guess for the circuit parameters will be refined
        using the circuit fitter (default is True).
    parallel : bool, optional
        If True, the MCMC chains will be run in parallel (default is True).

    Returns
    -------
    InferenceResult | list[InferenceResult]
        InferenceResult object. If multiple circuits are provided (with a
        single freq/Z pair), or if multiple freq/Z pairs are provided (with a
        single circuit), a list of InferenceResult objects will be returned.

    Notes
    -----
    You cannot provide multiple circuits and multiple datasets together, i.e.,
    either pass a single circuit with multiple datasets or multiple circuits
    with a single dataset.
    """
    # Validate inputs data types
    circuit = _validate_circuit(circuit)
    freq, Z = _validate_impedance_data(freq, Z)
    # Convert dataframe to list if necessary
    if isinstance(circuit, pd.DataFrame):
        p0 = circuit["Parameters"].tolist()
        circuit = circuit["circuitstring"].tolist()

    # Standardize circuits and impedance data to always be a list
    circuit = [circuit] if isinstance(circuit, str) else circuit
    freq, Z = ([freq], [Z]) if isinstance(freq, np.ndarray) else (freq, Z)
    datasets = [utils.ImpedanceData(freq_, Z_) for freq_, Z_ in zip(freq, Z)]

    # Determine processing workflow: SCSD, SCMD, MCSD
    num_inferences = max(len(circuit), len(datasets))
    # Raise error if multiple circuits and datasets are provided together
    if len(circuit) > 1 and len(datasets) > 1:
        raise ValueError("Can't handle multiple circuits and multiple datasets together.")
    # Ensure circuits and datasets are of the same length (if not, broadcast)
    circuit = circuit * num_inferences if len(circuit) == 1 else circuit
    datasets = datasets * num_inferences if len(datasets) == 1 else datasets

    # Ensure p0 and priors are valid and ready for parallel inference
    p0 = _validate_p0(p0, broadcast_to=num_inferences)
    if refine_p0 or (p0[0] is None):
        p0 = _refine_p0(p0, circuit, datasets, progress_bar)
    if priors is None:
        priors = [utils.initialize_priors(p0_) for p0_ in p0]
    priors = _validate_priors(priors, broadcast_to=num_inferences)

    # Generate N random seeds (one for each inference)
    seed = _validate_seed(seed, num_splits=num_inferences)
    # Ensure seed is a list since MCMD requires all inputs to be lists
    seed = [seed] if not isinstance(seed, list) else seed

    # Perform Bayesian inference
    if parallel and num_inferences > 1:
        results = _perform_bayesian_inference_MCMD(
            circuit=circuit,
            dataset=datasets,
            priors=priors,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            method=method,
            seed=seed,
            progress_bar=progress_bar,
        )
    else:
        results = []
        for i in tqdm(range(num_inferences), desc="Running Bayesian Inference", leave=False):
            result = _perform_bayesian_inference_SCSD(
                circuit=circuit[i],
                freq=datasets[i].freq,
                Z=datasets[i].Z,
                priors=priors[i],
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                seed=seed[i],
                method=method,
                progress_bar=False,
            )
            results.append(result)

    # Return a single InferenceResult object if n_inferences = 1, else a list
    return results[0] if len(results) == 1 else results


def _perform_bayesian_inference_SCSD(
    circuit: str,
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    priors: Mapping[str, Distribution],
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    method: str,
    seed: int | jax.Array,
    progress_bar: bool = False,
) -> InferenceResult:
    log.info("Performing Bayesian inference on the circuit {circuit}.")

    subkey = _validate_seed(seed)
    circuit_fn = utils.generate_circuit_fn(circuit, jit=True)
    method = method.replace("-", "_")

    nuts_kernel = NUTS(
        model=getattr(models, f"circuit_regression_{method}"),
        init_strategy=numpyro.infer.init_to_median,
    )
    kwargs_mcmc = {
        "num_samples": num_samples,
        "num_warmup": num_warmup,
        "num_chains": num_chains,
        "progress_bar": progress_bar,
    }
    mcmc = MCMC(nuts_kernel, **kwargs_mcmc)
    kwargs_inference = {
        "freq": freq,
        "Z": Z,
        "priors": priors,
        "circuit_fn": circuit_fn,
    }

    try:
        # HACK: Handle when circuit fitter fails to find p0 (to be caught by RuntimeError)
        if priors is None:
            raise RuntimeError(
                f"Inference couldn't be performed for circuit: {circuit}, "
                "because 'priors' are not provided, possibly because "
                "circuit fitter failed to find an initial guess 'p0'."
            )
        mcmc.run(subkey, **kwargs_inference)
        converged = True
    except RuntimeError as e:
        log.error(f"Inference failed for circuit: {circuit}. Error: {e}")
        converged = False

    return InferenceResult(circuit, mcmc, converged=converged, freq=freq, Z=Z)


def _perform_bayesian_inference_MCMD(
    circuit: Iterable[str],
    dataset: Iterable[utils.ImpedanceData],
    priors: Iterable[Mapping[str, Distribution]],
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    method: str,
    seed: jax.Array,
    progress_bar: bool,
) -> list[InferenceResult]:
    """Performs inference in batch mode; multiple-circuits multiple-data (MCMD)."""
    # Sanity check on input lengths
    msg = "'circuit', 'dataset', and 'priors' must have the same length."
    assert len(circuit) == len(dataset) == len(priors), msg

    # Fetch freq, Z pairs from the dataset
    freq, Z = zip(*[(dataset_.freq, dataset_.Z) for dataset_ in dataset])

    # If a single inference is requested, perform SCSD directly to avoid overhead
    if len(circuit) == 1:
        results = _perform_bayesian_inference_SCSD(
            circuit[0],
            freq[0],
            Z[0],
            priors=priors[0],
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            method=method,
            seed=seed[0],
            progress_bar=False,
        )
        results = [results]
    else:
        results = utils.distribute_task(
            _perform_bayesian_inference_SCSD,
            # --- _perform_bayesian_inference_SCSD args ---
            circuit,  # 0
            freq,  # 1
            Z,  # 2
            priors,  # 3
            num_warmup,  # 4
            num_samples,  # 5
            num_chains,  # 6
            method,  # 7
            seed,  # 8
            # --- distribute_task kwargs ---
            static=(4, 5, 6, 7),  # indices of static args
            progress_bar=progress_bar,
            desc="Performing Bayesian Inference",
        )
    return results


def filter_implausible_circuits(circuits: pd.DataFrame) -> pd.DataFrame:
    """Applies heuristic rules to exclude implausible circuits.

    Parameters
    ----------
    circuits : pd.DataFrame
        Dataframe containing circuits.

    Returns
    -------
    circuits : pd.DataFrame
        Dataframe containing the filtered circuits.
    """
    log.info("Filtering the circuits using heuristic rules.")
    circuits = circuits.copy(deep=True)

    if len(circuits) == 0:
        log.warning("Circuits' dataframe is empty!")
        return circuits

    circuits = split_components(circuits)
    circuits = capacitance_filter(circuits)
    circuits = series_filter(circuits)
    circuits = ohmic_resistance_filter(circuits)
    circuits = merge_identical_circuits(circuits)

    # Drop the columns added by split_components
    circuits = circuits.drop(columns=["Resistors", "Capacitors", "Inductors", "CPEs"])

    if len(circuits) == 0:
        log.warning("No plausible circuits left after post-filtering")  # fmt: skip

    return circuits


def perform_full_analysis(
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    iters: int = 100,
    parallel: bool = True,
    tol_linKK: float = 5e-2,
    tol: float = 1e-2,
    num_warmup: int = 2500,
    num_samples: int = 1000,
) -> pd.DataFrame:
    """Performs automated EIS analysis by generating plausible ECMs that
    fit the impedance data, followed by Bayesian inference on components.

    Parameters
    ----------
    freq : np.ndarray[float]
        Frequencies corresponding to the impedance data.
    Z : np.ndarray[complex]
        Impedance data as a complex array.
    iters : int, optional
        Number of iterations for ECM generation. Default is 100.
    parallel : bool, optional
        If True, the ECM generation will be done in parallel. Default is True.
    tol_linKK : float, optional
        Tolerance for acceptable measurements based on linKK residuals.
    tol : float, optional
        Convergence threshold for the ECM generation. Default is 1e-2.
    num_warmup : int, optional
        Number of warmup samples for the MCMC. Default is 2500.
    num_samples : int, optional
        Number of samples for the MCMC. Default is 1000.

    Returns
    -------
    circuits: pd.DataFrame
        Dataframe containing circuits, parameters, and inference results.
    """
    raise NotImplementedError(
        "This function has some issues and is not yet ready to use. "
        "Regardless, it is recommended to use the step-by-step approach for "
        "more control. Please refer to the documentation for more details."
    )

    # Filter out bad impedance data
    freq, Z = utils.preprocess_impedance_data(freq, Z, tol_linKK=tol_linKK)

    # Generate a pool of potential ECMs via an evolutionary algorithm
    kwargs = {"iters": iters, "complexity": 12, "tol": tol, "parallel": parallel}
    circuits_unfiltered = generate_equivalent_circuits(freq, Z, **kwargs)

    # Apply heuristic rules to filter unphysical circuits
    circuits = filter_implausible_circuits(circuits_unfiltered)

    # Perform Bayesian inference on the filtered ECMs
    kwargs_mcmc = {"num_warmup": num_warmup, "num_samples": num_samples}
    results = perform_bayesian_inference(circuits, freq, Z, **kwargs_mcmc)

    # Add the results to the circuits dataframe as a new column
    circuits["MCMC"] = [result.mcmc for result in results]
    circuits["success"] = [result.converged for result in results]
    circuits["divergences"] = [result.num_divergences for result in results]

    # Compute fitness metrics and add to the dataframe
    circuits = compute_fitness_metrics(circuits, freq, Z)

    return circuits

"""
Collection of functions core to AutoEIS functionality.

.. currentmodule:: autoeis.core

.. autosummary::
   :toctree: generated/

    perform_full_analysis
    generate_equivalent_circuits
    filter_implausible_circuits
    perform_bayesian_inference
    perform_bayesian_inference_batch
    preprocess_impedance_data 

"""
import os
import sys
import time
import warnings
from typing import Union

import jax
import jax.numpy as jnp  # noqa: F401
import numpy as np
import numpyro
import pandas as pd
import psutil
from impedance.validation import linKK
from mpire import WorkerPool
from numpyro.infer import MCMC, NUTS
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

import autoeis.julia_helpers as julia_helpers
import autoeis.visualization as viz
from autoeis import io, metrics, parser, utils
from autoeis.models import circuit_regression, circuit_regression_wrapped  # noqa: F401

# AutoEIS datasets are not small-enough that CPU is much faster than GPU
numpyro.set_platform("cpu")

# HACK: Suppress output until ECSHackWeek/impedance.py/issues/280 is fixed
linKK = utils.suppress_output(linKK)
warnings.filterwarnings("ignore", category=Warning, module="arviz.*")
log = utils.get_logger(__name__)

# Initialize Julia runtime
Main = julia_helpers.init_julia()
ec = julia_helpers.import_backend(Main)

__all__ = [
    "perform_full_analysis",
    "preprocess_impedance_data",
    "generate_equivalent_circuits",
    "filter_implausible_circuits",
    "perform_bayesian_inference",
]


# TODO: Breaks when data is noisy -> use curve_fit to extrapolate R0
def compute_ohmic_resistance(Z: np.ndarray[complex], freq: np.ndarray[float]) -> float:
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


# TODO: Needs heavy refactoring
def preprocess_impedance_data(
    impedance: np.ndarray[complex],
    freq: np.ndarray[float],
    threshold: float = 5e-2,
    plot: bool = False,
) -> tuple[np.ndarray[complex], np.ndarray[float], float]:
    """Preprocess impedance data, filtering out data with a positive
    imaginary part in high frequencies and applying KK validation.

    Parameters
    ----------
    impedance : np.ndarray[complex]
        Impedance measurements.
    freq : np.ndarray[float]
        Frequencies corresponding to impedance measurements.
    threshold : float
        Controls the filtering effect during KK validation.
    plot : bool, optional
        If True, a plot of the processed data will be generated. Default is False.

    Returns
    -------
    tuple
        - Z: Filtered impedance data.
        - freq: Frequencies corresponding to the filtered measurements.
        - rmse: Root mean square error of KK validated data vs. measurements.
    """
    log.info("Pre-processing impedance data using KK filter.")

    # Fetch the real and imaginary part of the impedance data
    Re_Z = impedance.real
    Im_Z = impedance.imag

    # Filter 1 - High Frequency Region
    # Find index where phase_Zwe == minimum, remove all high frequency imag values below zero
    # Find index: 10khz - 100khz
    # ?: What's the logic behind this?
    # BUG: ???
    try:
        pos = np.where((1000 <= freq) & (freq <= 1000000))
        # Find minimum phase value, note returns as a tuple
        (index,) = np.where(abs(Im_Z[pos]) == abs(Im_Z[pos]).min())
    except ValueError:
        index = [0]
    mask_phase = [True] * len(Im_Z)
    for i in range(len(Im_Z)):
        if i < index:
            mask_phase[i] = False

    freq = freq[index[0] :]
    Z = impedance[index[0] :]
    Re_Z = Re_Z[index[0] :]
    Im_Z = Im_Z[index[0] :]

    # Filter 1.2: Delete all impedance points with positive im impedance at high frequency
    pos2 = np.where(Im_Z > 0)
    positive_im = pos2[0][np.where(pos2[0] <= len(pos[0]))]
    freq = np.delete(freq, positive_im)
    Re_Z = np.delete(Re_Z, positive_im)
    Im_Z = np.delete(Im_Z, positive_im)
    Z = np.delete(Z, positive_im)

    # Filter 2 - Low Frequency Region
    # Lin-KK data validation to remove 'noisy' data
    # For Lin-KK, the residuals of Re(Z) and Im(Z) are what will be used as a filter.
    # I have found based on the data set that somewhere ~0.05% works the best
    M, mu, Z_linKK, res_real, res_imag = linKK(
        freq, Z, c=0.5, max_M=100, fit_type="complex", add_cap=True
    )
    rmse = metrics.rmse_score(Z, Z_linKK)

    # Plot residuals of Lin-KK validation
    if plot:
        viz.plot_linKK_residuals(freq, res_real, res_imag)

    # Need to set a threshold limit for when to filter out the noisy data
    # of the residuals threshold = 0.05 !!! USER DEFINED !!!

    # NOTE: 2023/05/03 modification by Runze Zhang
    # ?: What's the logic behind this?
    Zdf_mask = np.arange(1)

    # Keep track of the initial threshold value
    threshold_init = threshold
    step = 0.01

    while len(Zdf_mask) <= 0.7 * len(Z):
        # Filter the data according to imaginary residuals
        mask = [False] * (len(res_imag))
        for i in range(len(res_imag)):
            if res_imag[i] < threshold:
                mask[i] = True
            else:
                break

        freq_mask = freq[mask]
        Z_mask = Z[mask]
        Re_Z_mask = Re_Z[mask]
        Im_Z_mask = Im_Z[mask]

        # Filter the data according to real residuals
        mask = [False] * (len(res_real))
        for i in range(len(res_real)):
            if res_real[i] < threshold:
                mask[i] = True
            else:
                break

        freq_mask = freq[mask]
        Z_mask = Z[mask]
        Re_Z_mask = Re_Z[mask]
        Im_Z_mask = Im_Z[mask]

        # Find the ohmic resistance
        try:
            ohmic_resistance = compute_ohmic_resistance(Z_mask, freq_mask)
            ohmic_resistance_found = True
        except ValueError:
            log.error(
                "Ohmic resistance not found. Check data or increase KK threshold."
            )
            ohmic_resistance_found = False

        # Convert the data to a dataframe for easier manipulation
        values_mask = np.array([freq_mask, Re_Z_mask, Im_Z_mask])
        labels = ["freq", "Zreal", "Zimag"]
        Zdf_mask = pd.DataFrame(values_mask.transpose(), columns=labels)
        threshold += step

    if ohmic_resistance_found:
        log.info(f"Ohmic resistance = {ohmic_resistance}")

    if not np.isclose(threshold - step, threshold_init):
        log.warning(f"Default threshold ({threshold_init}) dropped too many points.")

    Z = Zdf_mask["Zreal"].to_numpy() + 1j * Zdf_mask["Zimag"].to_numpy()
    freq = Zdf_mask["freq"].to_numpy()

    return Z, freq, rmse


def generate_equivalent_circuits(
    impedance: np.ndarray[complex],
    freq: np.ndarray[float],
    iters: int = 100,
    complexity: int = 12,
    tol: float = 1e-2,
    parallel: bool = True,
    generations: int = 30,
    population_size: int = 100,
    seed: int = None,
) -> pd.DataFrame:
    """Generates candidate circuits that fit impedance data using genetic algorithm.

    Parameters
    ----------
    Z : np.ndarray[complex]
        Impedance measurements.
    freq : np.ndarray[float]
        Frequencies corresponding to impedance measurements.
    iters : int, optional
        Number of ECM generation iterations (default is 100).
    complexity : int, optional
        Complexity of the ECM search space (default is 12).
    tol : float, optional
        Convergence threshold for the ECM search (default is 1e-2).
    parallel : bool, optional
        If True, the ECM search will be performed in parallel (default is True).
    generations : int, optional
        Number of generations for the ECM search (default is 30).
    population_size : int, optional
        Number of ECMs to generate per generation (default is 100).
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
        "terminals": "RLP",
        "convergence_threshold": tol,
        "generations": generations,
        "population_size": population_size,
    }

    ecm_generator = _generate_ecm_parallel_julia if parallel else _generate_ecm_serial
    circuits = ecm_generator(impedance, freq, iters, ec_kwargs, seed)

    # Convert Parameters column to dict, e.g., (R1 = 1.0, etc.) -> {"R1": 1.0, etc.}
    circuits = io.parse_ec_output(circuits)

    if not len(circuits):
        log.warning("No plausible circuits found. Try increasing `iters`.")

    return circuits


def _generate_ecm_serial(impedance, freq, iters, ec_kwargs, seed):
    """Generates candidate circuits in serial."""
    global Main
    Main = julia_helpers.init_julia() if Main is None else Main
    ec = julia_helpers.import_backend(Main)

    # Set random seed for reproducibility
    Main.eval(f"import Random; Random.seed!({seed})")

    circuits = []
    for _ in tqdm(range(iters), desc="Circuit Evolution"):
        try:
            circuit = ec.circuit_evolution(impedance, freq, **ec_kwargs)
        except Exception as e:
            log.error(f"Error generating circuit: {e}")
            continue
        if circuit != Main.nothing:
            circuits.append(circuit)

    # Format circuits as a dataframe with columns "circuitstring" and "Parameters"
    df = [(c.circuitstring, Main.string(c.Parameters)) for c in circuits]
    df = pd.DataFrame(df, columns=["circuitstring", "Parameters"])

    return df


# TODO: This function is deprecated, use _generate_ecm_parallel_julia instead
def _generate_ecm_parallel_mpire(impedance, freq, iters, ec_kwargs, seed):
    """Generates candidate circuits in parallel via Python multiprocessing."""

    def circuit_evolution(seed: int):
        """Closure to generate a single circuit to be used with multiprocessing."""
        global Main
        Main = julia_helpers.init_julia() if Main is None else Main
        # Set random seed for reproducibility
        Main.eval(f"import Random; Random.seed!({seed})")
        ec = julia_helpers.import_backend(Main)
        try:
            circuit = ec.circuit_evolution(impedance, freq, **ec_kwargs)
        except Exception as e:
            log.error(f"Error generating circuit: {e}")
            return None
        if circuit == Main.nothing:
            return None
        # Format output as list of strings since Julia objects cannot be pickled
        return [circuit.circuitstring, Main.string(circuit.Parameters)]

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
        raise RuntimeError(
            "Julia must not be manually initialized, restart the kernel."
        )

    # Remove None values
    circuits = [circuit for circuit in circuits if circuit is not None]

    # Format circuits as a dataframe with columns "circuitstring" and "Parameters"
    df = pd.DataFrame(circuits, columns=["circuitstring", "Parameters"])

    return df


def _generate_ecm_parallel_julia(impedance, freq, iters, ec_kwargs, seed):
    """Generates candidate circuits in parallel using Julia multiprocessing."""
    global Main
    Main = julia_helpers.init_julia() if Main is None else Main
    # Set random seed for reproducibility (Python and Julia)
    # FIXME: This doesn't work when multiprocessing, use @everywhere instead
    Main.eval(f"import Random; Random.seed!({seed})")
    Main.eval("import Logging; Logging.disable_logging(Logging.Warn)")

    ec = julia_helpers.import_backend(Main)

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

    with tqdm(total=iters, desc="Circuit Evolution", miniters=1) as pbar:
        for iters_ in iters_chunked:
            try:
                circuits_ = ec.circuit_evolution_batch(
                    impedance, freq, **ec_kwargs, iters=iters_, quiet=True
                )
            except Exception as e:
                log.error(f"Error generating circuits: {e}")
                circuits_ = []
            circuits += circuits_
            pbar.update(iters_)

    # Format output as list of strings since Julia objects cannot be pickled
    circuits_py = []
    for circuit in circuits:
        if circuit != Main.nothing:
            circuits_py.append([circuit.circuitstring, Main.string(circuit.Parameters)])

    # Format circuits as a dataframe with columns "circuitstring" and "Parameters"
    df = pd.DataFrame(circuits_py, columns=["circuitstring", "Parameters"])

    return df


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
    circuits = circuits.copy()

    for row in circuits.itertuples():
        variables = row.Parameters.keys()
        contains_capacitor = any("C" in var for var in variables)
        if contains_capacitor:
            circuits.drop(row.Index, inplace=True)
    circuits.reset_index(drop=True, inplace=True)

    return circuits


def ohmic_resistance_filter(circuits: pd.DataFrame) -> pd.DataFrame:
    """Excludes circuits without an ohmic resistance from the circuits dataframe."""
    circuits = circuits.copy()

    for row in circuits.itertuples():
        circuit = row.circuitstring
        resistors = parser.find_ohmic_resistors(circuit)
        if not resistors:
            circuits.drop(row.Index, inplace=True)

    circuits.reset_index(drop=True, inplace=True)
    return circuits


def series_filter(circuits: pd.DataFrame) -> pd.DataFrame:
    """Excludes circuits with series-only components from the circuits dataframe."""
    circuits = circuits.copy()

    for row in circuits.itertuples():
        circuit = row.circuitstring
        contains_parallel_route = "[" in circuit
        if not contains_parallel_route:
            circuits.drop(row.Index, inplace=True)

    circuits.reset_index(drop=True, inplace=True)
    return circuits


def merge_identical_circuits(circuits: "pd.DataFrame") -> "pd.DataFrame":
    """Merges identical circuits from the circuits dataframe."""
    circuits = circuits.copy()

    for i, row_i in circuits.iterrows():
        circuit_i = row_i.circuitstring
        for j, row_j in circuits.loc[i + 1 :].iterrows():
            if utils.are_circuits_equivalent(circuit_i, row_j.circuitstring):
                circuits.drop(j, inplace=True)

    circuits.reset_index(drop=True, inplace=True)
    return circuits


def perform_bayesian_inference(
    circuits: Union[pd.DataFrame, list[str], str],
    Z: np.ndarray[complex],
    freq: np.ndarray[float],
    p0: Union[np.ndarray, dict, list[dict], list[np.ndarray]] = None,
    num_warmup=2500,
    num_samples=1000,
    num_chains=1,
    seed: Union[int, jax.Array] = None,
    progress_bar: bool = True,
):
    # Ensure inputs are lists
    if isinstance(circuits, str):
        circuits = [circuits]
    if p0 is None:
        p0 = [None] * len(circuits)

    # Validate inputs' types and lengths
    if isinstance(circuits, pd.DataFrame):
        utils.validate_circuits_dataframe(circuits)
        # NOTE: p0 from the dataframe overrides the input p0
        p0 = circuits["Parameters"].tolist()
        circuits = circuits["circuitstring"].tolist()
    for circuit, p0_ in zip(circuits, p0):
        assert isinstance(circuit, str), f"Circuit must be a string: {circuit}"
        valid_p0_types = (dict, np.ndarray, type(None))
        assert isinstance(p0_, valid_p0_types), f"Invalid p0 type: {p0_}"
        num_params = len(parser.get_parameter_labels(circuit))
        assert len(p0_) == num_params, f"Invalid p0 length: {p0_}"

    # Short-circuit if no circuits are provided
    if len(circuits) == 0:
        log.warning("Circuits' dataframe is empty!")
        return None

    return _perform_bayesian_inference_batch(
        circuits=circuits,
        Z=Z,
        freq=freq,
        p0=p0,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        progress_bar=progress_bar,
    )


def _perform_bayesian_inference(
    circuit: str,
    Z: np.ndarray[complex],
    freq: np.ndarray[float],
    p0: Union[np.ndarray[float], dict[str, float]] = None,
    num_warmup=2500,
    num_samples=1000,
    num_chains=1,
    seed: Union[int, jax.Array] = None,
    progress_bar: bool = True,
) -> tuple[numpyro.infer.mcmc.MCMC, int]:
    """Performs Bayesian inference on the circuit based on impedance data.

    Parameters
    ----------
    circuit : str
        Circuit string.
    Z : np.ndarray[complex]
        Complex impedance data.
    freq: np.ndarray[float]
        Frequency data.
    p0 : Union[np.ndarray[float], dict[str, float]], optional
        Initial guess for the circuit parameters (default is None).
    seed : Union[int, jax.Array], optional
        Random seed for reproducibility (default is None).

    Returns
    -------
    tuple(numpyro.infer.mcmc.MCMC, int)
        MCMC object and exit code (0 if successful, -1 if failed).
    """
    log.info("Performing Bayesian inference on the circuit {circuit}.")

    # If the seed is already set using JAX, use it
    assert isinstance(seed, (int, jax.Array, type(None)))
    if isinstance(seed, jax.Array) and len(seed) == 2:
        subkey = seed
    # If seed is int -> use it, else generate a new seed using time
    else:
        seed = time.time_ns() % 2**32 if seed is None else seed
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)

    # Deal with initial values for the circuit parameters
    if p0 is None:
        p0 = utils.fit_circuit_parameters(circuit, Z, freq)
    assert isinstance(p0, dict), "p0 must be a dictionary"

    circuit_fn = utils.generate_circuit_fn(circuit, jit=True)

    # Compute prior predictive distribution using the initial guess
    priors = utils.initialize_priors(p0, variables=p0.keys())
    nuts_kernel = NUTS(
        model=circuit_regression_wrapped,
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
        "Z": Z,
        "freq": freq,
        "priors": priors,
        "circuit_fn": circuit_fn,
    }

    try:
        mcmc.run(subkey, **kwargs_inference)
        exit_code = 0
    except RuntimeError:
        log.error(f"Inference failed for circuit: {circuit}")
        exit_code = -1
    return mcmc, exit_code


def _perform_bayesian_inference_batch(
    circuits: list[str],
    Z: np.ndarray[complex],
    freq: np.ndarray[float],
    p0: list[dict[str, float]] = None,
    num_warmup=2500,
    num_samples=1000,
    num_chains=1,
    seed: Union[int, jax.Array] = None,
    progress_bar=True,
):
    """Performs Bayesian inference on a list of circuits in parallel.

    Parameters
    ----------
    circuits : pd.DataFrame or list[str]
        Dataframe containing circuits or list of circuit strings.
    Z : np.ndarray[complex]
        Complex impedance data.
    freq: np.ndarray[float]
        Frequency data.
    p0 : list[dict[str, float]], optional
        Initial guess for the circuit parameters (default is None).
    seed : int, optional
        Random seed for reproducibility (default is None).

    Returns
    -------
    list[tuple[numpyro.infer.mcmc.MCMC, int]]
        List of MCMC objects and exit codes (0 if successful, -1 if failed).
    """
    # Generate a random seed for each circuit
    N = len(circuits)
    seed = seed or time.time_ns() % 2**32
    key = jax.random.PRNGKey(seed)
    key, *subkeys = jax.random.split(key, N + 1)

    # Multiprocessing requires all inputs to be iterables of the same length
    bi_kwargs = {
        "circuits": circuits,
        "Z": [Z] * N,
        "freq": [freq] * N,
        "p0": p0 if isinstance(p0, list) else [p0] * N,
        "num_warmup": [num_warmup] * N,
        "num_samples": [num_samples] * N,
        "num_chains": [num_chains] * N,
        "seed": subkeys,
        "progress_bar": [False] * N,
    }

    n_cores = psutil.cpu_count(logical=False)

    # Perform Bayesian inference in parallel
    with WorkerPool(n_jobs=n_cores, use_dill=True) as pool:
        results = pool.map(
            _perform_bayesian_inference,
            zip(*bi_kwargs.values()),
            progress_bar=progress_bar,
            progress_bar_style="notebook" if utils.is_notebook() else "rich",
            iterable_len=len(circuits),
        )

    return results


def filter_implausible_circuits(circuits: pd.DataFrame) -> pd.DataFrame:
    """Apply heuristic rules to filter the generated ECMs.

    Parameters
    ----------
    circuits : pd.DataFrame
        Dataframe containing circuits.

    Returns
    -------
    circuits : pd.DataFrame
        Dataframe containing circuits (filtered for plausibility)
    """
    log.info("Filtering the circuits using heuristic rules.")

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
        log.warning(
            "No plausible circuits found. Increase `iters` or `tol` and rerun"
            " `generate_equivalent_circuits`"
        )

    return circuits


def perform_full_analysis(
    Z: np.ndarray[complex],
    freq: np.ndarray[float],
    iters: int = 100,
    parallel: bool = True,
    linKK_threshold: float = 5e-2,
    tol: float = 1e-2,
    num_warmup: int = 2500,
    num_samples: int = 1000,
) -> pd.DataFrame:
    """Performs automated EIS analysis by generating plausible ECMs that
    fit the impedance data, followed by Bayesian inference on components.

    Parameters
    ----------
    Z : np.ndarray[complex]
        Impedance data.
    freq : np.ndarray[float]
        Frequencies corresponding to the impedance data.
    iters : int, optional
        Number of iterations for ECM generation. Default is 100.
    parallel : bool, optional
        If True, the ECM generation will be done in parallel. Default is True.
    linKK_threshold : float, optional
        Threshold for the Lin-KK validation. Default is 5e-2.
    tol : float, optional
        Convergence threshold for the ECM generation. Default is 1e-2.
    num_warmup : int, optional
        Number of warmup samples for the MCMC. Default is 2500.
    num_samples : int, optional
        Number of samples for the MCMC. Default is 1000.

    Returns
    -------
    results: pd.DataFrame
        Dataframe containing circuits, parameters, and MCMC results.
    """
    # Filter out bad impedance data
    Z, freq, rmse = preprocess_impedance_data(Z, freq, threshold=linKK_threshold)

    # Generate a pool of potential ECMs via an evolutionary algorithm
    kwargs = {"iters": iters, "complexity": 12, "tol": tol, "parallel": parallel}
    circuits_unfiltered = generate_equivalent_circuits(Z, freq, **kwargs)

    # Apply heuristic rules to filter unphysical circuits
    circuits = filter_implausible_circuits(circuits_unfiltered)

    # Perform Bayesian inference on the filtered ECMs
    kwargs_mcmc = {"num_warmup": num_warmup, "num_samples": num_samples}
    mcmcs = perform_bayesian_inference(circuits, Z, freq, **kwargs_mcmc)

    # Add the results to the circuits dataframe as a new column
    chains, status = zip(*mcmcs)
    circuits["MCMC (chain)"] = chains
    circuits["MCMC (status)"] = status

    return circuits

"""
Collection of functions core to AutoEIS functionality.

.. currentmodule:: autoeis.core

.. autosummary::
   :toctree: generated/

    perform_full_analysis
    generate_equivalent_circuits
    filter_implausible_circuits
    perform_bayesian_inference
    preprocess_impedance_data

"""

import logging
import os
import time
import warnings
from collections.abc import Iterable, Mapping

import arviz as az
import jax
import jax.numpy as jnp  # noqa: F401
import numpy as np
import numpyro
import pandas as pd
import psutil
from box import Box
from deprecated import deprecated
from impedance.validation import linKK
from jax import config
from mpire import WorkerPool
from numpyro.distributions import Distribution
from numpyro.infer import MCMC, NUTS
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

from autoeis import io, julia_helpers, metrics, parser, utils
from autoeis.models import circuit_regression_complex

# Enforce FP64, otherwise circuit fitter fails (because FP32 gradients not sufficient?)
config.update("jax_enable_x64", True)
# Tell JAX to use CPUs to avoid the annoying "GPU might be present" warning
config.update("jax_platforms", "cpu")
# EIS datasets are not big-enough -> CPU much faster than GPU
numpyro.set_platform("cpu")

# TODO: Suppress output until ECSHackWeek/impedance.py/issues/280 is fixed
linKK = utils.suppress_output_legacy(linKK)

warnings.filterwarnings("ignore", category=Warning, module="arviz.*")
log = logging.getLogger(__name__)

# Initialize Julia runtime
julia_helpers.ensure_julia_deps_ready()
jl = julia_helpers.init_julia()
ec = julia_helpers.import_backend(jl)

__all__ = [
    "perform_full_analysis",
    "preprocess_impedance_data",
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


def preprocess_impedance_data(
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    tol_linKK: float = 5e-2,
    high_freq_threshold: float = 1e3,
    return_aux: bool = False,
) -> tuple[np.ndarray[float], np.ndarray[complex], Box]:
    """Preprocesses/cleans up impedance measurements.

    The preprocessing does the following steps:
        - Discard invalid high frequency measurements (see Notes section)
        - Filter out data with a positive imaginary part in high frequencies
        - Enforce the Kramers-Kronig validation (aka Lin-KK)

    Parameters
    ----------
    freq : np.ndarray[float]
        Frequencies corresponding to impedance measurements.
    Z : np.ndarray[complex]
        Impedance measurements as a complex array.
    tol_linKK : float
        Tolerance for acceptable measurements based on linKK residuals.
    high_freq_threshold : float
        Lower bound for what is considered a high frequency measurement.
    return_aux : bool, optional
        If True, returns the preprocessed data along with auxiliary
        information. Default is False.

    Returns
    -------
    tuple[np.ndarray[float], np.ndarray[complex], Box]
        Tuple containing the preprocessed data with the following elements:
            - freq: Frequencies corresponding to the impedance data.
            - Z: Filtered impedance data.
            - aux: Box containing the Lin-KK validation results with keys:
                - res.real: Residual array for real part of the impedance data.
                - res.imag: Residual array for imaginary part of the impedance data.
                - rmse: Root mean square error of KK validated data vs. measurements.
    """
    log.info("Preprocessing/cleaning up impedance data.")
    n0 = len(freq)

    # Make sure frequency is sorted in descending order
    Z = Z[np.argsort(freq)[::-1]]
    freq = freq[np.argsort(freq)[::-1]]

    # Heuristic 1: @freq->∞: |Z.im|->0 => highest_valid_freq = freq @ np.argmin(|Z.im|)
    high_freq = freq > high_freq_threshold
    idx_highest_valid_freq = np.argmin(np.abs(Z.imag[high_freq]))
    # Filter out frequencies above the highest valid frequency (only works if freq is sorted)
    freq = freq[idx_highest_valid_freq:]
    Z = Z[idx_highest_valid_freq:]

    # Heuristic 2: Remove the data whose Z.imag is positive at high frequencies
    high_freq = freq > high_freq_threshold
    positive_im = Z.imag > 0
    mask = high_freq & positive_im
    Z = Z[~mask]
    freq = freq[~mask]

    # Heuristic 3: Kramers-Kronig validation (aka Lin-KK)
    linKK_kwargs = {"c": 0.5, "max_M": 100, "fit_type": "complex", "add_cap": True}
    M, mu, Z_linKK, res_real, res_imag = linKK(freq, Z, **linKK_kwargs)
    rmse = metrics.rmse_score(Z, Z_linKK)

    mask = (np.abs(res_real) < tol_linKK) & (np.abs(res_imag) < tol_linKK)
    freq = freq[mask]
    Z = Z[mask]

    if (n0 - len(freq)) / n0 > 0.1:
        log.warning("More than 10% of the data was filtered out.")

    aux = Box(res=Box(real=res_real, imag=res_imag), rmse=rmse)

    return (freq, Z, aux) if return_aux else (freq, Z)


def generate_equivalent_circuits(
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    iters: int = 100,
    complexity: int = 12,
    tol: float = 1e-2,
    parallel: bool = True,
    generations: int = 30,
    population_size: int = 100,
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
    circuits = ecm_generator(Z, freq, iters, ec_kwargs, seed)

    # Convert output to DataFrame with columns ("circuitstring", "Parameters")
    circuits = io.parse_ec_output(circuits)

    if not len(circuits):
        log.warning("No plausible circuits found. Increase `iters` or lower `tol`.")

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
    for _ in tqdm(range(iters), desc="Circuit Evolution"):
        try:
            circuit = ec.circuit_evolution(Z, freq, **ec_kwargs, quiet=True)
        except Exception as e:
            log.error(f"Error generating circuit: {e}")
            continue
        circuits.append(circuit)

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

    with tqdm(total=iters, desc="Circuit Evolution", miniters=1) as pbar:
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
    circuits = circuits.copy()
    mcmcs = circuits["MCMC"]

    # Compute WAIC and add to the dataframe
    waic_re = [az.waic(x, var_name="obs_real", scale="deviance") for x in mcmcs]
    waic_im = [az.waic(x, var_name="obs_imag", scale="deviance") for x in mcmcs]
    circuits["WAIC (real)"] = [x["elpd_waic"] + 2 * x["p_waic"] for x in waic_re]
    circuits["WAIC (imag)"] = [x["elpd_waic"] + 2 * x["p_waic"] for x in waic_im]

    # Compute the posterior predictive and add to the dataframe
    # NOTE: axis=1 because posterior is of shape (num_samples, num_obs)
    _fn = lambda r: utils.eval_posterior_predictive(r.MCMC, r.circuitstring, freq)
    circuits["Z_pred"] = circuits.apply(_fn, axis=1)

    # Compute R^2 and add to the dataframe
    _fn = lambda r: metrics.r2_score(Z.real, r.Z_pred.real, axis=1)
    circuits["R^2 (real)"] = circuits.apply(_fn, axis=1)
    _fn = lambda r: metrics.r2_score(Z.imag, r.Z_pred.imag, axis=1)
    circuits["R^2 (imag)"] = circuits.apply(_fn, axis=1)
    # Since R^2 is a vector (num_samples), also compute its mean for convenience
    circuits["R^2 (ravg)"] = circuits["R^2 (real)"].apply(np.mean)
    circuits["R^2 (iavg)"] = circuits["R^2 (imag)"].apply(np.mean)

    # Compute MAPE and add to the dataframe
    _fn = lambda r: metrics.mape_score(Z.real, r.Z_pred.real, axis=1)
    circuits["MAPE (real)"] = circuits.apply(_fn, axis=1)
    _fn = lambda r: metrics.mape_score(Z.imag, r.Z_pred.imag, axis=1)
    circuits["MAPE (imag)"] = circuits.apply(_fn, axis=1)
    # Since MAPE is a vector (num_samples), also compute its mean for convenience
    circuits["MAPE (ravg)"] = circuits["MAPE (real)"].apply(np.mean)
    circuits["MAPE (iavg)"] = circuits["MAPE (imag)"].apply(np.mean)

    # Add number of parameters to the dataframe
    circuits["n_params"] = circuits.apply(lambda r: len(r.Parameters), axis=1)

    # Rank the circuits based on WAIC
    circuits["WAIC (sum)"] = (circuits["WAIC (real)"] * circuits["WAIC (imag)"]) ** 0.5
    circuits.sort_values(by=["WAIC (sum)"], ascending=True, inplace=True, ignore_index=True)

    return circuits


def perform_bayesian_inference(
    circuits: pd.DataFrame | Iterable[str] | str,
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    p0: Iterable[float]
    | Mapping[str, float]
    | Iterable[Iterable[float]]
    | Iterable[Mapping[str, float]] = None,
    priors: Mapping[str, Distribution] = None,
    num_warmup: int = 2500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int | jax.Array = None,
    progress_bar: bool = True,
    refine_p0: bool = False,
) -> list[tuple[MCMC, int]]:
    """Performs Bayesian inference on the circuits based on impedance data.

    Parameters
    ----------
    circuits : pd.DataFrame | Iterable[str] | str
        Dataframe containing circuits or list of circuit strings.
    freq: np.ndarray[float]
        Frequency data corresponding to the impedance data.
    Z : np.ndarray[complex]
        Impedance data as a complex array.
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
    progress_bar : bool, optional
        If True, a progress bar will be displayed (default is True).
    refine_p0 : bool, optional
        If True, the initial guess for the circuit parameters will be refined
        using the circuit fitter (default is False).

    Returns
    -------
    list[tuple[numpyro.infer.mcmc.MCMC, int]]
        List of MCMC objects and exit codes (0 if successful, -1 if failed).
    """
    # Ensure inputs are lists
    if isinstance(circuits, str):
        circuits = [circuits]
    if p0 is None:
        p0 = [None] * len(circuits)
    elif isinstance(p0, (dict, np.ndarray)):
        p0 = [p0] * len(circuits)
    elif isinstance(p0, list):
        assert len(p0) == len(circuits), "Invalid p0 length"

    # Override initial guess in dataframe if p0 is provided
    if isinstance(circuits, pd.DataFrame):
        utils.validate_circuits_dataframe(circuits)
        if p0[0] is not None:
            log.warning("Ignoring parameters in 'circuits' dataframe in favor of 'p0'.")
        else:
            p0 = circuits["Parameters"].tolist()
        circuits = circuits["circuitstring"].tolist()

    # Refine the initial guess if requested or if p0 is still uninitialized
    if refine_p0 or p0[0] is None:
        args = circuits, freq, Z, p0, 25  # iters=25
        p0 = utils.distribute_task(
            utils.fit_circuit_parameters,
            *args,
            static=(1, 2, 4),  # static args = (freq, Z, iters)
            progress_bar=progress_bar,
            desc="Refining p0",
        )

    # Validate inputs' types and lengths
    for circuit, p0_ in zip(circuits, p0):
        assert isinstance(circuit, str), f"Circuit must be a string: {circuit}"
        valid_p0_types = (dict, np.ndarray, type(None))
        assert isinstance(p0_, valid_p0_types), f"Invalid p0 type: {p0_}"
        num_params = len(parser.get_parameter_labels(circuit))
        assert len(p0_) == num_params, f"Invalid p0 length: {p0_}"

    # Short-circuit if no circuits are provided
    if len(circuits) == 0:
        log.warning("'circuits' dataframe is empty!")
        return None

    bi_kwargs = {
        "freq": freq,
        "Z": Z,
        "priors": priors,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "seed": seed,
        "progress_bar": progress_bar,
    }

    if len(circuits) == 1:
        # NOTE: Single inference gets slowed down by progress bar
        bi_kwargs["progress_bar"] = False
        # ?: For single inference, DON'T return a list -> crashes multiprocessing (maybe not?)
        results = [_perform_bayesian_inference(circuits[0], p0=p0[0], **bi_kwargs)]
    else:
        results = _perform_bayesian_inference_batch(circuits, p0=p0, **bi_kwargs)

    return results


def _perform_bayesian_inference(
    circuit: str,
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    p0: Iterable[float] | Mapping[str, float] = None,
    priors: Mapping[str, Distribution] = None,
    num_warmup: int = 2500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int | jax.Array = None,
    progress_bar: bool = False,
) -> tuple[numpyro.infer.mcmc.MCMC, int]:
    """Performs Bayesian inference on the circuit based on impedance data.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    freq : np.ndarray[float]
        Frequencies corresponding to the impedance data.
    Z : np.ndarray[complex]
        Complex impedance data.
    p0 : Iterable[float] | Mapping[str, float], optional
        Initial guess for the circuit parameters (default is None).
    priors: Mapping[str, Distribution], optional
        Priors for the circuit parameters (default is None).
    num_warmup : int, optional
        Number of warmup samples for the MCMC (default is 2500).
    num_samples : int, optional
        Number of samples for the MCMC (default is 1000).
    num_chains : int, optional
        Number of MCMC chains (default is 1).
    seed : int | jax.Array, optional
        Random seed for reproducibility (default is None).
    progress_bar : bool, optional
        If True, a progress bar will be displayed (default is False).

    Returns
    -------
    tuple[MCMC, int]
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

    # TODO: Remove this, circuit fitting must be done in the public API
    # Deal with initial values for the circuit parameters
    if priors is None:
        if p0 is None:
            p0 = utils.fit_circuit_parameters(circuit, freq, Z)
        assert isinstance(p0, dict), "p0 must be a dictionary"
        # Create priors for the circuit parameters based on the initial guess
        priors = utils.initialize_priors(p0, variables=p0.keys())
    else:
        assert isinstance(priors, dict), "'priors' must be a dictionary"

    circuit_fn = utils.generate_circuit_fn(circuit, jit=True)

    nuts_kernel = NUTS(
        model=circuit_regression_complex,
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
        mcmc.run(subkey, **kwargs_inference)
        exit_code = 0
    except RuntimeError as e:
        log.error(f"Inference failed for circuit: {circuit}. Error: {e}")
        exit_code = -1
    return mcmc, exit_code


def _perform_bayesian_inference_batch(
    circuits: Iterable[str],
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    p0: Iterable[Mapping[str, float]] = None,
    priors: Iterable[Mapping[str, Distribution]] = None,
    num_warmup: int = 2500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int | jax.Array = None,
    progress_bar=True,
):
    """Performs Bayesian inference on a list of circuits in parallel.

    Parameters
    ----------
    circuits : pd.DataFrame or list[str]
        Dataframe containing circuits or list of circuit strings.
    freq : np.ndarray[float]
        Frequencies corresponding to the impedance data.
    Z : np.ndarray[complex]
        Complex impedance data.
    p0 : Iterable[Mapping[str, float]], optional
        Initial guess for the circuit parameters (default is None).
    priors : Iterable[Mapping[str, Distribution]], optional
        Priors for the circuit parameters (default is None).
    num_warmup : int, optional
        Number of warmup samples for the MCMC (default is 2500).
    num_samples : int, optional
        Number of samples for the MCMC (default is 1000).
    num_chains : int, optional
        Number of MCMC chains (default is 1).
    seed : int | jax.Array, optional
        Random seed for reproducibility (default is None).
    progress_bar : bool, optional
        If True, a progress bar will be displayed (default is True).

    Returns
    -------
    list[tuple[MCMC, int]]
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
        "freq": [freq] * N,
        "Z": [Z] * N,
        "p0": p0 if isinstance(p0, list) else [p0] * N,
        "priors": priors if isinstance(priors, list) else [priors] * N,
        "num_warmup": [num_warmup] * N,
        "num_samples": [num_samples] * N,
        "num_chains": [num_chains] * N,
        "seed": subkeys,
        "progress_bar": [False] * N,
    }

    n_jobs = min(psutil.cpu_count(logical=False), N)

    # Perform Bayesian inference in parallel
    with warnings.catch_warnings():
        # JAX doesn't work well with multiprocessing, but "spawn" should be fine
        msg_to_ignore = ".*os\\.fork\\(\\).*"
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=msg_to_ignore)
        with WorkerPool(n_jobs=n_jobs, use_dill=True, start_method="spawn") as pool:
            results = pool.map(
                _perform_bayesian_inference,
                zip(*bi_kwargs.values()),
                progress_bar=progress_bar,
                progress_bar_style="notebook" if utils.is_notebook() else "rich",
                progress_bar_options={"desc": "Performing Bayesian Inference"},
                iterable_len=len(circuits),
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
        log.warning("No plausible circuits found. Increase `iters` or `tol` and rerun "
                    "`generate_equivalent_circuits`")  # fmt: skip

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
    results: pd.DataFrame
        Dataframe containing circuits, parameters, and MCMC results.
    """
    # Filter out bad impedance data
    freq, Z = preprocess_impedance_data(freq, Z, tol_linKK=tol_linKK)

    # Generate a pool of potential ECMs via an evolutionary algorithm
    kwargs = {"iters": iters, "complexity": 12, "tol": tol, "parallel": parallel}
    circuits_unfiltered = generate_equivalent_circuits(Z, freq, **kwargs)

    # Apply heuristic rules to filter unphysical circuits
    circuits = filter_implausible_circuits(circuits_unfiltered)

    # Perform Bayesian inference on the filtered ECMs
    kwargs_mcmc = {"num_warmup": num_warmup, "num_samples": num_samples}
    mcmc_results = perform_bayesian_inference(circuits, freq, Z, **kwargs_mcmc)

    # Add the results to the circuits dataframe as a new column
    mcmcs, status = zip(*mcmc_results)
    circuits["MCMC (chain)"] = mcmcs
    circuits["MCMC (status)"] = status

    # Compute fitness metrics and add to the dataframe
    circuits = compute_fitness_metrics(circuits, freq, Z)

    return circuits

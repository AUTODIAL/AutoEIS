"""
Core functions including finding the best fit and Bayesian analysis.

.. currentmodule:: autoeis.core

.. autosummary::
   :toctree: generated/

    perform_full_analysis
    generate_equivalent_circuits
    perform_bayesian_inference
    preprocess_impedance_data 

"""
import itertools
import os
import re
import time
import warnings
from pathlib import Path
from typing import Union

import jax
import jax.numpy as jnp  # noqa: F401
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from impedance.validation import linKK
from jax import random
from mpire import WorkerPool
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

import autoeis.julia_helpers as julia_helpers
import autoeis.utils as utils
import autoeis.visualization as viz

# AutoEIS datasets are not small-enough that CPU is much faster than GPU
numpyro.set_platform("cpu")

# HACK: Suppress output until ECSHackWeek/impedance.py/issues/280 is fixed
linKK = utils.suppress_output(linKK)
warnings.filterwarnings("ignore", category=Warning, module="arviz.*")
log = utils.get_logger(__name__)


__all__ = [
    "perform_full_analysis",
    "generate_equivalent_circuits",
    "perform_bayesian_inference",
    "preprocess_impedance_data",
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
        return (a*x + b) / (x + c)
    bounds = ([0, -np.inf, -np.inf], np.inf)
    try:
        (a, b, c), pcov = curve_fit(func, x, y, bounds=bounds)
        R = 1 / a
    except RuntimeError:
        # Fall back to returning Re(Z) @ highest frequency
        R = Z.real[0]
        log.warning("Failed to fit ohmic resistance, falling back to Re(Z) @ highest frequency.")

    return R


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
    rmse = utils.rmse_score(Z, Z_linKK)

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
            log.error("Ohmic resistance not found. Check data or increase KK threshold.")
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
    tol: float = 5e-4,
    parallel: bool = True,
    generations: int = 30,
    population_size: int = 100,
    seed: int = None,
) -> pd.DataFrame:
    """Generate potential ECMs using evolutionary algorithms.

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
    saveto : str, optional
        Path to the directory where the results will be saved (default is None).
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
        DataFrame containing ECM solutions or None if no solutions are found.
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

    ecm_generator = _generate_ecm_parallel if parallel else _generate_ecm_serial
    circuits = ecm_generator(impedance, freq, iters, ec_kwargs, seed)

    # Convert Parameters column to dict, e.g., (R1 = 1.0, etc.) -> {"R1": 1.0, etc.}
    circuits = utils.parse_circuit_dataframe(circuits)

    if not len(circuits):
        log.warning("No plausible circuits found. Try increasing `iters`.")

    return circuits


def _generate_ecm_serial(impedance, freq, iters, ec_kwargs, seed):
    """Generate potential ECMs using EquivalentCircuits.jl in serial."""
    Main = julia_helpers.init_julia()
    # Suppress Julia warnings (coming from Optim.jl)
    Main.redirect_stderr()
    ec = julia_helpers.import_backend(Main)

    # Set random seed for reproducibility (Python and Julia)
    np.random.seed(seed)
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


def _generate_ecm_parallel(impedance, freq, iters, ec_kwargs, seed):
    """Generate potential ECMs using EquivalentCircuits.jl in parallel."""

    def circuit_evolution(seed: int):
        """Closure to generate a single circuit to be used with multiprocessing."""
        Main = julia_helpers.init_julia()
        # Set random seed for reproducibility (Python and Julia)
        np.random.seed(seed)
        Main.eval(f"import Random; Random.seed!({seed})")
        # Suppress Julia warnings (coming from Optim.jl)
        Main.redirect_stderr()
        # Suppress Julia output (coming from EquivalentCircuits.jl) until
        # MaximeVH/EquivalentCircuits.jl/issues/28 is fixed
        Main.redirect_stdout()
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
        raise RuntimeError("Julia must not be manually initialized, restart the kernel.")

    # Remove None values
    circuits = [circuit for circuit in circuits if circuit is not None]

    # Format circuits as a dataframe with columns "circuitstring" and "Parameters"
    df = pd.DataFrame(circuits, columns=["circuitstring", "Parameters"])

    return df


def split_components(circuits: pd.DataFrame) -> pd.DataFrame:
    """Adds an individual column for each component in the circuit with its value."""
    # Initialize lists to populate the component columns later
    components = {"R": [], "C": [], "L": [], "P": []}
    labels = {"R": "Resistors", "C": "Capacitors", "L": "Inductors", "P": "CPEs"}

    for row in circuits.itertuples():
        circuit = row.circuitstring
        # Find components of each kind
        pgroups = utils.group_parameters_by_component(circuit)
        for ctype in components.keys():
            components[ctype].append(pgroups.get(ctype, []))

    # Add component columns to the dataframe
    for ctype, params in components.items():
        column_label = labels[ctype]
        circuits[column_label] = params

    return circuits


def capacitance_filter(circuits: pd.DataFrame) -> pd.DataFrame:
    """Exclude ideal capacitors from the circuits dataframe.

    Parameters
    ----------
    circuits: pd.DataFrame
       Dataframe containing ECMs (6 columns)

    Returns
    -------
    df_circuits: pd.DataFrame
       Dataframe containing ECMs without ideal capacitors (6 columns)
    """
    circuits = circuits.copy()

    for row in circuits.itertuples():
        variables = row.Parameters.keys()
        contains_capacitor = any("C" in var for var in variables)
        if contains_capacitor:
            circuits.drop(row.Index, inplace=True)
    circuits.reset_index(drop=True, inplace=True)

    return circuits


def ohmic_resistance_filter(circuits: pd.DataFrame) -> pd.DataFrame:
    """Filters the circuits that don't have ohmic resistance in the main chain."""
    circuits = circuits.copy()

    for row in circuits.itertuples():
        circuit = row.circuitstring
        resistors = utils.find_ohmic_resistors(circuit)
        if not resistors:
            circuits.drop(row.Index, inplace=True)

    circuits.reset_index(drop=True, inplace=True)
    return circuits


def series_filter(circuits: pd.DataFrame) -> pd.DataFrame:
    """Filters out circuits without any components connected in parallel."""
    circuits = circuits.copy()

    for row in circuits.itertuples():
        circuit = row.circuitstring
        contains_parallel_route = "[" in circuit
        if not contains_parallel_route:
            circuits.drop(row.Index, inplace=True)

    circuits.reset_index(drop=True, inplace=True)
    return circuits


def circuit_to_function(circuit: str, use_jax=True) -> callable:
    """Converts a circuit string to a callable function."""
    circuit_df = pd.DataFrame([circuit], columns=["circuitstring"])
    eqn = generate_mathematical_expression(circuit_df)["Mathematical expressions"][0]
    if use_jax:
        eqn = eqn.replace("np", "jnp")
    def _fn(X, F):
        assert utils.count_params(circuit) == len(X), "Invalid number of parameters."
        return eval(eqn)
    return jax.jit(_fn) if use_jax else _fn


def generate_mathematical_expression(df_circuits: pd.DataFrame) -> pd.DataFrame:
    """
    Generates the mathematical expression of each circuit.

    Parameters
    ----------
    df_circuits: pd.DataFrame
        Dataframe containing the generated ECMs (6 columns)

    Returns
    -------
    df_circuits: pd.DataFrame
        Dataframe containing the generated ECMs with mathematical expressions (7 columns)

    """
    # Define two kinds of pattern to find all elements in the circuit
    test_pattern = re.compile(r"([CLRP])([0-9]+)+")
    test_pattern_2 = re.compile(r"[CLRP][0-9]+")

    # Create a list to store the mathematical expressions
    new_circuits = []

    for i in range(len(df_circuits["circuitstring"])):
        circuit = df_circuits["circuitstring"][i]
        for j, k in zip(["-", "[", ",", "]"], ["+", "((", ")**(-1)+(", ")**(-1))**(-1)"]):
            circuit = circuit.replace(j, k)
        test_results = test_pattern.findall(circuit)
        test_results_2 = test_pattern_2.findall(circuit)

        test_results.reverse()
        test_results_2.reverse()

        for m in range(len(test_results)):
            if test_results[m][0] == "R":
                circuit = circuit.replace(test_results_2[m], "X")
            elif test_results[m][0] == "C":
                circuit = circuit.replace(test_results_2[m], "(1/(2*1j*np.pi*F*X))")
            elif test_results[m][0] == "L":
                circuit = circuit.replace(test_results_2[m], "(2*1j*np.pi*F*X)")
            elif test_results[m][0] == "P":
                circuit = circuit.replace(
                    test_results_2[m],
                    "(1/(X*(2*1j*np.pi*F)**(Y)))"
                )

        new_temp_circuit = []
        counter = 0

        for n in range(len(circuit)):
            if circuit[n] == "X":
                new_temp_circuit.append(f"X[{str(counter)}]")
                counter += 1
            elif circuit[n] == "Y":
                new_temp_circuit.append(f"X[{str(counter)}]")
                counter += 1
            else:
                new_temp_circuit.append(circuit[n])
        new_circuit = "".join(new_temp_circuit)
        new_circuits.append(new_circuit)

    df_circuits["Mathematical expressions"] = new_circuits

    return df_circuits


def s_to_a_convert(circuit: str) -> "np.ndarray":
    """
    Converts the circuit configuration strings to ndarray (representing
    components by numbers).

    Parameters
    ----------
    circuit: str
        String that stores the configuration of the circuit

    Returns
    -------
    circuit_array: np.ndarray
        The ndarray that stores the circuit configurations

    """
    # Delete the numbers included in str
    circuit = re.sub(r"[0-9]+", "", circuit)
    pat = re.compile(r"[RCLP\[\]\-,]")
    str = pat.findall(circuit)
    circuit_array = np.zeros((1, len(str)))
    for i in range(len(str)):
        # Encoding rules: R=1，C=2，L=3，P=4，[=5,]=6,"-"=7,","=8
        if str[i] == "R":
            circuit_array[0, i] = 1
        elif str[i] == "C":
            circuit_array[0, i] = 2
        elif str[i] == "L":
            circuit_array[0, i] = 3
        elif str[i] == "P":
            circuit_array[0, i] = 4
        elif str[i] == "[":
            circuit_array[0, i] = 5
        elif str[i] == "]":
            circuit_array[0, i] = 6
        elif str[i] == "-":
            circuit_array[0, i] = 7
        elif str[i] == ",":
            circuit_array[0, i] = 8
    return circuit_array[0]


def count_components(circuit: str, symbols: bool = True) -> "np.ndarray":
    """
    Counts the occurances of each kind of component given a circuit
    string (R/C/L/P/-/,/[/]).

    Parameterss
    -----------
    circuit: str
        String that stores the configuration of the circuit
    symbols: bool
        Determines whether to count the number of connectors (-/,/[/])

    Returns
    -------
    array_components_numbers: np.ndarray
        The ndarray that stores the number of each kind of components

    """
    # Define the pattern for each components (p denotes pattern)
    r_p = re.compile(r"[R]")
    c_p = re.compile(r"[C]")
    l_p = re.compile(r"[L]")
    p_p = re.compile(r"[P]")
    if symbols:
        b_p = re.compile(r"[\[\]]")
        d_p = re.compile(r"[\-]")
        comma_p = re.compile(r"[,]")

    # Find the individual components (n denotes number)
    r_n = r_p.findall(circuit)
    c_n = c_p.findall(circuit)
    l_n = l_p.findall(circuit)
    p_n = p_p.findall(circuit)
    if symbols:
        b_n = b_p.findall(circuit)
        d_n = d_p.findall(circuit)
        comma_n = comma_p.findall(circuit)

    # Create array to store the numbers of components
    if symbols:
        array_components_numbers = np.zeros((1, 7))
    else:
        array_components_numbers = np.zeros((1, 4))

    # Store the numbers of components into array
    array_components_numbers[0, 0] = len(r_n)
    array_components_numbers[0, 1] = len(c_n)
    array_components_numbers[0, 2] = len(l_n)
    array_components_numbers[0, 3] = len(p_n)
    if symbols:
        array_components_numbers[0, 4] = len(
            b_n
        )  # /2 can be used to find the number of separated parallel structure
        array_components_numbers[0, 5] = len(d_n)
        array_components_numbers[0, 6] = len(
            comma_n
        )  # Add this number with the number of separated parallel sturcture can get the number of subcircuits

    return array_components_numbers[0]


def rank_the_structure(circuit_array: "np.ndarray") -> "np.ndarray":
    """
    Ranks each component in a given circuit according to its 'complexity'
    (defined by the number of parallel structures it has).

    Parameters
    ----------
    circuit_array: np.ndarray
        The ndarray that stores the circuit configurations

    Returns
    -------
    ranks_array: np.array
        The ndarray that stores the level information of given circuits

    """

    ranks_array = np.zeros([1, len(circuit_array)])
    ranker = 0
    for i in range(len(circuit_array)):
        if circuit_array[i] == 5:
            ranker += 1
            # -1: [
            ranks_array[0, i] = -1
            # elif circuit_array[i] == 8:
        #    ranker += 0.5
        #    ranks_store[0,i] = -3
        # -3: ,
        elif circuit_array[i] == 6:
            ranker -= 1
            ranks_array[0, i] = -2
        else:
            ranks_array[0, i] = ranker
    return ranks_array[0]


def structure_deconstructor(ranks_array: "np.ndarray") -> list:
    """
    Extracts the index information of each level circuit according to
    the ranks array.

    Parameters
    ----------
    ranks_array: np.ndarray
        The ndarray that stores the level information of given circuits

    Returns
    -------
    indexs_lists: list
        The list that stores the index information of each level circuit

    """
    # Initialize a list to store the indexs
    indexs_list = []
    # Initialize a list to store the index lists
    indexs_lists = []
    for i in range(len(ranks_array)):
        if ranks_array[i] < 0:
            indexs_lists.append(indexs_list)
            indexs_list = []
        elif ranks_array[i] >= 1:
            indexs_list.append(i)
    indexs_lists = [x for x in indexs_lists if x != []]
    return indexs_lists


def structure_extractor(
    circuit_array: "np.ndarray", ranks_array: "np.ndarray", indexs_lists: list
) -> "np.ndarray":
    """
    Extracts the circuit configuration at each level.

    Parameters
    ----------
    circuit_array: np.andrray
        The ndarray that stores the circuit configurations
    ranks_array: np.ndarray
        The ndarray that stores the level information of given circuits
    indexs_lists: list
        The list that stores the index information of each level circuit

    Return
    ------
    characteristic_array: np.ndarray
        The nparray that stores the circuit configuration separated at different levels

    """
    characteristic_array = np.zeros(
        [len(indexs_lists), max(len(index) for index in indexs_lists) + 1]
    )
    for i in range(len(indexs_lists)):
        if ranks_array[indexs_lists[i][0]] == 1:
            characteristic_array[i][0] = 1
        elif ranks_array[indexs_lists[i][0]] == 2:
            characteristic_array[i][0] = 2
        elif ranks_array[indexs_lists[i][0]] == 3:
            characteristic_array[i][0] = 3
        elif ranks_array[indexs_lists[i][0]] == 4:
            characteristic_array[i][0] = 4
        elif ranks_array[indexs_lists[i][0]] == 5:
            characteristic_array[i][0] = 5
        else:
            log.error("Circuit is too complex")

        segment = np.array(sorted(circuit_array[indexs_lists[i]]))
        characteristic_array[i][1 : 1 + len(segment)] = segment.reshape(1, len(segment))

    sort_list = [characteristic_array[:, i] for i in range(characteristic_array.shape[1])]
    idex = np.lexsort(sort_list)

    characteristic_array = characteristic_array[idex, :]

    return characteristic_array


def precise_rank_the_structure(circuit_array: np.array) -> np.array:
    """Rank each component in given circuits according to its 'complexity'
    defined by how many parallel structures it's nested in.

    Parameters
    ----------
    circuit_array: np.array
        the nparray that stores the circuit configurations

    Returns
    -------
    ranks_array: np.array
        the nparray that stores the level information of given circuits
    """
    ranks_array = np.zeros([1, len(circuit_array)])
    ranker = 0
    for i in range(len(circuit_array)):
        if circuit_array[i] == 5:
            ranker += 1
            ranker = int(ranker)
            # -1: [
            ranks_array[0, i] = ranker
        elif circuit_array[i] == 8:
            ranker += 0.1
            # -3: ,
            ranks_array[0, i] = ranker
        elif circuit_array[i] == 6:
            ranker -= 1
            ranker = int(ranker)
            # -2 : ]
            ranks_array[0, i] = ranker
        else:
            ranks_array[0, i] = ranker
    return ranks_array[0]


def precise_extractor(circuit_array: np.ndarray, precise_ranks_array: np.ndarray) -> list:
    """Extracts the index information of each level circuit according to
    ranking array in a more precise way.

    Parameters
    ----------
    circuit_array: np.ndarray
        The ndarray that stores the circuit configurations
    precise_ranks_array:np.array
        The ndarray that stores the level information of given circuits

    Returns
    -------
    level_lists: list
        The list that stores the index information of each level circuit
    """
    level_lists = []
    if circuit_array[0] == 5:
        # detect [
        # print(1,0)
        level_list = []
        level_list.append(int(precise_ranks_array[0]))
        for j in range(0, len(precise_ranks_array) - 1):
            if (
                precise_ranks_array[j] > int(precise_ranks_array[j])
                and int(precise_ranks_array[j]) == precise_ranks_array[0]
            ):
                # print("1 end:",j)
                level_list.extend(z for z in circuit_array[0 + 1 : j])
                level_lists.append(level_list)
                break
    for i in range(len(precise_ranks_array) - 1):
        if int(precise_ranks_array[i + 1]) - int(precise_ranks_array[i]) == 1:
            # detect[
            # print(1,i)
            level_list = []
            level_list.append(int(precise_ranks_array[i + 1]))
            for j in range(i, len(precise_ranks_array) - 1):
                if (
                    precise_ranks_array[j] > int(precise_ranks_array[j])
                    and int(precise_ranks_array[j]) == precise_ranks_array[i + 1]
                ):
                    # print("1 end:",j)
                    level_list.extend(z for z in circuit_array[i + 2 : j])
                    level_lists.append(level_list)
                    break
        elif round(precise_ranks_array[i + 1] - precise_ranks_array[i], 1) == 0.1:
            # detect ","
            # print(2,i)
            level_list = []
            level_list.append(int(precise_ranks_array[i + 1]))
            for j in range(i + 2, len(precise_ranks_array)):
                if (
                    precise_ranks_array[j] > precise_ranks_array[j - 1]
                    and int(precise_ranks_array[j]) == int(precise_ranks_array[j - 1])
                    and int(precise_ranks_array[j]) == int(precise_ranks_array[i + 1])
                    or precise_ranks_array[j] == int(precise_ranks_array[i] - 1)
                ):
                    # print("2 end:", j)
                    level_list.extend(z for z in circuit_array[i + 2 : j])
                    level_lists.append(level_list)
                    break
    return level_lists


def sort_level_lists(level_lists: "list") -> "list":
    """Sorts the level lists.

    Parameters
    ----------
    level_lists: list
        The list that stores the index information of each level circuit

    Returns
    -------
    level_lists: list
        The sorted list that stores the index information of each level circuit
    """
    for i in range(len(level_lists)):
        level_lists[i] = sorted(level_lists[i])
    level_lists = sorted(level_lists)
    return level_lists


def feature_store(circuit: str) -> dict:
    """Extract the features of the circuit and store them as a dictionary
     with the following keys:

        - Feature 1: the component numbers (R/C/L/P/-/,/[/]) of a given circuit
        - Feature 2: the component numbers in series part of a given circuit
        - Feature 2.5: the parallel parts of a given circuit at different levels
        - Feature 3: the information of all parallel subcircuits

    Parameters
    ----------
    circuit: str
        The string that stores the circuit configuration

    Returns
    -------
    characteristic_features: dict
        The dictionary that stores the above 4 characteristics of a given circuit
    """
    circuit = circuit
    circuit_array = s_to_a_convert(circuit)
    ranks_array = rank_the_structure(circuit_array)
    indexs_lists = structure_deconstructor(ranks_array)
    precise_ranks_array = precise_rank_the_structure(circuit_array)
    test_pattern = re.compile(r"\[")

    # Features 1 - the numbers of each kind of element are equal
    components_numbers = count_components(circuit)
    # Feature 2 - same series configurations
    series_numbers = count_components(find_series_elements(circuit))
    if test_pattern.findall(circuit) is True:
        # Feature 2.5 - same configurations at different parallel levels
        characteristic_array = structure_extractor(circuit_array, ranks_array, indexs_lists)
        # Feature 3 - all parallel subcircuit shoule be identical
        level_lists = precise_extractor(circuit_array, precise_ranks_array)
        level_lists = sort_level_lists(level_lists)

    # Store the features
    characteristic_features = {}
    characteristic_features["Circuit_Name"] = circuit
    characteristic_features["Feature 1"] = components_numbers
    characteristic_features["Feature 2"] = series_numbers
    if test_pattern.findall(circuit) is True:
        characteristic_features["Feature 2.5"] = characteristic_array
        characteristic_features["Feature 3"] = level_lists

    return characteristic_features


def identifior(df_circuits: "pd.DataFrame") -> (list, list):
    """Identfy the identical circuits configurations by the above features

    Parameters
    ----------
    df_circuits: pd.DataFrame
        Dataframe containing filtered ECMs with mathematical expressions (7 columns)

    Returns
    -------
    equal_lists: list
        The list that stores the strings of identical circuit configurations
    equal_lists_seq: list
        The list that stores the index of identical circuits
    """
    equal_lists = []
    equal_lists_seq = []
    for i in range(len(df_circuits["circuitstring"])):
        feature_i = feature_store(df_circuits["circuitstring"][i])
        equal_list = []
        equal_list_seq = []
        for j in range(len(df_circuits["circuitstring"])):
            feature_j = feature_store(df_circuits["circuitstring"][j])
            # if (feature_i['Feature 1'] == feature_j['Feature 1']).all() and (feature_i['Feature 2'] == feature_j['Feature 2']).all() and (feature_i['Feature 2.5'] == feature_j['Feature 2.5']).all() and feature_i['Feature 3'] == feature_j['Feature 3']:
            if len(feature_i) == len(feature_j) == 5:
                if (
                    feature_i["Feature 1"].tolist() == feature_j["Feature 1"].tolist()
                    and feature_i["Feature 2"].tolist() == feature_j["Feature 2"].tolist()
                    and feature_i["Feature 2.5"].tolist() == feature_j["Feature 2.5"].tolist()
                    and feature_i["Feature 3"] == feature_j["Feature 3"]
                ):
                    equal_list.append(df_circuits["circuitstring"][j])
                    equal_list_seq.append(j)
            else:
                if (
                    feature_i["Feature 1"].tolist() == feature_j["Feature 1"].tolist()
                    and feature_i["Feature 2"].tolist() == feature_j["Feature 2"].tolist()
                ):
                    equal_list.append(df_circuits["circuitstring"][j])
                    equal_list_seq.append(j)
        equal_lists.append(equal_list)
        equal_lists_seq.append(equal_list_seq)
    return equal_lists, equal_lists_seq


def filter(similar_circuits: "list") -> "list":
    """Filter the repeated "identical circuits list" in the list.

    Parameters
    ----------
    similar_circuits: list
        The list that stores the index of identical circuits or the strings
        of identical circuit configurations

    Returns
    -------
    equal_list_filtered: list
        The processed list that stores the index of
    """

    similar_circuits.sort()
    equal_list_filtered = list(
        similar_circuits for similar_circuits, _ in itertools.groupby(similar_circuits)
    )
    return equal_list_filtered


def circuit_expression_combine_lists(df_circuits: "pd.DataFrame") -> (list, list):
    """Identfy the identical circuits configurations by the above features.

    Parameters
    ----------
    df_circuits: pd.DataFrame
        Dataframe containing filtered ECMs with mathematical expressions (7 columns)

    Returns
    -------
    similar_expression: list
        The processed list that stores the strings of identical circuit configurations
    similar_expression_index: list
        The processed list that stores the index of identical circuits
    """
    similar_lists = identifior(df_circuits)
    similar_expression = filter(similar_lists[0])
    similar_expression_index = filter(similar_lists[1])

    return similar_expression, similar_expression_index


def component_values(input: "pd.Series") -> (list, list, list):
    """Before combination, separates the components names and values for
    further comparison to identify identical circuit values.

    Parameters
    ----------
    df_circuits['Parameters']: pd.Series
        the series that stores the names and values information of a given circuit

    Returns
    -------
    component_values_lists: list
        The list that stores the component values and component names
    values_lists: list
        The list that only stores the component values
    names_lists: list
        The list that only stores the component names
    """
    # Remove brackets '(' and ')' from the string
    remove_brackets = re.compile(r"[^()]")

    # Store the values of each component in 4 digits
    digit_p = re.compile(r"\-?[0-9]+\.[0-9]+e*-*[0-9]*")

    # Store the names of each component
    name_p = re.compile(r"[A-Z][0-9]*[a-z]? = ")

    # Define a pattern to round the number
    e_p = re.compile(r"e")

    # Create lists to store these data
    values_lists = []
    component_values_lists = []
    names_lists = []

    for i in range(len(input)):
        # Store the values of each component
        values_list = digit_p.findall(input[i])
        for j in range(len(values_list)):
            if e_p.findall(values_list[j]) is False:
                values_list[j] = "%e" % values_list[j]
            values_list[j] = float(values_list[j])
            values_list[j] = "{:0.4e}".format(values_list[j])
            values_list[j] = float(values_list[j])
        values_lists.append(values_list)

        # Store the names of each component
        names_list = name_p.findall(input[i])
        names_lists.append(names_list)

        # Combine the names with values
        component_values_list = []
        for k in range(len(values_list)):
            seq = [names_list[k], str(values_list[k])]
            component_values_list.append("".join(seq))
        component_values_lists.append(component_values_list)

    return component_values_lists, values_lists, names_lists


def merge_identical_circuits(circuits: "pd.DataFrame") -> "pd.DataFrame":
    """Merges identical circuits (removes rows with equivalent circuits)."""
    circuits = circuits.copy()

    for i, row_i in circuits.iterrows():
        circuit_i = row_i.circuitstring
        for j, row_j in circuits.loc[i+1:].iterrows():
            if utils.is_equivalent(circuit_i, row_j.circuitstring):
                circuits.drop(j, inplace=True)

    circuits.reset_index(drop=True, inplace=True)
    return circuits
 
 
def perform_bayesian_inference(
    circuit: str,
    Z: np.ndarray[complex],
    freq: np.ndarray[float],
    p0: Union[np.ndarray[float], dict[str, float]] = None,
    num_warmup=500,
    num_samples=500,
    seed: int = None,
    progress_bar: bool = True,
) -> numpyro.infer.mcmc.MCMC:
    """Performs Bayesian inference on the circuit based on EIS measurements.

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
    seed : int, optional
        Random seed for reproducibility (default is None).
        
    Returns
    -------
    mcmc : numpyro.infer.mcmc.MCMC
        MCMC object.
    """
    log.info("Performing Bayesian inference on the circuit {circuit}.")

    # Set the seed for reproducibility (if not set, use current time in nanoseconds)
    seed = seed or time.time_ns() % 2**32
    np.random.seed(seed)
    rng_key = random.PRNGKey(seed)

    # Deal with initial values for the circuit parameters
    if p0 is None:
        p0 = utils.fit_circuit_parameters(circuit, Z, freq)
    assert isinstance(p0, dict), "p0 must be a dictionary"
    p0_values = np.array(list(p0.values()))

    # Zfunc = circuit_to_function(circuit, use_jax=True)
    circuit_fn = utils.generate_circuit_fn(circuit)
    Z_pred = circuit_fn(p0_values, freq)

    # Compute R2, MSE, RMSE, and MAPE using the initial guess
    r2_init = utils.r2_score(Z, Z_pred)
    r2_real_init = utils.r2_score(Z.real, Z_pred.real)
    r2_imag_init = utils.r2_score(Z.imag, Z_pred.imag)
    mse_init = utils.mse_score(Z, Z_pred)
    rmse_init = utils.rmse_score(Z, Z_pred)
    mape_init = utils.mape_score(Z, Z_pred)

    log.info(f"R² = {r2_init:.3f} (initial)")
    log.info(f"R² (Re) = {r2_real_init:.3f} (initial)")
    log.info(f"R² (Im) = {r2_imag_init:.3f} (initial)")
    log.info(f"MSE = {mse_init:.3e} (initial)")
    log.info(f"RMSE = {rmse_init:.3e} (initial)")
    log.info(f"MAPE = {mape_init:.3f}% (initial)")

    def model(F, Z_true, priors: dict, circuit_func: callable):
        # Sample each element of X separately
        X = jnp.array([numpyro.sample(k, v) for k, v in priors.items()])
        # Predict Z using the model
        Z_pred = circuit_func(X, F)
        # Define observation model for real and imaginary parts of Z
        sigma_real = numpyro.sample("sigma_real", dist.Exponential(rate=1.0))
        numpyro.sample("obs_real", dist.Normal(Z_pred.real, sigma_real), obs=Z_true.real)
        sigma_imag = numpyro.sample("sigma_imag", dist.Exponential(rate=1.0))
        numpyro.sample("obs_imag", dist.Normal(Z_pred.imag, sigma_imag), obs=Z_true.imag)

    # Compute prior predictive distribution using the initial guess
    prior_predictive = Predictive(model, num_samples=200)
    priors = utils.initialize_priors(p0, variables=p0.keys())
    rng_key, rng_subkey = random.split(rng_key)
    kwargs = {"F": freq, "Z_true": Z, "priors": priors, "circuit_func": circuit_fn}
    prior_prediction = prior_predictive(rng_subkey, **kwargs)
    
    nuts_kernel = NUTS(model)
    kwargs_mcmc = {
        "num_samples": num_samples,
        "num_warmup": num_warmup,
        "num_chains": 1,
        "progress_bar": progress_bar
    }
    mcmc = MCMC(nuts_kernel, **kwargs_mcmc)
    rng_key, rng_subkey = jax.random.split(rng_key)
    mcmc.run(rng_subkey, F=freq, Z_true=Z, priors=priors, circuit_func=circuit_fn)

    # Calculate AIC
    # FIXME: Remove next line once confirmed that `iloc` is correctly used.
    # AIC_value = az.waic(mcmc_i)[0] * (-2) + 2 * utils.count_params(circuit)
    # aic = az.waic(mcmc).iloc[0] * (-2) + 2 * utils.count_params(circuit)
    # log.info(f"AIC = {aic:.1f}")

    return mcmc


# TODO: Don't filter for ohmic resistance value, just ensure circuit contains it
def apply_heuristic_rules(circuits: pd.DataFrame) -> pd.DataFrame:
    """Apply heuristic rules to filter the generated ECMs.

    Parameters
    ----------
    circuits : pd.DataFrame
        DataFrame containing the generated ECMs.

    Returns
    -------
    circuits : pd.DataFrame
        DataFrame containing the filtered ECMs.
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

    return circuits


def perform_full_analysis(
    impedance: np.ndarray[complex],
    freq: np.ndarray[float],
    iters: int = 100,
    saveto: str = None,
    plot: bool = False,
    draw_ecm: bool = False,
    parallel: bool = True,
) -> pd.DataFrame:
    """Perform automated EIS analysis by generating plausible ECMs
    followed by Bayesian inference on component values.

    Parameters
    ----------
    impedance : np.ndarray[complex]
        Impedance data.
    freq : np.ndarray[float]
        Frequencies corresponding to the impedance data.
    iters : int, optional
        Number of iterations for ECM generation. Default is 100.
    saveto : str
        Path to the directory where the results will be saved.
    plot : bool, optional
        If True, the results will be plotted. Default is False.
    draw_ecm : bool, optional
        If True, the ECM will be plotted. Default is False.
    parallel : bool, optional
        If True, the ECM generation will be done in parallel. Default is True.

    Returns
    -------
    results: pd.DataFrame
        Dataframe containing plausible ECMs with Bayesian inference results.
    """
    # Make a new folder to store the results
    if saveto is not None:
        Path(saveto).mkdir(parents=True, exist_ok=True)

    # Filter out bad impedance data
    Z, freq, rmse = preprocess_impedance_data(impedance, freq, threshold=0.05, plot=plot)
    
    # Generate a pool of potential ECMs via an evolutionary algorithm
    kwargs = {"iters": iters, "complexity": 12, "tol": 1e-1, "saveto": saveto, "parallel": parallel}
    circuits_unfiltered = generate_equivalent_circuits(Z, freq, **kwargs)

    # Apply heuristic rules to filter unphysical circuits
    ohmic_resistance = compute_ohmic_resistance(Z, freq)
    circuits = apply_heuristic_rules(circuits_unfiltered, ohmic_resistance)

    # Perform Bayesian inference on the filtered ECMs
    eis_data = pd.DataFrame({"freq": freq, "Zreal": Z.real, "Zimag": Z.imag})
    kwargs = {"saveto": saveto, "plot": plot, "draw_ecm": draw_ecm}
    results = perform_bayesian_inference(eis_data, circuits, **kwargs)

    return results

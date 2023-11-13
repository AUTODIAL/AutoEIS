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

import arviz as az
import dill
import jax.numpy as jnp  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from impedance.validation import linKK
from jax import random
from mpire import WorkerPool
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, Predictive
from tqdm.auto import tqdm

import autoeis.julia_helpers as julia_helpers
import autoeis.utils as utils
import autoeis.visualization as viz

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


def find_ohmic_resistance(
    reals: np.array, imags: np.array, saveto: str = None
) -> float:
    """Extracts the ohmic resistance of impedance data by performing 5th
    order polynomial fit (with 5% tolerance as relative error).

    Parameters
    ----------
    reals: np.ndarray[float]
        The real part of the impedance data
    imags: np.ndarray[float]
        The imag part of the impedance data
    fname: str
        The storage path

    Returns
    -------
    ohmic_resistance: float
        The ohmic resistance of impedance data
    """
    # TODO: The logic behind this function is not robust enough, need to be improved
    # Select the high-frequency impedance data
    high_f_real = reals[0:10]
    high_f_imag = imags[0:10]
    high_f_phase = np.arctan(high_f_imag / high_f_real)
    # Find minimum phase value, note returns as a tuple
    (index,) = np.where(abs(high_f_phase) == abs(high_f_phase).min())
    # Extract the ohmic resistance
    ohmic_resistance = high_f_real[index[0]]

    if saveto is not None:
        fpath = os.path.join(saveto, "ohmic_resistance.txt")
        np.savetxt(fpath, [ohmic_resistance])

    return ohmic_resistance


def preprocess_impedance_data(
    impedance: np.ndarray[complex],
    freq: np.ndarray[float],
    threshold: float,
    saveto: str = None,
    plot: bool = False
) -> tuple[pd.DataFrame, float, float]:
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
    saveto : str
        Path to the directory where the results will be saved.
    plot : bool, optional
        If True, a plot of the processed data will be generated. Default is False.

    Returns
    -------
    tuple
        - Zdf_mask (pd.DataFrame): Preprocessed impedance data.
        - ohmic_resistance (float): Ohmic resistance extracted from impedance data.
        - rmse (float): Root mean square error of KK validated data vs. measurements.
    """
    log.info("Pre-processing impedance data using KK filter.")

    # Make a new folder to store the results
    if saveto is not None:
        Path(saveto).mkdir(parents=True, exist_ok=True)

    # Fetch the real and imaginary part of the impedance data
    Re_Z = impedance.real
    Im_Z = impedance.imag

    if plot:
        fpath = os.path.join(saveto, "nyquist_and_bode_raw.png")
        viz.plot_impedance(impedance, freq, saveto=fpath)

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
    rmse = RMSE_calculator(Z, Z_linKK)

    # Plot residuals of Lin-KK validation
    if plot:
        fpath = os.path.join(saveto, "linkk_residuals.png")
        viz.plot_linKK_residuals(freq, res_real, res_imag, saveto=fpath)

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
            ohmic_resistance = find_ohmic_resistance(Re_Z_mask, Im_Z_mask, saveto=saveto)
        except ValueError:
            log.error("Ohmic resistance not found. Check data or increase KK threshold.")

        # Convert the data to a dataframe for easier manipulation
        values_mask = np.array([freq_mask, Re_Z_mask, Im_Z_mask])
        labels = ["freq", "Zreal", "Zimag"]
        Zdf_mask = pd.DataFrame(values_mask.transpose(), columns=labels)
        threshold += step

    log.info(f"Ohmic resistance = {ohmic_resistance}")

    # Plot the filtered Nyquist and Bode plots
    if plot:
        saveto = os.path.join(saveto, "nyquist_and_bode_filtered.png")
        viz.plot_impedance(Z_mask, freq_mask, saveto=saveto)

    if not np.isclose(threshold - step, threshold_init):
        log.warning(f"Default threshold ({threshold_init}) dropped too many points.")

    return Zdf_mask, ohmic_resistance, rmse


def generate_equivalent_circuits(
        impedance: np.ndarray[complex],
        freq: np.ndarray[float],
        iters: int = 100,
        complexity: int = 12,
        tol: float = 5e-4,
        saveto: str = None,
        parallel: bool = True,
        generations: int = 30,
        population_size: int = 100,
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

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing ECM solutions or None if no solutions are found.
    """
    log.info("Generating equivalent circuits via evolutionary algorithms.")

    ec_kwargs = {
        "head": complexity,
        "terminals": "RLP",
        "convergence_threshold": tol,
        "generations": generations,
        "population_size": population_size,
    }

    ecm_generator = _generate_ecm_parallel if parallel else _generate_ecm_serial
    circuits = ecm_generator(impedance, freq, iters, ec_kwargs)

    if not len(circuits):
        log.warning("No plausible circuits found. Try increasing `iters`.")

    if saveto is not None:
        fpath = os.path.join(saveto, "circuits_dataframe.csv")
        circuits.to_csv(fpath, index=False)

    return circuits


def _generate_ecm_serial(impedance, freq, iters, ec_kwargs):
    """Generate potential ECMs using EquivalentCircuits.jl in serial."""
    Main = julia_helpers.init_julia()
    # Suppress Julia warnings (coming from Optim.jl)
    Main.redirect_stderr()
    ec = julia_helpers.import_backend(Main)

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


def _generate_ecm_parallel(impedance, freq, iters, ec_kwargs):
    """Generate potential ECMs using EquivalentCircuits.jl in parallel."""

    def circuit_evolution(proc_id: int):
        """Closure to generate a single circuit to be used with multiprocessing."""
        Main = julia_helpers.init_julia()
        # Set a different random seed for each process (Python and Julia)
        np.random.seed(proc_id * time.time_ns() % 2**32)
        Main.eval(f"import Random; Random.seed!({proc_id}*time_ns())")
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

    with WorkerPool(n_jobs=nproc) as pool:
        try:
            circuits = pool.map(circuit_evolution, range(iters), **mpire_kwargs)
        except RuntimeError:
            runtime_error = True

    if runtime_error:
        raise RuntimeError("Julia must not be manually initialized, restart the kernel.")

    # Remove None values
    circuits = [circuit for circuit in circuits if circuit is not None]

    # Format circuits as a dataframe with columns "circuitstring" and "Parameters"
    df = pd.DataFrame(circuits, columns=["circuitstring", "Parameters"])

    return df


def split_components(df_circuits: pd.DataFrame) -> pd.DataFrame:
    """Split circuit components and their values for each ECM in the dataframe.

    Parameters
    ----------
    df_circuits: pd.DataFrame
        Dataframe containing ECMs (2 columns)

    Returns
    -------
    df_circuits: pd.DataFrame
        Dataframe containing ECMs (6 columns)
    """
    # Define some regular expression pattern to separate each kind of elements
    resistor_p = re.compile(r"[R][0-9][a-z]? = [0-9]*\.[0-9]*")
    capacitor_p = re.compile(r"[C][0-9][a-z]? = [0-9]*\.[0-9]*")
    inductor_p = re.compile(r"[L][0-9][a-z]? = [0-9]*\.[0-9]*")
    CPE_p = re.compile(r"[P][0-9][a-z]? = [0-9]*\.[0-9]*")

    # Initialize some lists to store the values of each kind of elements
    resistors_list = []
    capacitors_list = []
    inductors_list = []
    CPEs_list = []

    for i in range(len(df_circuits["Parameters"])):
        resistors = resistor_p.findall(df_circuits["Parameters"][i])
        capacitors = capacitor_p.findall(df_circuits["Parameters"][i])
        inductors = inductor_p.findall(df_circuits["Parameters"][i])
        CPEs = CPE_p.findall(df_circuits["Parameters"][i])

        resistors_list.append(resistors)
        capacitors_list.append(capacitors)
        inductors_list.append(inductors)
        CPEs_list.append(CPEs)

    df_circuits["Resistors"] = resistors_list
    df_circuits["Capacitors"] = capacitors_list
    df_circuits["Inductors"] = inductors_list
    df_circuits["CPEs"] = CPEs_list

    return df_circuits


def capacitance_filter(df_circuits: pd.DataFrame) -> pd.DataFrame:
    """Exclude ideal capacitors from the circuits dataframe.

    Parameters
    ----------
    df_circuits: pd.DataFrame
       Dataframe containing ECMs (6 columns)

    Returns
    -------
    df_circuits: pd.DataFrame
       Dataframe containing ECMs without ideal capacitors (6 columns)
    """
    for i in range(len(df_circuits["Capacitors"])):
        if df_circuits["Capacitors"][i] != []:
            df_circuits.drop([i], inplace=True)
    df_circuits.reset_index(drop=True, inplace=True)
    return df_circuits


def find_series_elements(circuit: str) -> str:
    """
    Extracts the series componenets from a circuit.

    Parameters
    ----------
    circuit: str
        String that stores the configuration of the circuit

    Returns
    -------
    series_circuit: str
        String that stores the series components of the circuit

    """
    series_circuit = []
    identifior = 0
    for i in range(len(circuit)):
        if circuit[i] == "[":
            identifior += 1
        if identifior == 0:
            series_circuit.append(circuit[i])
        if circuit[i] == "]":
            identifior -= 1
        # elif identifior != 0:
        #    index_list.append([False])
    series_circuit = "".join(series_circuit)
    return series_circuit


def ohmic_resistance_filter(df_circuits: pd.DataFrame, ohmic_resistance: float) -> pd.DataFrame:
    """Extracts the ohmic resistance of each circuit and filters those
    that are not within 15% of the ohmic resistance of the EIS data.

    Parameters
    ----------
    df_circuits: pd.DataFrame
       Dataframe containing ECMs (6 columns)
    ohmic_resistance: float
        The ohmic resistance of the given EIS data

    Returns
    -------
    df_circuits: pd.DataFrame
        Dataframe containing ECMs filtered based on ohmic resistance (6 columns)
    """
    for i in range(len(df_circuits["circuitstring"])):
        # Find the series elements
        series_circuit = find_series_elements(circuit=df_circuits["circuitstring"][i])
        # Find the series resistors
        find_R = re.compile(r"R[0-9]")
        series_resistors = find_R.findall(series_circuit)
        # Initiate a list to store series resistors' values for future comparison
        R_values_list = []
        for j in range(len(series_resistors)):
            value_R_p = re.compile(f"{series_resistors[j]} = [0-9]*\.[0-9]*")
            values_R_withid = value_R_p.findall("".join(df_circuits["Resistors"][i]))
            value_R_p2 = re.compile(r"[0-9]*\.[0-9]*")
            for k in range(len(values_R_withid)):
                R_value = value_R_p2.findall(values_R_withid[k])
                R_values_list.append(R_value)
        if R_values_list == []:
            df_circuits.drop([i], inplace=True)
        else:
            value_identify_list = []
            for m in range(len(R_values_list)):
                if (
                    float(R_values_list[m][0]) < ohmic_resistance * 0.85
                    or float(R_values_list[m][0]) > ohmic_resistance * 1.15
                ):
                    value_identify_list.append(False)
                else:
                    value_identify_list.append(True)
            if True not in value_identify_list:
                df_circuits.drop([i], inplace=True)

    df_circuits.reset_index(drop=True, inplace=True)
    return df_circuits


def series_filter(df_circuits: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the circuits by checking if any parallel route includes capacitors.

    Parameters
    ----------
    df_circuits: pd.DataFrame
        Dataframe containing the generated ECMs (6 columns)

    Returns
    -------
    df_circuits: pd.DataFrame
        Dataframe containing the generated ECMs without parallel capacitors (6 columns)

    """
    test_pattern = re.compile(r"\[")
    for i in range(len(df_circuits["circuitstring"])):
        test_circuit = df_circuits["circuitstring"][i]
        if test_pattern.findall(test_circuit) is False:
            df_circuits.drop([i], inplace=True)
    df_circuits.reset_index(drop=True, inplace=True)
    return df_circuits


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
                    # "X*(2*np.pi*F)**(-Y)*(np.cos((np.pi*Y)*0.5)-np.sin((np.pi*X)*0.5)*1j)",
                    "(1/(X*(2*1j*np.pi*F)**(Y)))" # modify the math function
                )

        new_temp_circuit = []
        counter = 0

        for n in range(len(circuit)):
            if circuit[n] == "X":
                new_temp_circuit.append(f"X[{str(counter)}]")
                counter += 1
            elif circuit[n] == "Y":
                new_temp_circuit.append(f"X[{str(counter)}]")
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


def combine_expression(df_circuits: "pd.DataFrame") -> "pd.DataFrame":
    """Combines the identical circuits.

    Parameters
    ----------
    df_circuits: pd.DataFrame
        Dataframe containing filtered ECMs with mathematical expressions (7 columns)

    Returns
    -------
    df_circuits: pd.DataFrame
        Dataframe containing filtered ECMs with mathematical expressions (r columns)
    """
    combined_expressions = []
    combined_values = []
    mathematical_expressions = []
    counts = []

    similar_expression, similar_expression_index = circuit_expression_combine_lists(
        df_circuits
    )
    component_values_lists, values_lists, names_lists = component_values(
        input=df_circuits["Parameters"]
    )

    for i in range(len(similar_expression_index)):
        combined_expressions.append(df_circuits["circuitstring"][similar_expression_index[i][0]])
        combined_value = []
        for j in range(len(similar_expression_index[i])):
            if j == 0:
                combined_value.append(component_values_lists[similar_expression_index[i][j]])
            else:
                if sorted(values_lists[similar_expression_index[i][j]]) != sorted(
                    values_lists[similar_expression_index[i][j - 1]]
                ):
                    combined_value.append(
                        component_values_lists[similar_expression_index[i][j]]
                    )
                    combined_expressions[i] = [
                        df_circuits["circuitstring"][similar_expression_index[i][0]]
                    ]
                    combined_expressions[i].append(
                        df_circuits["circuitstring"][similar_expression_index[i][j]]
                    )
                    # BUG: Correspond the combined circuit values with expressions

        if len(combined_value) > 1:
            # Calculate the statistical information about each component
            combined_component_value_list = []
            for k in range(len(combined_value)):
                combined_value_copy = combined_value.copy()
                combined_value_copy[k] = sorted(combined_value[k])
                digit_p = re.compile(r"\-?[0-9]+\.[0-9]+e*-*[0-9]*")
                combined_component_value = digit_p.findall(",".join(combined_value_copy[k]))
                for m in range(len(combined_component_value)):
                    combined_component_value[m] = float(combined_component_value[m])
                combined_component_value_list.append(combined_component_value)
            combined_component_value_array = np.array(combined_component_value_list)

            name_p = re.compile(r"[A-Z][1-9][a-z]? = ")
            combined_name = name_p.findall(",".join(df_circuits["Parameters"][1][0]))

            statistical_info = {
                "components_name": combined_name,
                "mean": np.mean(combined_component_value_array, axis=0),
                "std": np.std(combined_component_value_array, axis=0),
                "var": np.var(combined_component_value_array, axis=0),
                "max": np.max(combined_component_value_array, axis=0),
                "min": np.min(combined_component_value_array, axis=0),
            }
            combined_value.append(statistical_info)
        combined_values.append(combined_value)

        mathematical_expression = df_circuits["Mathematical expressions"][
            similar_expression_index[i][0]
        ]
        mathematical_expressions.append(mathematical_expression)

        count = len(similar_expression_index[i])
        counts.append(count)

    df_list = {
        "Combined Circuits": combined_expressions,
        "Combined Values": combined_values,
        "Mathematical expressions": mathematical_expressions,
        "Counts": counts,
    }
    df_circuits = pd.DataFrame(df_list)
    return df_circuits


def calculate_length(df_circuits: "pd.DataFrame") -> "pd.DataFrame":
    """Counts how many different value sets are in identical circuits.

    Parameters
    ----------
    df_circuits: pd.DataFrame
        Dataframe containing filtered ECMs with mathematical expressions (4 columns)

    Returns
    -------
    df_circuits: pd.DataFrame
        Dataframe containing filtered ECMs with mathematical expressions (5 columns)
    """

    counts = []
    for i in range(len(df_circuits["Combined Values"])):
        if len(df_circuits["Combined Values"][i]) > 1:
            count = len(df_circuits["Combined Values"][i]) - 1
            counts.append(count)
        else:
            count = 1
            counts.append(count)

    df_circuits["Different value sets"] = counts
    return df_circuits


def split_variables(df_circuits: "pd.DataFrame") -> "pd.DataFrame":
    """Separate the value and name of each component.

    Parameters
    ----------
    df_circuits: pd.DataFrame
        Dataframe containing filtered ECMs with mathematical expressions (5 columns)

    Returns
    -------
    df_circuits: pd.DataFrame
        Dataframe containing filtered ECMs with mathematical expressions (7 columns)
    """
    # Create some lists to store the names and values for BI
    variables_names = []
    variables_values = []

    # Create some RE pattern to split the values and names
    digit_p = re.compile(r"\-?[0-9]+\.[0-9]+e*-*[0-9]*")
    name_p = re.compile(r"[A-Z][0-9]+[a-z]?")

    for i in range(len(df_circuits["Combined Values"])):
        variables_name = []
        variables_value = []
        if len(df_circuits["Combined Values"][i]) == 1:
            for k in range(len(df_circuits["Combined Values"][i][0])):
                variable = df_circuits["Combined Values"][i][0][k]
                variables_name.append(name_p.findall(variable)[0])
                variables_value.append(float(digit_p.findall(variable)[0]))
        elif len(df_circuits["Combined Values"][i]) != 1:
            for k in range(len(df_circuits["Combined Values"][i][0])):
                variable = df_circuits["Combined Values"][i][0][k]
                variables_name.append(name_p.findall(variable)[0])
                variables_value.append(float(digit_p.findall(variable)[0]))
        variables_names.append(variables_name)
        variables_values.append(variables_value)

    df_circuits["Variables_names"] = variables_names
    df_circuits["Variables_values"] = variables_values

    df_circuits = df_circuits.reset_index()
    df_circuits.drop(["index"], axis=1, inplace=True)

    return df_circuits


def temperate_filter(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    TODO: This is just a temporary filtering rule to delete ECMs with 0.0
    value after approximation with 4 digits.

    Parameter:
    ----------
    df: pd.DataFrame
        Dataframe containing filtered ECMs (7 columns)

    Return:
    -------
    df: pd.DataFrame
        Dataframe containing filtered ECMs (7 columns)
    """
    for i in range(len(df["Combined Circuits"])):
        value_set = df["Variables_values"][i]
        for j in range(len(value_set)):
            if round(value_set[j], 12) == 0:
                df = df.drop([i], axis=0)
                break
    df = df.reset_index()
    df = df.drop(["index"], axis=1)
    return df


def r2_calculator(y_true, y_pred):
    """Calculate of the coefficient of determintion, R^2.

    Parameters
    ----------
    y_true: np.ndarray
        Ndarray that stores the ground truth values
    y_pred: np.ndarray
        Ndarray that stores the predictive values

    Returns
    -------
    r2: float
        The coefficient of determintion
    """
    sse = (abs(y_pred - y_true) ** 2).sum()
    sst = (abs(y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - sse / sst
    return r2


def MSE_calculator(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate of the mean square error.

    Parameters
    ----------
    y_true: np.ndarray
        Array containing the ground truth values
    y_pred: np.ndarray
        Array containing the predictive values

    Returns
    -------
    MSE: float
        The mean square error
    """
    mse = np.array(((abs(y_true - y_pred)) ** 2)).mean()
    return mse


def RMSE_calculator(y_true: np.array, y_pred: np.array) -> float:
    """Calculate of the root mean square error.

    Parameters
    ----------
    y_true: np.ndarray
        Array containing the ground truth values
    y_pred: np.ndarray
        Array containing the predictive values

    Return:
    -------
    RMSE: float
        The root mean square error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = (
        sum((y_pred.real - y_true.real) ** 2 + (y_pred.imag - y_true.imag) ** 2) / len(y_true)
    ) ** (1 / 2)
    return rmse


def MAPE_calculator(y_true: np.array, y_pred: np.array) -> float:
    """Calculate of the mean abosulte percentage error.

    Parameters
    ----------
    y_true: np.ndarray
        Array containing the ground truth values
    y_pred: np.ndarray
        Array containing the predictive values

    Returns
    -------
    mean_absolute_error_percentage: float
        The mean absolute percentage error
    """
    error_term = abs(y_true - y_pred)
    absolute_error_percentage = error_term / abs(y_true)
    mean_absolute_error_percentage = (100 / len(y_true)) * sum(absolute_error_percentage)
    return mean_absolute_error_percentage


def posterior_evaluation(posteriors):
    """Evaluate the posterior distributions according to their shapes.

    Parameters
    ----------
    posteriors: Axesubplots
        The axesubplots that record posterior distribution

    Returns
    -------
    marker: float
        Indicator of the quality of posterior distributions
    """
    marker = 0
    posterior_x = []
    posterior_y = []
    for i in range(len(posteriors) - 1):
        test_dist = posteriors[i][0].lines[0]
        test_x, test_y = test_dist.get_xydata().T
        test_y_percent = test_y / sum(test_y)

        posterior_x.append(test_x)
        posterior_y.append(test_y_percent)

        if (
            np.where(test_y_percent == test_y_percent.max())[0][0] == 0
            or np.where(test_y_percent == test_y_percent.max())[0][0] == 511
            or test_y_percent.max() >= 0.01
            or test_y_percent.max() <= 0.003
        ):
            marker = marker - 1
        else:
            if (
                test_y_percent[np.where(test_x == test_x.max())[0][0]] >= 0.001
                or test_y_percent[np.where(test_x == test_x.max())[0][0]] >= 0.001
            ):
                marker = marker - 0.5

    return marker


def model_evaluation(results):
    # TODO: Relying on the column index is not a good idea, refactor.
    evaluation_results = results[results.columns[[0, 17, 18, 19, 20, 23, 25]]]

    # FIXME: Remove next line once confirmed that `loc` is correctly used.
    # evaluation_results["Consistency"] = pd.to_numeric(evaluation_results["Consistency"], errors="coerce")
    evaluation_results.loc[:, "Consistency"] = pd.to_numeric(evaluation_results["Consistency"], errors="coerce")
    evaluation_results.loc[evaluation_results["Consistency"].isna(), "Consistency"] = np.inf

    def absdiff(x):
        return np.inf if np.isinf(x) else np.abs(x - 1)

    def custom_sort(x):
        return -1000 if x == "F" else x

    # FIXME: Remove next line once confirmed that `loc` is correctly used.
    # evaluation_results["Consistency"] = evaluation_results["Consistency"].apply(absdiff)
    evaluation_results.loc[:, "Consistency"] = evaluation_results["Consistency"].apply(absdiff)    
    # FIXME: Remove next line once confirmed that `loc` is correctly used.
    # evaluation_results["Posterior_shape"] = evaluation_results["Posterior_shape"].apply(custom_sort)
    evaluation_results.loc[:, "Posterior_shape"] = evaluation_results["Posterior_shape"].apply(custom_sort)    

    evaluation_results_sorted = evaluation_results.sort_values(
        by=[
            "Divergences",
            "Posterior_shape",
            "Consistency",
            "Posterior_mean_r2_real",
            "Posterior_mean_r2_imag",
            "AIC Value",
        ],
        ascending=[True, False, True, False, False, True],
    )
    results_sorted = results.loc[evaluation_results_sorted.reset_index()["index"]]
    results_sorted = results_sorted.reset_index(drop=True)
    return results_sorted


def perform_bayesian_inference(
    eis_data: pd.DataFrame,
    ecms: pd.DataFrame,
    saveto: str = None,
    plot: bool = False,
    draw_ecm=False,
) -> pd.DataFrame:
    """Perform Bayesian inference on the ECMs based on the EIS measurements.

    Parameters
    ----------
    eis_data : pd.DataFrame
        DataFrame with pre-processed data; expected columns are frequency,
        real part, and imaginary part of the impedance data.
    ecms : pd.DataFrame
        DataFrame with filtered ECMs.
    plot : bool, optional
        If True, plots the results (default is True).
    saveto : str, optional
        Path to the directory where the results will be saved (default is None).
    draw_ecm : bool, optional
        If True, draws the circuit model (default is False).

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the ECMs with the Bayesian inference results (12 columns)
    """
    log.info("Applying Bayesian inference on the circuits.")

    # Determine if there's any ECM that passed post-filtering process
    if len(ecms) == 0:
        raise Exception("Circuits' dataframe is empty!")

    freq = eis_data["freq"].to_numpy()
    Zreal = eis_data["Zreal"].to_numpy()
    Zimag = eis_data["Zimag"].to_numpy()
    Z = Zreal + 1j * Zimag

    # ?: Do we need this?
    amplifying_factor = abs(Zreal.max() - Zreal.min()) / abs(Zimag.max() - Zimag.min())
    relative_error_accepted = (((Zreal**2) + (Zimag**2)) ** (1 / 2)).mean()

    # Create a list to store the R2 value of each ECM
    R2_list = []
    R2_real_list = []
    R2_imag_list = []
    # Create a list to store fitting quality metrics of each ECM
    MSE_list = []
    RMSE_list = []
    MAPE_list = []
    # Create a list to store simulated ECM data
    ECMs_data = []
    # Create a list to store mean r2 in posteior distribution
    Posterior_r2 = []
    Posterior_r2_real = []
    Posterior_r2_imag = []
    # Create a list to store mean mse in posteior distribution
    Posterior_mape = []
    Posterior_mape_real = []
    Posterior_mape_imag = []
    # Create lists to store BI results
    models = []
    models_descriptions = []
    traces = []
    Prior_predictions = []
    Posterior_predictions = []
    AIC = []
    # Create a set of lists for model evaluation
    divergences = []
    posterior_shape = []
    consistency = []

    # Start from this source of randomness. We will split keys for subsequent operations.
    # FIXME: Figure out a more permanent solution for RNG
    rng_key = random.PRNGKey(time.time_ns())
    rng_key, rng_key_ = random.split(rng_key)

    # BI parts
    values = ecms["Variables_values"]
    names = ecms["Variables_names"]
    expressions_strs = ecms["Mathematical expressions"]
    circuit_names = ecms["Combined Circuits"]

    for i in tqdm(range(len(ecms["Combined Circuits"]))):
        circuit_name_i = circuit_names[i]
        value_i = values[i]
        name_i = names[i]
        expression_str_i = expressions_strs[i].replace("np.", "jnp.")
        function_i = eval(f"lambda X,F:{expression_str_i}")

        log.info(f"Circuit {i}: {circuit_name_i}")
        log.info(f"Elements: ({name_i})\nValues: ({value_i})")

        if plot and draw_ecm:
            viz.draw_circuit(circuit_name_i)

        Zsim = function_i(value_i, freq)
        ECMs_data.append(Zsim)

        log.info("Julia circuit's fitting")

        r2_value = float(r2_calculator(Zreal + 1j * Zimag, Zsim))
        log.info(f"R² = {r2_value}")
        R2_list.append(r2_value)

        r2_real = r2_calculator(Zreal, Zsim.real)
        log.info(f"R² (Re) = {r2_real}")
        R2_real_list.append(r2_real)
        r2_imag = r2_calculator(Zimag, Zsim.imag)
        log.info(f"R² (Im) = {r2_imag}")
        R2_imag_list.append(r2_imag)

        MSE_value = float(MSE_calculator(Zreal + 1j * Zimag, Zsim))
        log.info(f"MSW:{MSE_value}")
        MSE_list.append(MSE_value)

        RMSE_value = float(MSE_calculator(Zreal + 1j * Zimag, Zsim) ** (1 / 2))
        log.info(f"RMSE:{RMSE_value}")
        RMSE_list.append(RMSE_value)

        MAPE_value = float(MAPE_calculator(Zreal + 1j * Zimag, Zsim) ** (1 / 2))
        log.info(f"MAPE:{MAPE_value}")
        MAPE_list.append(MAPE_value)

        if plot:
            fig, ax = plt.subplots()
            viz.plot_nyquist(Z=Zsim, fmt="o", color="r", label="simulated", ax=ax)
            viz.plot_nyquist(Z=Z, fmt="-", color="b", label="experiment", ax=ax)
            if saveto is not None:
                fpath = os.path.join(saveto, f"nyquist_simulated_{i}.png")
                fig.savefig(fpath, dpi=300)

        def model_i(
            values=value_i,
            func=function_i,
            true_data=eis_data,
            error=relative_error_accepted
        ):
            true_freq = np.asarray(true_data["freq"])
            true_Zreal = np.asarray(true_data["Zreal"])
            true_Zimag = np.asarray(true_data["Zimag"])

            variables_list = []
            for j in range(len(name_i)):
                name = name_i[j]
                value = value_i[j]
                if "n" in name:
                    free_variable = numpyro.sample(f"{name}", dist.Uniform(0, 1))
                    variables_list.append(free_variable)
                else:
                    free_variable = numpyro.sample(f"{name}", dist.LogNormal(2.5, 1.7))
                    real_variable = value * free_variable
                    variables_list.append(real_variable)

            true_obs = true_Zreal + true_Zimag * 1j
            mu = function_i(variables_list, true_freq)
            error_term = numpyro.sample("err", dist.HalfNormal())
            numpyro.sample("obs", dist.HalfNormal(error_term), obs=abs(true_obs - mu))

        # ?: Why 200?
        prior_predictive = Predictive(model_i, num_samples=200)
        prior_prediction = prior_predictive(rng_key)
        rng_key, rng_key_ = random.split(rng_key)
        Prior_predictions.append(prior_prediction)

        # ?: Why 10,000?
        kernel = NUTS(model_i, target_accept_prob=0.8)
        num_samples = 10000
        mcmc_i = MCMC(kernel, num_warmup=1000, num_samples=num_samples, num_chains=1)
        mcmc_i.run(
            rng_key,
            values=value_i,
            func=function_i,
            true_data=eis_data,
            error=relative_error_accepted,
        )

        # Results
        models.append(mcmc_i)
        models_descriptions.append(mcmc_i.print_summary)

        trace = az.convert_to_inference_data(mcmc_i)
        traces.append(trace)

        # Export MCMC results to netcdf
        if saveto is not None:
            fpath = os.path.join(saveto, f"mcmc_circuit_{i}.nc")
            trace.to_netcdf(fpath)

        # Calculate AIC
        # FIXME: Remove next line once confirmed that `iloc` is correctly used.
        # AIC_value = az.waic(mcmc_i)[0] * (-2) + 2 * len(name_i)
        AIC_value = az.waic(mcmc_i).iloc[0] * (-2) + 2 * len(name_i)
        AIC.append(AIC_value)
        log.info(f"AIC value = {AIC_value}")

        divergence = np.asarray(mcmc_i.get_extra_fields()["diverging"].sum()).ravel()[0]
        divergences.append(divergence)

        # Prior distributions
        if plot:
            ax = az.plot_trace(prior_prediction, var_names=name_i)
            if saveto is not None:
                ax.figure.savefig(f"prior_distributions_{i}.png", dpi=300)

        # Prior predictions
        if plot:
            fig, ax = plt.subplots()
        # ?: Unused variable?
        prior_R2_list = []
        # ?: Why 100?
        for j in range(100):
            vars = []
            for k in range(len(name_i)):
                if "n" in name_i[k]:
                    var = prior_prediction[name_i[k]][j]
                    vars.append(var)
                else:
                    var = prior_prediction[name_i[k]][j] * value_i[k]
                    vars.append(var)
            y = function_i(vars, freq)
            if plot:
                ax.plot(y.real, -y.imag, color="k", alpha=0.4)
        if plot:
            ax.plot(Zreal, -Zimag, c="b", alpha=1)
            ax.set_xlabel("Re(Z)")
            ax.set_ylabel("Im(Z)")
            ax.set_title("Prior predictive checks")
            if saveto is not None:
                fig.savefig(f"prior_predictions_{i}.png", dpi=300)

        # Posterior distributions
        if plot:
            for i in range(len(name_i)):
                name = name_i[i]
                value = value_i[i]
                if "n" not in name:
                    trace.posterior[name] = trace.posterior[name] * value
            posterior_HDI = az.plot_posterior(trace, var_names=name_i)
            # ?: What's this for loop for?
            #             for i in range(posterior_HDI.shape[0]):
            #                 for j in range(posterior_HDI.shape[1]):
            #                     rc_id = i*3 + j
            #                     if rc_id < len(value_i):
            #                         y_values = posterior_HDI[i][j].lines[0].get_ydata()
            #                         posterior_HDI[i][j].lines[0].set_data(np.multiply(posterior_HDI[i][j].lines[0].get_xydata()[:,0],value_i[rc_id]),y_values)
            #                         new_lim = np.multiply(posterior_HDI[i][j].get_xlim(),value_i[rc_id])
            #                         posterior_HDI[i][j].set_xlim(new_lim)
            if saveto is not None:
                posterior_HDI.figure.savefig(f"posterior_predictions_with_HDI_{i}.png", dpi=300)

        # Posterior trajectories
        posterior_dist = az.plot_trace(trace, var_names=name_i)

        if plot:
            if saveto is not None:
                posterior_dist.figure.savefig(f"posterior_distributions_{i}.png", dpi=300)

        # Posterior predictions, real part
        if plot:
            fig, ax = plt.subplots()
        samples = mcmc_i.get_samples()
        Posterior_predictions.append(samples)
        sep_mape_real_list = []
        sep_r2_real_list = []
        # ?: Why 100?
        for j in range(100):
            vars = []
            for k in range(len(name_i)):
                if "n" in name_i[k]:
                    var = samples[name_i[k]][j]
                    vars.append(var)
                else:
                    var = samples[name_i[k]][j] * value_i[k]
                    vars.append(var)
            BI_data = function_i(vars, freq)
            if plot:
                ax.plot(freq, BI_data.real, marker=".", color="grey", alpha=0.5)
                ax.set_xscale("log")
            sep_mape_real = float(MAPE_calculator(Zreal, BI_data.real))
            sep_mape_real_list.append(sep_mape_real)
            sep_r2_real = float(r2_calculator(Zreal, BI_data.real))
            sep_r2_real_list.append(sep_r2_real)

        avg_mape_real = np.array(sep_mape_real_list).mean()
        avg_r2_real = np.array(sep_r2_real_list).mean()
        log.info(f"Posterior fit (real): MAPE = {avg_mape_real}, R² = {avg_r2_real}")
        Posterior_r2_real.append(avg_r2_real)
        Posterior_mape_real.append(avg_mape_real)

        if plot:
            ax.plot(freq, BI_data.real, marker=".", ms=15, color="grey", alpha=0.5, label="predictive")
            ax.plot(freq, Zreal, "--", marker="o", c="b", alpha=0.9, ms=8, label="ground truth")
            ax.set_xscale("log")
            ax.set_xlabel("frequency")
            ax.set_ylabel("Re(Z)")
            ax.set_title("Posterior predictive checks (real)")
            ax.legend()
            if saveto is not None:
                fig.savefig("posterior_predictions_real.png", dpi=300)

        # Posterior predictions, imaginary part
        if plot:
            fig, ax = plt.subplots()
        sep_mape_imag_list = []
        sep_r2_imag_list = []
        # ?: Why 100?
        for j in range(100):
            vars = []
            for k in range(len(name_i)):
                if "n" in name_i[k]:
                    var = samples[name_i[k]][j]
                    vars.append(var)
                else:
                    var = samples[name_i[k]][j] * value_i[k]
                    vars.append(var)
            BI_data = function_i(vars, freq)
            if plot:
                ax.plot(freq, -BI_data.imag, marker=".", color="grey", alpha=0.5)
                ax.set_xscale("log")
            sep_mape_imag = float(MAPE_calculator(Zimag, BI_data.imag))
            sep_mape_imag_list.append(sep_mape_imag)
            sep_r2_imag = float(r2_calculator(Zimag, BI_data.imag))
            sep_r2_imag_list.append(sep_r2_imag)

        avg_mape_imag = np.array(sep_mape_imag_list).mean()
        avg_r2_imag = np.array(sep_r2_imag_list).mean()
        log.info(f"Posterior fit (imag): MAPE = {avg_mape_imag}, R² = {avg_r2_imag}")
        Posterior_r2_imag.append(avg_r2_imag)
        Posterior_mape_imag.append(avg_mape_imag)
        if plot:
            ax.plot(freq, -BI_data.imag, marker=".", ms=15, color="grey", alpha=0.5, label="predictive")
            ax.plot(freq, -Zimag, "--", marker="o", c="b", alpha=0.9, ms=8, label="ground truth")
            ax.set_xscale("log")
            ax.set_xlabel("frequency")
            ax.set_ylabel("-Im(Z)")
            ax.set_title("Posterior predictive checks (imag)")
            ax.legend()
            if saveto is not None:
                fig.savefig("posterior_predictions_imag.png", dpi=300)

        # Posterior predictions
        if plot:
            fig, ax = plt.subplots()
        sep_mape_list = []
        sep_r2_list = []
        # ?: Why 100?
        for j in range(100):
            vars = []
            for k in range(len(name_i)):
                if "n" in name_i[k]:
                    var = samples[name_i[k]][j]
                    vars.append(var)
                else:
                    var = samples[name_i[k]][j] * value_i[k]
                    vars.append(var)
            BI_data = function_i(vars, freq)
            if plot:
                ax.plot(BI_data.real, -BI_data.imag, color="grey", marker=".", alpha=0.5)
            sep_mape = float(MAPE_calculator(Zreal + 1j * Zimag, BI_data))
            sep_mape_list.append(sep_mape)
            sep_r2 = float(r2_calculator(Zreal + 1j * Zimag, BI_data))
            sep_r2_list.append(sep_r2)

        # ?: Why commented out?
        # avg_mse = np.array(sep_mse_list).mean()
        avg_mape = np.array(sep_mape_list).mean()
        avg_r2 = np.array(sep_r2_list).mean()
        log.info(f"Posterior fit: MAPE = {avg_mape}, R² = {avg_r2}")
        Posterior_r2.append(avg_r2)
        Posterior_mape.append(avg_mape)

        if plot:
            ax.plot(BI_data.real, -BI_data.imag, marker=".", ms=15, color="grey", alpha=0.5, label="predictions")
            ax.plot(Zreal, -Zimag, "--", marker="o", c="b", alpha=0.9, ms=8, label="grount truth")
            ax.set_xlabel("Re(Z)")
            ax.set_ylabel("Im(Z)")
            ax.set_title("Posterior predictive checks")
            ax.legend(loc="upper left")
            if saveto is not None:
                fig.savefig(f"posterior_predictions_{i}.png", dpi=300)

        # Pair relationship
        if plot:
            ax = az.plot_pair(mcmc_i, var_names=name_i)
            if saveto is not None:
                ax.figure.savefig(f"pair_relationship_plot_{i}.png", dpi=300)

        # Estimate posterior distribution
        if any(len(result[0].lines[0].get_xydata().T[0]) == 2 for result in posterior_dist[:]):
            posterior_mark = "F"
        else:
            posterior_mark = posterior_evaluation(posterior_dist)
        posterior_shape.append(posterior_mark)

        r_hats = []
        for i in range(len(name_i)):
            r_hats.append(
                summary(
                    mcmc_i.get_samples(),
                    prob=0.94,
                    group_by_chain=False
                )[f"{name_i[i]}"]["r_hat"]
            )
        posterior_rhat = np.mean(r_hats)
        consistency.append(posterior_rhat)

    ecms["ECM Data"] = ECMs_data
    ecms["R_square"] = R2_list
    ecms["Mean Square Error"] = MSE_list
    ecms["Mean Absolute Percentage Error"] = MAPE_list
    ecms["Root Mean Square Error"] = RMSE_list
    ecms["BI_models"] = models
    ecms["Traces"] = traces
    ecms["BI_models_description"] = models_descriptions
    ecms["Priors_prediction"] = Prior_predictions
    ecms["Posterior_prediction"] = Posterior_predictions
    ecms["AIC Value"] = AIC
    ecms["Divergences"] = divergences
    ecms["Consistency"] = consistency
    ecms["Posterior_shape"] = posterior_shape
    ecms["Posterior_mean_r2"] = Posterior_r2
    ecms["Posterior_mean_mape"] = Posterior_mape
    ecms["Posterior_mean_r2_real"] = Posterior_r2_real
    ecms["Posterior_mean_mape_real"] = Posterior_mape_real
    ecms["Posterior_mean_r2_imag"] = Posterior_r2_imag
    ecms["Posterior_mean_mape_imag"] = Posterior_mape_imag

    ecms = model_evaluation(ecms)

    # Export the results to pickle
    if saveto is not None:
        _saveto = os.path.join(saveto, "results.pkl")
        with open(_saveto, "wb") as f:
            dill.dump(ecms.to_dict(), f)

    return ecms


def apply_heuristic_rules(circuits: pd.DataFrame, ohmic_resistance) -> pd.DataFrame:
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

    # Make a copy to avoid modifying the original dataframe
    circuits = circuits.copy()

    if len(circuits) == 0:
        log.warning("Circuits' dataframe is empty!")
        return circuits
    
    circuits = split_components(circuits)
    circuits = capacitance_filter(circuits)
    circuits = series_filter(circuits)
    circuits = ohmic_resistance_filter(circuits, ohmic_resistance)
    circuits = generate_mathematical_expression(circuits)
    circuits = combine_expression(circuits)
    circuits = calculate_length(circuits)
    circuits = split_variables(circuits)

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

    # Preprocessing + store preprocessed data
    kwargs = {"threshold": 0.05, "saveto": saveto, "plot": plot}
    eis_data, ohmic_resistance, rmse = preprocess_impedance_data(impedance, freq, **kwargs)
    Z_clean = eis_data["Zreal"].to_numpy() + 1j * eis_data["Zimag"].to_numpy()
    freq_clean = eis_data["freq"].to_numpy()
    
    # Generate a pool of potential ECMs via an evolutionary algorithm
    kwargs = {"iters": iters, "complexity": 12, "tol": 1e-1, "saveto": saveto, "parallel": parallel}
    circuits_unfiltered = generate_equivalent_circuits(Z_clean, freq_clean, **kwargs)

    # Apply heuristic rules to filter unphysical circuits
    circuits = apply_heuristic_rules(circuits_unfiltered, ohmic_resistance)

    # Perform Bayesian inference on the filtered ECMs
    kwargs = {"data_path": saveto, "plot": plot, "draw_ecm": draw_ecm}
    results = perform_bayesian_inference(eis_data, circuits, **kwargs)

    return results

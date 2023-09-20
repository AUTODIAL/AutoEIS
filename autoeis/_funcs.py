import itertools
import json
import logging
import math
import os
import pickle
import re
import sys
import warnings
from os.path import basename

import arviz as az
import dill
import impedance
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from impedance import preprocessing
from impedance.validation import linKK
from IPython.display import set_matplotlib_formats
from jax import random, vmap
from jax.scipy.special import logsumexp
from julia import Julia
from numpyro import handlers
from numpyro.diagnostics import hpdi
from numpyro.infer import (
    HMC,
    HMCECS,
    MCMC,
    NUTS,
    BarkerMH,
    DiscreteHMCGibbs,
    HMCGibbs,
    Predictive,
)
from numpyro.infer.util import log_density, log_likelihood

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def set_julia(julia_exec_path: str):
    """
    Set the julia environment in python

    Parameters
    ----------
    julia_executable_path:str
        Path of Julia executable in your computer

    Returns
    -------
    Handle to Julia runtime

    """
    return Julia(runtime=julia_exec_path, compiled_modules=False)


def initialize_julia():
    """Installs Julia's dependencies"""
    from julia import Base, Pkg

    # Install the required Julia packages
    Base.ENV["PYTHON"] = sys.executable
    Pkg.build("PyCall")
    Pkg.add("EquivalentCircuits")
    Pkg.add("DelimitedFiles")
    Pkg.add("StringEncodings")
    Pkg.add("Pandas")
    Pkg.add("DataFrames")


def import_julia():
    """Imports the required packages in Julia and returns the handles."""
    from julia import Base, DataFrames, DelimitedFiles
    from julia import EquivalentCircuits as ec
    from julia import Pandas, Pkg, StringEncodings

    return ec, DataFrames, Pandas, Base


def set_parameter():
    """Modifies the default plotting parameters for matplotlib."""
    plt.rcParams["figure.figsize"] = (19.6, 10.8)
    az.style.use("arviz-darkgrid")


def mkdir(path):
    """Create a new folder in the current working directory."""
    if not os.path.exists(path):
        os.makedirs(path)


def reset_storage():
    """Resets the storage path to the current working directory."""
    for _ in range(2):
        os.chdir(os.path.abspath(os.path.dirname(os.getcwd())))


def load_data(fname: str) -> "pd.DataFrame":
    """
    Loads the EIS data(impedance and frequency) from file.

    Parameters
    ----------
    fname: str
        Path of the EIS data file

    Returns
    --------
    df: pd.DataFrame
        Dataframe containing the impedance and frequency data

    """
    try:
        f = open(fname)
    except FileNotFoundError:
        log.error("No such file/directory in the current folder.")
        return
    if fname.endswith(".json"):
        data = json.load(f)
        df = pd.DataFrame(data)
    elif fname.endswith(".csv"):
        df = pd.read_csv(f)
    elif fname.endswith(".txt"):
        df = pd.read_csv(fname, sep="\t")
    elif fname.endswith(".xlsx"):
        df = pd.read_excel(fname)
    elif fname.endswith(".pkl"):
        with open(fname, "rb") as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
    try:
        return df
    except UnboundLocalError:
        log.error("Unsupported file format.")


def plot_EIS(
    freq: "np.array",
    impedance: "np.array" = "",
    reals: "np.array" = "",
    imags: "np.arary" = "",
    kind: str = "Nyquist",
):
    """
    Plots Nyquist and Bode plots of the impedance data.

    Parameters
    ----------
    freq: np.ndarray[float]
        The frequencies of EIS data points
    impedance: np.ndarray[complex]
        The impedance data
    reals: np.ndarray[float]
        The real part of the impedance data
    imags: np.ndarray[float]
        The imag part of the impedance data
    kind: str
        The kind of plots, can be either Nyquist or Bode

    Returns
    -------
    fig: matplotlib.figure.Figure

    """

    # Nyquist plot
    if kind == "Nyquist":
        if impedance != "":
            reals = impedance.real
            imags = impedance.imag
        frequencies = freq

        fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), dpi=300)

        axes[0].scatter(reals, -imags, s=1.5)
        axes[0].set_xlabel(r"$Re(Z) / \Omega$")
        axes[0].set_ylabel(r"$-Im(Z) / \Omega$")
        axes[0].set_title("Nyquist plot")

        axes[1].scatter(freq, reals, s=1.5)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("freq (Hz)")
        axes[1].set_ylabel(r"$Re(Z) / \Omega$")
        axes[1].set_title("Bode plot - real part")

        axes[2].scatter(freq, -imags, s=1.5)
        axes[2].set_xscale("log")
        axes[2].set_ylabel(r"$-Im(Z) / \Omega$")
        axes[2].set_xlabel("freq (Hz)")
        axes[2].set_title("Bode plot - imaginary part")
        plt.show()

    # Bode plot
    elif kind == "Bode":
        if impedance != "":
            reals = impedance.real
            imags = impedance.imag
        frequencies = freq

        # Calculate the magnitude/phase of the impedance data
        magnitude = (reals**2 + imags**2) ** (1 / 2)
        phase = np.arctan(imags / reals)

        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)

        # Magnitude plot
        ax.scatter(np.log10(frequencies), magnitude, c="b", label="Magnitude")
        ax.set_xlabel("log(freq)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Bode plot")
        # Phase plot
        ax2 = ax.twinx()
        ax2.scatter(np.log10(frequencies), phase, c="r", alpha=0.7, marker="v", label="phase")
        ax2.set_ylabel("Phase")
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        fig.show()

    return fig


def find_ohmic_resistance(
    reals: "np.array", imags: "np.array", fname: str, save: bool = True
) -> "float":
    """
    Extracts the ohmic resistance of impedance data by doing a 5th order
    polynomial fit (with 5% tolerance as relative error).

    Parameters:
    ----------
    reals: np.ndarray[float]
        The real part of the impedance data
    imags: np.ndarray[float]
        The imag part of the impedance data
    fname: str
        The storage path

    Return:
    -------
    ohmic_resistance: float
        The ohmic resistance of impedance data

    """
    # Select the high-frequency impedance data
    high_f_real = reals[0:10]
    high_f_imag = imags[0:10]
    high_f_phase = np.arctan(high_f_imag / high_f_real)
    # Find minimum phase value, note returns as a tuple
    (index,) = np.where(abs(high_f_phase) == abs(high_f_phase).min())
    # Extract the ohmic resistance
    ohmic_resistance = high_f_real[index[0]]

    if save:
        dirStr, ext = os.path.splitext(fname)
        folder_name = dirStr.split("\\")[-1]
        mkdir(folder_name)
        with open(
            f"{folder_name}\\ohmic_resistance = {round(ohmic_resistance, 4)}.txt", "w"
        ) as f:
            f.write(f"ohmic resistance = {ohmic_resistance}")

    return ohmic_resistance


def pre_processing(
    impedance: "np.ndarray", freq: "np.ndarray", threshold: float, fname: str
) -> "Returns: pd.DataFrame / float / float":
    """Pre-process impedance data by deleting data with positive imaginary part at high-freq range, and by kk validation

    Parameters
    ----------
    impedance: np.ndarray[complex]
        The impedance data
    freq: np.ndarray[float]
        The frequencies of EIS data points
    threshold: float
        The parameter that controls the filtering effect of KK validation
    fname: str
        The storage path

    Returns
    -------
    Zdf_mask: pd.DataFrame
        The impedance data after pre-processing
    ohmic_resistance: float
        The ohmic resistance of impedance data
    RMSE_value:float
        The mean square error between KK validated data and original data

    """
    # Set plotting parameters and export path
    tick_size = 9
    label_size = 11

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["xtick.labelsize"] = tick_size
    plt.rcParams["ytick.labelsize"] = tick_size
    plt.rcParams["axes.labelsize"] = label_size
    plt.rcParams["legend.fontsize"] = tick_size - 1

    # Set the storage folder
    dirStr, ext = os.path.splitext(fname)
    folder_name = dirStr.split("\\")[-1]
    mkdir(folder_name)

    # Load the data
    Re_Z = impedance.real
    Im_Z = impedance.imag
    # Set plotting parameters for the non-filtered data
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), dpi=300)

    # Plot the non-filtered plots
    axes[0].scatter(Re_Z, -Im_Z, s=1.5)
    axes[0].set_xlabel(r"$Re(Z) / \Omega$")
    axes[0].set_ylabel(r"$-Im(Z) / \Omega$")
    axes[0].set_title("Non-filtered")
    axes[1].scatter(freq, Re_Z, s=1.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("freq (Hz)")
    axes[1].set_ylabel(r"$Re(Z) / \Omega$")
    axes[1].set_title("Non-filtered")
    axes[2].scatter(freq, -Im_Z, s=1.5)
    axes[2].set_xscale("log")
    axes[2].set_ylabel(r"$-Im(Z) / \Omega$")
    axes[2].set_xlabel("freq (Hz)")
    axes[2].set_title("Non-filtered")
    plt.savefig(f"{folder_name}\\Non-filtered_Nyquist and Bode plots.png", dpi=300)
    plt.show()

    # Filter 1 - High Frequency Region
    # Find index where phase_Zwe == minimum, remove all high frequency imag values below zero
    # Find index: 10khz - 100khz
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
    RMSE_value = RMSE_calculator(Z, Z_linKK)

    # Plot residuals of Lin-KK for visualization purposes.
    plt.plot(freq, res_imag, label="delta Im")
    plt.plot(freq, res_real, label="delta Re")
    plt.xlabel("freq (Hz)")
    plt.ylabel("Delta %")
    plt.xscale("log")
    plt.title("Lin-KK Validation")
    plt.legend()
    plt.savefig(f"{folder_name}\\Lin-KK Validation.png", dpi=300)

    # Need to set a threshold limit for when to filter out the noisy data
    # of the residuals threshold = 0.05 # USER DEFINE!!!

    # NOTE: 2023/05/03 modification by Runze Zhang
    Zdf_mask = np.arange(1)

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
            ohmic_resistance = find_ohmic_resistance(Re_Z_mask, Im_Z_mask, fname)
        except ValueError:
            msg = "Ohmic resistance not found. Recheck data or increase KK threshold."
            log.error(msg)
        # Putting into a dataframe for use with plotting and program
        values_mask = np.array([freq_mask, Re_Z_mask, Im_Z_mask])
        labels = ["freq", "Zreal", "Zimag"]
        Zdf_mask = pd.DataFrame(values_mask.transpose(), columns=labels)
        threshold += 0.01

    print(f"ohmic_resistance = {ohmic_resistance}")

    # Plot the data to see how the filter performed
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), dpi=300)
    axes[0].scatter(Zdf_mask["Zreal"].values, -Zdf_mask["Zimag"].values, s=1.5)
    # if index == 0:
    #    axes[0].scatter(Re_Z_mask, fz(Re_Z_mask),s=1.5,c='r')
    axes[0].set_xlabel(r"$Re(Z) / \Omega$")
    axes[0].set_ylabel(r"$-Im(Z) / \Omega$")
    axes[0].set_title("Nyquist Plot")
    axes[1].scatter(Zdf_mask["freq"].values, Zdf_mask["Zreal"].values, s=1.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("freq (Hz)")
    axes[1].set_ylabel(r"$Re(Z) / \Omega$")
    axes[1].set_title("Bode plot - Real part")
    axes[2].scatter(Zdf_mask["freq"].values, -Zdf_mask["Zimag"].values, s=1.5)
    axes[2].set_xscale("log")
    axes[2].set_ylabel(r"$-Im(Z) / \Omega$")
    axes[2].set_xlabel("freq (Hz)")
    axes[2].set_title("Bode plot - Imaginary Part")
    plt.savefig(f"{folder_name}\\Filtered_Nyquist and Bode plots.png", dpi=300)
    plt.show()

    if threshold != 0.06:
        log.warn(f"Default threshold ({threshold-0.01}) dropped too many points; Proceed with caution.")

    return Zdf_mask, ohmic_resistance, RMSE_value


def save_processed_data(input_name: str, data: "pd.DataFrame") -> str:
    """
    Exports the processed EIS data in a CSV file.

    Parameters
    ----------
    input_name: str
        The data path of the given EIS data
    data_stored: pd.DataFrame
        The impedance data after pre-processing

    Returns
    -------
    file_name: str
        the data path of the processed EIS data

    """
    dirStr, ext = os.path.splitext(input_name)
    folder_name = dirStr.split("\\")[-1]
    mkdir(folder_name)

    if input_name.endswith(".json"):
        file_name = input_name.split(".json")[0]
    elif input_name.endswith(".csv"):
        file_name = input_name.split(".csv")[0]
    elif input_name.endswith(".txt"):
        file_name = input_name.split(".txt")[0]
    elif input_name.endswith(".xlsx"):
        file_name = input_name.split(".xlsx")[0]
    elif input_name.endswith(".pkl"):
        file_name = input_name.split(".pkl")[0]
    data.to_csv(
        folder_name + "//" + folder_name + "_processed" + ".csv", header=None, index=False
    )

    data.to_csv("temp" + ".csv", header=None, index=False)
    return file_name + ".csv"


def ECM_generation(data: "pd.DataFrame", times: int = 100, head: int = 12, save: bool = True):
    """
    Generate ECMs via evolutionary algorithms.

    Parameters
    ----------
    data: pd.DataFrame
        The processed EIS data (ideally after KK validation)
    times: int
        The number of times the ECM generation process is performed (default = 100)
    head: int
        The complexity of ECM search space (default = 12)
    save: bool
        Determines whether to export the results to disk

    Returns
    -------
     df_results: pd.DataFrame
        Dataframe containing ECM solutions generated by the evolutionary algorithm.

    """
    ec, jl_df, jl_pd, jl_Base = import_julia()

    ECM_solution = []
    for i in range(times):
        circuit_i = ec.circuit_evolution(
            np.array(data["Zreal"]) + 1j * np.array(data["Zimag"]),
            np.array(data["freq"]),
            head=head,
            terminals="RLP",
        )
        if circuit_i != jl_Base.nothing:
            ECM_solution.append(circuit_i)
    df_results = jl_pd.DataFrame(jl_df.DataFrame(ECM_solution))
    if len(df_results) == 0:
        log.warn("No plausible ECMs found. Try increasing the iterations.")
        return
    for i in range(len(df_results)):
        df_results["Parameters"][i] = jl_Base.string(df_results["Parameters"][i])
    if save:
        df_results.to_csv("df_circuits.csv", index=False)
    return df_results


def load_results(fname: str) -> "pd.DataFrame":
    """
    Loads the ECMs generated by the Julia program and converts it to a dataframe.

    Parameters
    ----------
    fname: str
        Path of the file containing the generated ECMs

    Returns:
    --------
    df_circuits: pd.DataFrame
        Dataframe containing the generated ECMs (2 columns)

    """

    df_circuits = pd.read_csv(fname)
    if len(df_circuits) == 0:
        log.error("No plausible ECMs found. Consider increasing the iterations.")
    return df_circuits


def split_components(df_circuits: "pd.DataFrame") -> "pd.DataFrame":
    """
    Splits all the components and their corresponding values of each ECM.

    Parameters
    ----------
    df_circuits: pd.DataFrame
        Dataframe containing the generated ECMs (2 columns)

    Returns
    -------
    df_circuits: pd.DataFrame
        Dataframe containing the generated ECMs (6 columns)

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


def capacitance_filter(df_circuits: "pd.DataFrame") -> "pd.DataFrame":
    """
    Excludes ideal capacitors from the circuits.

    Parameters
    ----------
    df_circuits: pd.DataFrame
       Dataframe containing the generated ECMs (6 columns)

    Returns
    -------
    df_circuits: pd.DataFrame
       Dataframe containing the generated ECMs without ideal capacitors (6 columns)

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


def ohmic_resistance_filter(
    df_circuits: "pd.DataFrame", ohmic_resistance: float
) -> "pd.DataFrame":
    """
    Extracts the ohmic resistance of each circuit and filters the circuits
    according to the values (with 15% buffer).

    Parameters
    ----------
    df_circuits: pd.DataFrame
       Dataframe containing the generated ECMs (6 columns)

    ohmic_resistance: float
        The ohmic resistance of the given EIS data

    Returns
    -------
    df_circuits: pd.DataFrame
        Dataframe containing the generated ECMs with correct ohmic resistance (6 columns)

    """

    for i in range(len(df_circuits["Circuit"])):
        # Find the series elements
        series_circuit = find_series_elements(circuit=df_circuits["Circuit"][i])
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


def series_filter(df_circuits: "pd.DataFrame") -> "pd.DataFrame":
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
    for i in range(len(df_circuits["Circuit"])):
        test_circuit = df_circuits["Circuit"][i]
        if test_pattern.findall(test_circuit) == False:
            df_circuits.drop([i], inplace=True)
    df_circuits.reset_index(drop=True, inplace=True)
    return df_circuits


def generate_mathematical_expression(df_circuits: "pd.DataFrame") -> "pd.DataFrame":
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

    for i in range(len(df_circuits["Circuit"])):
        circuit = df_circuits["Circuit"][i]
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
                    "X*(2*np.pi*F)**(-Y)*(np.cos((np.pi*Y)*0.5)-np.sin((np.pi*X)*0.5)*1j)",
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


def precise_rank_the_structure(circuit_array: "np.array") -> "np.array":
    """Rank each component in given circuits according to its 'complexity' in a more precise way (defined by how many parallel structures it's in)

    Parameter:
    ----------
    circuit_array: np.array
        the nparray that stores the circuit configurations

    Return:
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


def precise_extractor(circuit_array: "np.ndarray", precise_ranks_array: "np.ndarray") -> list:
    """
    Extracts the index information of each level circuit according to
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
    """
    Sorts the level lists.

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
    """
    Extracts the features of a given circuit and stores them as a
    dictionary with the following keys:

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
    if test_pattern.findall(circuit) == True:
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
    if test_pattern.findall(circuit) == True:
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
    for i in range(len(df_circuits["Circuit"])):
        feature_i = feature_store(df_circuits["Circuit"][i])
        equal_list = []
        equal_list_seq = []
        for j in range(len(df_circuits["Circuit"])):
            feature_j = feature_store(df_circuits["Circuit"][j])
            # if (feature_i['Feature 1'] == feature_j['Feature 1']).all() and (feature_i['Feature 2'] == feature_j['Feature 2']).all() and (feature_i['Feature 2.5'] == feature_j['Feature 2.5']).all() and feature_i['Feature 3'] == feature_j['Feature 3']:
            if len(feature_i) == len(feature_j) == 5:
                if (
                    feature_i["Feature 1"].tolist() == feature_j["Feature 1"].tolist()
                    and feature_i["Feature 2"].tolist() == feature_j["Feature 2"].tolist()
                    and feature_i["Feature 2.5"].tolist() == feature_j["Feature 2.5"].tolist()
                    and feature_i["Feature 3"] == feature_j["Feature 3"]
                ):
                    equal_list.append(df_circuits["Circuit"][j])
                    equal_list_seq.append(j)
            else:
                if (
                    feature_i["Feature 1"].tolist() == feature_j["Feature 1"].tolist()
                    and feature_i["Feature 2"].tolist() == feature_j["Feature 2"].tolist()
                ):
                    equal_list.append(df_circuits["Circuit"][j])
                    equal_list_seq.append(j)
        equal_lists.append(equal_list)
        equal_lists_seq.append(equal_list_seq)
    return equal_lists, equal_lists_seq


def filter(similar_circuits: "list") -> "list":
    """
    Filters the repeated "identical circuits list" in the list.

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
    """
    Identfies the identical circuits configurations by the above features.

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
    """
    Before combination, separates the components names and values for
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
    # Delete the ( and ) in the string
    delete_p = re.compile(r"[^()]")

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
            if e_p.findall(values_list[j]) == False:
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
    """
    Combines the identical circuits.

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
        combined_expressions.append(df_circuits["Circuit"][similar_expression_index[i][0]])
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
                        df_circuits["Circuit"][similar_expression_index[i][0]]
                    ]
                    combined_expressions[i].append(
                        df_circuits["Circuit"][similar_expression_index[i][j]]
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
    """
    Counts how many different value sets are in identical circuits.

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
    """
    Separates the value and name of each component.

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
    """
    Calculates of the coefficient of determintion, goodness of fit metric.

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


def MSE_calculator(y_true: "np.ndarray", y_pred: "np.ndarray"):
    """
    Calculates of the mean square error, goodness of fit metric.

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


def RMSE_calculator(y_true: "np.array", y_pred: "np.array") -> "float":
    """
    Calculates of the root mean square error, goodness of fit metric.

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


def MAPE_calculator(y_true: "np.array", y_pred: "np.array") -> "float":
    """
    Calculates of the mean abosulte percentage error, goodness of fit metric.

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
    """
    Evaluates the posterior distributions according to their shapes.

    Parameters
    -----------
    posteriors: Axesubplots
        The axesubplots that record posterior distribution

    Return:
    -----------
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
    evaluation_results = results[results.columns[[0, 17, 18, 19, 20, 23, 25]]]

    evaluation_results["Consistency"] = pd.to_numeric(
        evaluation_results["Consistency"], errors="coerce"
    )
    evaluation_results.loc[evaluation_results["Consistency"].isna(), "Consistency"] = np.inf

    def absolute_difference(x):
        if np.isinf(x):
            return np.inf
        else:
            return abs(x - 1)

    def custom_sort(x):
        if x == "F":
            return -1000
        else:
            return x

    evaluation_results["Consistency"] = evaluation_results["Consistency"].apply(
        absolute_difference
    )
    evaluation_results["Posterior_shape"] = evaluation_results["Posterior_shape"].apply(
        custom_sort
    )
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


def convert_text(text: str, save=True):
    """
    Visualizes circuit string (convert the string of ECMs to figures)

    Parameters
    ----------
    text: str
        The string that stores the circuit configuration

    Returns
    -------
    circuit_figure: lcapy.figure
        Handle of the circuit figure

    """
    from lcapy import CPE as P
    from lcapy import C, L, R

    # Replace square brackets with parentheses
    text = text.replace("[", "(").replace("]", ")")
    # Replace commas with vertical bars
    text = text.replace(",", "|")
    # Replace dashes with plus signs
    text = text.replace("-", "+")
    # Surround all numbers with parentheses
    text = re.sub(r"([A-Z])(\d+)", r'\1("\1\2")', text)
    circuit_figure = eval(text)
    circuit_figure.draw(style="american")

    return circuit_figure


def Bayesian_inference(
    data: "pd.DataFrame",
    df: "pd.DataFrame",
    data_path: str,
    plot: bool = True,
    save: bool = True,
    ECM_figure=False,
) -> "pd.DataFrame":
    """
    Performs Bayesian inference on the ECMs.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe containing data after pre-processing (3 columns expected);
        frequency, real part, and the imaginary part of impedance data
    df: pd.DataFrame
        Dataframe containing filtered ECMs
    data_path: str
        The data path of the original EIS data for storage
    plot: bool
        Controls whether to plot or not
    save: bool
        Controls whether to save data or not

    Returns
    -------
    df: pd.DataFrame
        Dataframe containing the ECMs with the Bayesian inference results (12 columns)

    """
    # Determine if there's any ECM that passed post-filtering process
    if len(df) == 0:
        log.error("No plausible ECMs found. Try increasing the iterations.")

    # Set the parameters for plots
    az.style.use("arviz-darkgrid")

    freq = np.array(data["freq"])
    Zreal = np.array(data["Zreal"])
    Zimag = np.array(data["Zimag"])

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

    # Start from this source of randomness. We will split keys for subsequent operations.
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    # BI parts:
    values = df["Variables_values"]
    names = df["Variables_names"]
    expressions_strs = df["Mathematical expressions"]
    circuit_names = df["Combined Circuits"]
    num_of_values = df["Combined Values"]
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

    dirStr, ext = os.path.splitext(data_path)
    #     folder_name = dirStr.split('\\')[-1]
    folder_name = dirStr

    current_path = os.getcwd()

    for i in range(len(df["Combined Circuits"])):
        circuit_name_i = circuit_names[i]
        value_i = values[i]
        name_i = names[i]
        expression_str_i = expressions_strs[i].replace("np.", "jnp.")
        function_i = eval(f"lambda X,F:{expression_str_i}")

        # create a new folder to store these results
        mkdir(folder_name + f"\\{circuit_name_i}")
        os.chdir(folder_name + f"\\{circuit_name_i}")

        print(f"----------Circuit {i}:{circuit_name_i}--------------")
        print(f"Circuit elements: {name_i})  \nValues: ({value_i})")

        if plot:
            if ECM_figure:
                ECM_plot = convert_text(circuit_name_i)

        ECM_data = function_i(value_i, freq)
        ECMs_data.append(ECM_data)

        print("Julia circuit's fitting")

        r2_value = float(r2_calculator(Zreal + 1j * Zimag, ECM_data))
        print(f"r2_value:{r2_value}")
        R2_list.append(r2_value)

        r2_real = r2_calculator(Zreal, ECM_data.real)
        print(f"r2_real_part:{r2_real}")
        R2_real_list.append(r2_real)
        r2_imag = r2_calculator(Zimag, ECM_data.imag)
        print(f"r2_imag_part:{r2_imag}")
        R2_imag_list.append(r2_imag)

        MSE_value = float(MSE_calculator(Zreal + 1j * Zimag, ECM_data))
        print(f"MSE_value:{MSE_value}")
        MSE_list.append(MSE_value)

        RMSE_value = float(MSE_calculator(Zreal + 1j * Zimag, ECM_data) ** (1 / 2))
        print(f"RMSE_value:{RMSE_value}")
        RMSE_list.append(RMSE_value)

        MAPE_value = float(MAPE_calculator(Zreal + 1j * Zimag, ECM_data) ** (1 / 2))
        print(f"MAPE_value:{MAPE_value}")
        MAPE_list.append(MAPE_value)

        if plot:
            plt.scatter(ECM_data.real, -ECM_data.imag, c="r", s=12, label="Simulated data")
            plt.scatter(Zreal, -Zimag, c="b", s=12, label="Original data")
            plt.xlabel("Real impedance")
            plt.ylabel("- Imag impedance")
            plt.title("Nyquist plots of original data and simulated data")
            plt.legend()
            if save:
                plt.savefig(f"Nyquist_plots_of_simulated_data.png", dpi=300)
            plt.show()

        def model_i(
            values=value_i, func=function_i, true_data=data, error=relative_error_accepted
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

        prior_predictive = Predictive(model_i, num_samples=200)
        prior_prediction = prior_predictive(rng_key)
        Prior_predictions.append(prior_prediction)

        kernel = NUTS(model_i, target_accept_prob=0.8)
        num_samples = 10000
        mcmc_i = MCMC(kernel, num_warmup=1000, num_samples=num_samples, num_chains=1)
        mcmc_i.run(
            rng_key_,
            values=value_i,
            func=function_i,
            true_data=data,
            error=relative_error_accepted,
        )

        # Results
        models.append(mcmc_i)
        models_descriptions.append(mcmc_i.print_summary)

        trace = az.convert_to_inference_data(mcmc_i)
        trace.to_netcdf("MCMC_Results.nc")
        traces.append(trace)
        # Calculate AIC

        AIC_value = az.waic(mcmc_i)[0] * (-2) + 2 * len(name_i)
        AIC.append(AIC_value)
        print(f"AIC value = {AIC_value}")

        divergence = np.asarray(mcmc_i.get_extra_fields()["diverging"].sum()).ravel()[0]

        divergences.append(divergence)

        # Prior distributions
        if plot:
            print(f"{circuit_name_i}:Prior distributions with trajectories")
            az.plot_trace(prior_prediction, var_names=name_i)
            if save:
                plt.savefig(f"Prior distributions.png", dpi=300)
            plt.show()

        # Prior predictions
        if plot:
            print(f"{circuit_name_i}:Prior prediction")
            _, ax = plt.subplots()

        prior_R2_list = []
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
            ax.set_xlabel("real impedance ")
            ax.set_ylabel("imag impedance")
            ax.set_title("Prior predictive checks")
            if save:
                plt.savefig(f"Prior prediction.png", dpi=300)
            plt.show()

        # Posterior distributions
        if plot:
            print(f"{circuit_name_i}:Posterior distributions with HDI")
            for i in range(len(name_i)):
                name = name_i[i]
                value = value_i[i]
                if "n" not in name:
                    trace.posterior[name] = trace.posterior[name] * value
            posterior_HDI = az.plot_posterior(trace, var_names=name_i)
            #             for i in range(posterior_HDI.shape[0]):
            #                 for j in range(posterior_HDI.shape[1]):
            #                     rc_id = i*3 + j
            #                     if rc_id < len(value_i):
            #                         y_values = posterior_HDI[i][j].lines[0].get_ydata()
            #                         posterior_HDI[i][j].lines[0].set_data(np.multiply(posterior_HDI[i][j].lines[0].get_xydata()[:,0],value_i[rc_id]),y_values)
            # #                         new_lim = np.multiply(posterior_HDI[i][j].get_xlim(),value_i[rc_id])
            # #                         posterior_HDI[i][j].set_xlim(new_lim)
            if save:
                plt.savefig(f"Posterior prediction_with HDI.png", dpi=300)
            plt.show()

        # Posterior trajectories
        posterior_dist = az.plot_trace(trace, var_names=name_i)

        if plot:
            print(f"{circuit_name_i}:Posterior distributions with trajectories")
            if save:
                plt.savefig(f"Posterior distributions.png", dpi=300)
            plt.show()

        # Posterior predictions -- real part
        if plot:
            print(f"{circuit_name_i}:Posterior predictions - real part")
            _, ax = plt.subplots()

        samples = mcmc_i.get_samples()
        Posterior_predictions.append(samples)

        sep_mape_real_list = []
        sep_r2_real_list = []

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
                ax.plot(np.log10(freq), BI_data.real, marker=".", color="grey", alpha=0.5)
            sep_mape_real = float(MAPE_calculator(Zreal, BI_data.real))
            sep_mape_real_list.append(sep_mape_real)
            sep_r2_real = float(r2_calculator(Zreal, BI_data.real))
            sep_r2_real_list.append(sep_r2_real)

        avg_mape_real = np.array(sep_mape_real_list).mean()
        avg_r2_real = np.array(sep_r2_real_list).mean()
        if plot:
            print(f"Posterior real part's fit: MAPE = {avg_mape_real}; R2 = {avg_r2_real}")
        Posterior_r2_real.append(avg_r2_real)
        Posterior_mape_real.append(avg_mape_real)

        if plot:
            ax.plot(
                np.log10(freq),
                BI_data.real,
                marker=".",
                ms=15,
                color="grey",
                alpha=0.5,
                label="Predictive EIS",
            )
            ax.plot(
                np.log10(freq),
                Zreal,
                "--",
                marker="o",
                c="b",
                alpha=0.9,
                ms=8,
                label="Ground Truth EIS",
            )

            ax.set_xlabel("log(freq)")
            ax.set_ylabel("real impedance")
            ax.set_title("Posterior predictive checks - Real parts")
            if save:
                plt.savefig(f"Posterior prediction_real parts.png", dpi=300)
            plt.legend()
            plt.show()

        # Posterior predictions -- imag part
        if plot:
            print(f"{circuit_name_i}:Posterior predictions imag")
            _, ax = plt.subplots()

        sep_mape_imag_list = []
        sep_r2_imag_list = []

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
                ax.plot(np.log10(freq), -BI_data.imag, marker=".", color="grey", alpha=0.5)
            sep_mape_imag = float(MAPE_calculator(Zimag, BI_data.imag))
            sep_mape_imag_list.append(sep_mape_imag)
            sep_r2_imag = float(r2_calculator(Zimag, BI_data.imag))
            sep_r2_imag_list.append(sep_r2_imag)

        avg_mape_imag = np.array(sep_mape_imag_list).mean()
        avg_r2_imag = np.array(sep_r2_imag_list).mean()
        if plot:
            print(f"Posterior imag part's fit: MAPE = {avg_mape_imag}; R2 = {avg_r2_imag}")
        Posterior_r2_imag.append(avg_r2_imag)
        Posterior_mape_imag.append(avg_mape_imag)
        if plot:
            ax.plot(
                np.log10(freq),
                -BI_data.imag,
                marker=".",
                ms=15,
                color="grey",
                alpha=0.5,
                label="Predictive EIS",
            )
            ax.plot(
                np.log10(freq),
                -Zimag,
                "--",
                marker="o",
                c="b",
                alpha=0.9,
                ms=8,
                label="Ground Truth EIS",
            )

            ax.set_xlabel("log(freq) ")
            ax.set_ylabel("-imag impedance")
            ax.set_title("Posterior predictive checks - Imag parts")
            if save:
                plt.savefig(f"Posterior prediction_imag parts.png", dpi=300)
            plt.legend()
            plt.show()

        # Posterior predictions
        if plot:
            print(f"{circuit_name_i}:Posterior predictions")
            _, ax = plt.subplots()

        sep_mape_list = []
        sep_r2_list = []
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

        # avg_mse = np.array(sep_mse_list).mean()
        avg_mape = np.array(sep_mape_list).mean()
        avg_r2 = np.array(sep_r2_list).mean()
        if plot:
            print(f"Posterior fit: MAPE = {avg_mape}; R2 = {avg_r2}")
        Posterior_r2.append(avg_r2)
        Posterior_mape.append(avg_mape)

        if plot:
            #             ax.plot(BI_data.real, -BI_data.imag ,marker='.',ms=15,color='grey', alpha=0.5,label='Predictive EIS')
            #             ax.plot(Zreal,-Zimag,'--',marker='o',c='b',alpha=0.9,ms=8,label = 'Ground Truth EIS')
            ax.plot(
                BI_data.real,
                -BI_data.imag,
                marker=".",
                ms=15,
                color="grey",
                alpha=0.5,
                label="Predictions",
            )
            ax.plot(
                Zreal, -Zimag, "--", marker="o", c="b", alpha=0.9, ms=8, label="Ground Truth"
            )
            ax.set_xlabel("real impedance ")
            ax.set_ylabel("imag impedance")
            ax.set_title("Posterior predictive checks")
            if save:
                plt.savefig(f"Posterior prediction.png", dpi=300)
            plt.legend(loc="upper left", fontsize=18)
            plt.show()

        #         Pair relationship
        if plot:
            az.plot_pair(mcmc_i, var_names=name_i)
            if save:
                plt.savefig(f"Pair relationship plot_{circuit_name_i}.png", dpi=300)
            plt.show()

        #         estimate posterior distribution
        if any(len(result[0].lines[0].get_xydata().T[0]) == 2 for result in posterior_dist[:]):
            posterior_mark = "F"
        else:
            posterior_mark = posterior_evaluation(posterior_dist)
        posterior_shape.append(posterior_mark)

        r_hats = []
        for i in range(len(name_i)):
            r_hats.append(
                summary(mcmc_i.get_samples(), prob=0.94, group_by_chain=False)[f"{name_i[i]}"][
                    "r_hat"
                ]
            )
        posterior_rhat = np.mean(r_hats)
        consistency.append(posterior_rhat)

        for i in range(2):
            os.chdir(os.path.abspath(os.path.dirname(os.getcwd())))

    df["ECM Data"] = ECMs_data
    df["R_square"] = R2_list
    df["Mean Square Error"] = MSE_list
    df["Mean Absolute Percentage Error"] = MAPE_list
    df["Root Mean Square Error"] = RMSE_list
    df["BI_models"] = models
    df["Traces"] = traces
    df["BI_models_description"] = models_descriptions
    df["Priors_prediction"] = Prior_predictions
    df["Posterior_prediction"] = Posterior_predictions
    df["AIC Value"] = AIC
    df["Divergences"] = divergences
    df["Consistency"] = consistency
    df["Posterior_shape"] = posterior_shape
    df["Posterior_mean_r2"] = Posterior_r2
    df["Posterior_mean_mape"] = Posterior_mape
    df["Posterior_mean_r2_real"] = Posterior_r2_real
    df["Posterior_mean_mape_real"] = Posterior_mape_real
    df["Posterior_mean_r2_imag"] = Posterior_r2_imag
    df["Posterior_mean_mape_imag"] = Posterior_mape_imag

    df = model_evaluation(df)

    df_dict = df.to_dict()
    os.chdir(current_path)
    if save:
        with open(f"{folder_name}//results.pkl", "wb") as handle:
            dill.dump(df_dict, handle)

    #         with open(f'{data_path}_results.json','w') as file_obj:
    #             json.dum`bp(df_dict,file_obj)
    # load data:
    # with open('file.pkl', 'rb') as f:
    #     input_dict = dill.load(f)
    return df


def EIS_auto(
    impedance: "np.ndarray",
    freq: "np.ndarray",
    fname: str,
    iter_number: int = 100,
    plot_ECM: bool = False,
) -> "pd.DataFrame":
    """
    The main function to automate the whole EIS analysis by ECMs+ BI.

    Parameters
    ----------
    impedance: np.ndarray
        The impedance data
    freq: np.ndarray
        The frequencies of the impedance data
    fname: str
        The data path of the impedance data
    iter_number:int
        The number of times the ECM generation is performed
    plot_ECM:bool
        Determine whether to plot ECM or not

    Returns
    -------
    results: pd.DataFrame
        Dataframe containing effective ECMs after filtering + BI results (12 columns)

    """
    # Set the plotting style
    set_parameter()
    ec, jl_df, jl_pd, jl_Base = import_julia()

    # Preprocessing + store preprocessed data
    print("---------------Data_processing---------------")
    data_processed, ohmic_resistance, RMSE = pre_processing(impedance, freq, 0.05, fname)
    path_data_preprocessed = save_processed_data(
        input_name=fname, data=data_processed
    )

    # Call julia program
    print("---------------ECM generation in process---------------")
    df_results = ECM_generation(data=data_processed, times=iter_number)
    # an alternative method: direcly call julia script - this might be faster
    # run_julia = j.include('test_julia.jl')

    # Load the results - 1.from the results file
    # path_results = "df_results.csv"
    # df_circuits = load_results(file_path = path_results)

    # Load the results
    df_circuits = split_components(df_results)
    df_circuits = capacitance_filter(df_circuits)
    df_circuits = series_filter(df_circuits)
    df_circuits = ohmic_resistance_filter(df_circuits, ohmic_resistance)
    df_circuits = generate_mathematical_expression(df_circuits)
    new_df = combine_expression(df_circuits)
    new_df = calculate_length(new_df)
    new_df = split_variables(new_df)
    results = Bayesian_inference(
        data=data_processed, data_path=fname, df=new_df, ECM_figure=plot_ECM
    )

    return results


def EIS_auto_script(
    impedance: "np.ndarray",
    freq: "np.ndarray",
    data_path: str,
    iter_number: int = 100,
    plot_ECM: bool = False,
) -> "pd.DataFrame":
    """
    The main function to automate the whole EIS analysis by ECMs + BI.

    Parameters
    ----------
    impedance: np.ndarray
        The impedance data
    freq: np.ndarray
        The frequencies of the impedance data
    data_path: str
        The data path of the impedance data
    iter_number: int
        The number of times the ECM generation is performed
    plot_ECM: bool
        Determine whether to plot ECM or not

    Returns
    -------
    results: pd.DataFrame
        Dataframe containing effective ECMs after filtering + BI results (12 columns)

    """
    # Set the plotting style
    set_parameter()
    ec, jl_df, jl_pd, jl_Base = import_julia()

    # Preprocessing + store preprocessed data
    print("---------------Data_preprocessing---------------")
    data_processed, ohmic_resistance, RMSE = pre_processing(impedance, freq, 0.05, data_path)
    path_data_preprocessed = save_processed_data(
        input_name=data_path, data=data_processed
    )

    # ECM generation
    print("---------------ECM generation in process---------------")
    # Alternative method: direcly call julia script - this might be faster
    run_julia = j.include("test_julia.jl")

    # Load the results - 1.from the results file
    path_results = "df_results.csv"
    df_circuits = load_results(fname=path_results)

    # Load the results
    df_circuits = split_components(df_results)
    df_circuits = capacitance_filter(df_circuits)
    df_circuits = series_filter(df_circuits)
    df_circuits = ohmic_resistance_filter(df_circuits, ohmic_resistance)
    df_circuits = generate_mathematical_expression(df_circuits)
    new_df = combine_expression(df_circuits)
    new_df = calculate_length(new_df)
    new_df = split_variables(new_df)
    results = Bayesian_inference(
        data=data_processed, data_path=data_path, df=new_df, ECM_figure=plot_ECM
    )
    return results


if __name__ == "__main__":
    # Define the path of julia program
    j = set_julia(r"D:\Julia-1.7.2\bin\julia.exe")

    # Initialize the julia environment
    # Initialize_julia()

    # Set the parameter of plots
    set_parameter()

    # Load the EIS data and get access to the impedance and frequencies
    fname = "Test_data.txt"
    df = load_data(fname)

    frequencies = np.array(df["freq/Hz"]).astype(float)
    reals = np.array(df["Re(Z)/Ohm"]).astype(float)
    imags = -np.array(df["-Im(Z)/Ohm"]).astype(float)

    measurements = reals + imags * 1j

    # Analysis board
    results = EIS_auto(
        impedance=measurements, freq=frequencies, fname=fname, iter_number=100
    )
    print(results)
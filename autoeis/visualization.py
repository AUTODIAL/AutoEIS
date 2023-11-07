"""
Collection of functions for visualizing EIS data and results.

.. currentmodule:: autoeis.visualization

.. autosummary::
   :toctree: generated/

    draw_circuit
    plot_impedance
    plot_linKK_residuals
    set_plot_style

"""

import re

import arviz
import matplotlib.pyplot as plt

import autoeis.utils as utils

log = utils.get_logger(__name__)

__all__ = [
    "draw_circuit",
    "plot_impedance",
    "plot_linKK_residuals",
    "set_plot_style",
]


def draw_circuit(circuit: str):
    """Visualize circuit model using lcapy.

    Parameters
    ----------
    circuit: str
        The string that stores the circuit configuration

    Returns
    -------
    fig: lcapy.figure
        Handle of the circuit figure
    """
    try:
        from lcapy import CPE as P
        from lcapy import C, L, R
    except ImportError:
        msg = "lcapy is not installed. Please install it using `pip install lcapy`."
        log.error(msg)
        return

    # Replace square brackets with parentheses
    circuit = circuit.replace("[", "(").replace("]", ")")
    # Replace commas with vertical bars
    circuit = circuit.replace(",", "|")
    # Replace dashes with plus signs
    circuit = circuit.replace("-", "+")
    # Surround all numbers with parentheses
    circuit = re.sub(r"([A-Z])(\d+)", r'\1("\1\2")', circuit)

    fig = eval(circuit)
    fig.draw(style="american")

    return fig


def plot_impedance(Z, freq, saveto=None, size=1.5):
    """Plot EIS data in Nyquist and Bode plots."""
    Re_Z = Z.real
    Im_Z = Z.imag

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Nyquist plot
    axes[0].scatter(Re_Z, -Im_Z, s=size)
    axes[0].set_xlabel(r"$Re(Z) / \Omega$")
    axes[0].set_ylabel(r"$-Im(Z) / \Omega$")

    # Bode plot (magnitude) <- Re(Z)
    ax1 = axes[1]
    ax1.scatter(freq, Re_Z, s=size, color='blue', label=r'$Re(Z)$')
    ax1.set_xscale("log")
    ax1.set_xlabel("freq (Hz)")
    ax1.set_ylabel(r"$Re(Z) / \Omega$")
    ax1.legend(loc='upper left')

    # Bode plot (phase) <- Im(Z)
    ax2 = ax1.twinx()  # instantiate a second y-axis sharing the same x-axis
    ax2.scatter(freq, -Im_Z, s=size, color='red', label=r'$-Im(Z)$')
    ax2.set_ylabel(r"$-Im(Z) / \Omega$")
    ax2.legend(loc='upper right')

    if saveto is not None:
        fig.savefig(saveto, dpi=300)
    
    return fig, axes


def plot_linKK_residuals(frequencies, Re_res, Im_res, saveto=None):
    """Plot the residuals of the linear Kramers-Kronig validation."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(frequencies, Im_res, label="delta Im")
    ax.plot(frequencies, Re_res, label="delta Re")
    ax.set_xlabel("freq (Hz)")
    ax.set_ylabel("delta %")
    ax.set_xscale("log")
    ax.set_title("Lin-KK validation")
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.legend()

    if saveto is not None:
        fig.savefig(saveto, dpi=300)
    
    return fig, ax


def set_plot_style(use_arviz=True) -> None:
    """Modify the default arviz/matplotlib parameters for prettier plots."""
    # Arviz
    if use_arviz:
        arviz.style.use("arviz-darkgrid")

    # Matplotlib
    label_size = 11
    tick_size = label_size - 1
    title_size = label_size + 1
    legend_size = label_size - 1

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Verdana", "Tahoma", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["xtick.labelsize"] = tick_size
    plt.rcParams["ytick.labelsize"] = tick_size
    plt.rcParams["axes.labelsize"] = label_size
    plt.rcParams["axes.titlesize"] = title_size
    plt.rcParams["legend.fontsize"] = legend_size

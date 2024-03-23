"""
Collection of functions for visualizing EIS data and results.

.. currentmodule:: autoeis.visualization

.. autosummary::
   :toctree: generated/

    draw_circuit
    plot_impedance_combo
    plot_linKK_residuals
    plot_nyquist
    set_plot_style
    show_nticks

"""

import re

import arviz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import rich
import seaborn as sns
from matplotlib.axes import Axes
from rich.console import Console
from rich.table import Table

import autoeis.utils as utils

log = utils.get_logger(__name__)

__all__ = [
    "draw_circuit",
    "plot_impedance_combo",
    "plot_linKK_residuals",
    "plot_nyquist",
    "print_summary_statistics",
    "rich_print",
    "set_plot_style",
    "show_nticks",
]


def is_ipython_notebook() -> bool:  # pragma: no cover
    """Returns True if the code is running in a Jupyter notebook, False otherwise."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        if shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def rich_print(*args, **kwargs):
    """Overrides the built-in print() function with rich's print() function."""
    if is_ipython_notebook():
        # HACK: prevent rich from creating a new <div> for every print()
        # NOTE: works for notebooks, but breaks for interactive sessions
        console = Console(force_jupyter=False)
        console.print(*args, **kwargs)
    else:
        rich.print(*args, **kwargs)


def draw_circuit(circuit: str) -> mpl.figure.Figure:
    """Draws the circuit model using lcapy.

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


def plot_nyquist(
    Z: np.ndarray[complex],
    fmt: str = "o-",
    size: int = 4,
    color: str = None,
    alpha: int = 1,
    label: str = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots EIS data in Nyquist plot."""
    if ax is None:
        fig, ax = plt.subplots()

    # Remove color from fmt if present
    if fmt[0] in ["b", "g", "r", "c", "m", "y", "k", "w"]:
        color = fmt[0]
        fmt = fmt[1:]

    ax.plot(Z.real, -Z.imag, fmt, c=color, markersize=size, label=label, alpha=alpha)
    ax.set_xlabel("Re(Z)")
    ax.set_ylabel("-Im(Z)")
    ax.axis("equal")

    if label is not None:
        ax.legend()

    return ax.figure, ax


def plot_impedance_combo(
    Z: np.ndarray[complex],
    freq: np.ndarray[float],
    size: int = 10,
    ax: list[plt.Axes] = None,
    scatter=True,
    label=None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plots EIS data in Nyquist and Bode plots."""
    Re_Z = Z.real
    Im_Z = Z.imag

    if ax is None:
        fig, ax = plt.subplots(ncols=2)
    assert not isinstance(ax, Axes), "Incompatible 'ax'. Use plt.subplots(ncols=2)"
    fig = ax[0].figure
    fig.set_size_inches(8, 3)

    # Nyquist plot
    plot = getattr(ax[0], "scatter" if scatter else "plot")
    kwargs = {"s": size} if scatter else {}
    plot(Re_Z, -Im_Z, label=label, **kwargs)
    ax[0].set_xlabel(r"$Re(Z) / \Omega$")
    ax[0].set_ylabel(r"$-Im(Z) / \Omega$")
    ax[0].axis("equal")

    if label is not None:
        ax[0].legend()

    # Bode plot (magnitude) <- Re(Z)
    if not isinstance(ax[1], list):
        ax[1] = [ax[1], ax[1].twinx()]
    plot = getattr(ax[1][0], "scatter" if scatter else "plot")
    plot(freq, Re_Z, color="blue", label=r"$Re(Z)$", **kwargs)
    ax[1][0].set_xscale("log")
    ax[1][0].set_xlabel("freq (Hz)")
    ax[1][0].set_ylabel(r"$Re(Z) / \Omega$")
    ax[1][0].yaxis.label.set_color("blue")

    # Bode plot (phase) <- Im(Z)
    plot = getattr(ax[1][1], "scatter" if scatter else "plot")
    plot(freq, -Im_Z, color="red", label=r"$-Im(Z)$", **kwargs)
    ax[1][1].set_ylabel(r"$-Im(Z) / \Omega$")
    ax[1][1].yaxis.label.set_color("red")
    # Don't show grid lines for the second y-axis (ax[1][0] already has them)
    ax[1][1].grid(False)
    fig.tight_layout()

    return fig, ax


def plot_linKK_residuals(
    freq: np.ndarray[float],
    res_real: np.ndarray[float],
    res_imag: np.ndarray[float],
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots the residuals of the linear Kramers-Kronig validation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(freq, res_real, label="delta Re")
    ax.plot(freq, res_imag, label="delta Im")
    ax.set_xlabel("freq (Hz)")
    ax.set_ylabel("delta %")
    ax.set_xscale("log")
    ax.set_title("Lin-KK validation")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend()
    return ax.figure, ax


def print_summary_statistics(mcmc: "numpyro.MCMC", circuit: str):
    """Prints summary statistics for the MCMC run."""
    # Set up required Rich objects
    console = Console()
    num_div = mcmc.get_extra_fields()["diverging"].sum()
    num_samples = mcmc.get_extra_fields()["diverging"].size
    title = f"{circuit}, {num_div}/{num_samples} divergences"
    table = Table(title=title, show_header=True, header_style="bold")

    # Add columns to the table
    columns = ["Parameter", "Mean", "Std", "Median", "5.0%", "95.0%"]
    for column in columns:
        table.add_column(column, justify="right")

    # Fill the table with data
    for label, values in mcmc.get_samples().items():
        rows = [
            label,
            f"{np.mean(values):.2e}",
            f"{np.std(values):.2e}",
            f"{np.median(values):.2e}",
            f"{np.percentile(values, 5):.2e}",
            f"{np.percentile(values, 95):.2e}",
        ]
        # Highlight rows with high standard deviation
        row_style = "on yellow" if np.std(values) > np.mean(values) else ""
        # Add a row to the table
        table.add_row(*rows, style=row_style)

    # Print the table
    console.print(table)


def override_mpl_colors(override_named_colors: bool = True):
    """Override matplotlib's default colors with Flexoki colors."""
    # Define the Flexoki-Light color scheme based on the provided table
    # Original sequence: red, orange, yellow, green, cyan, blue, purple, magenta
    flexoki_light_colors = {
        "red": "#D14D41",
        "blue": "#4385BE",
        "green": "#879A39",
        "orange": "#DA702C",
        "purple": "#8B7EC8",
        "yellow": "#D0A215",
        "cyan": "#3AA99F",
        "magenta": "#CE5D97",
    }

    # Override default named colors
    if override_named_colors:
        mpl.colors._colors_full_map.update(flexoki_light_colors)

    # Define the Flexoki-Light style
    flexoki_light_style = {
        "axes.prop_cycle": mpl.cycler(color=list(flexoki_light_colors.values())),
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.labelcolor": "black",
        "figure.facecolor": "white",
        "grid.color": "whitesmoke",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.color": flexoki_light_colors["blue"],
        "patch.edgecolor": "black",
        # "axes.spines.top": False,
        # "axes.spines.right": False
    }

    # Apply the Flexoki-Light style
    plt.style.use(flexoki_light_style)


def set_plot_style(
    use_arviz: bool = True, use_seaborn: bool = True, use_flexoki: bool = True
):
    """Modifies the default arviz/matplotlib config for prettier plots."""
    # Arviz
    if use_arviz:
        arviz.style.use("arviz-viridish")

    # Seaborn
    if use_seaborn:
        sns.set_style("ticks")

    # Matplotlib
    label_size = 11
    tick_size = label_size - 1
    title_size = label_size + 1
    legend_size = label_size - 1

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Helvetica",
        "Arial",
        "Verdana",
        "Tahoma",
        "DejaVu Sans",
    ]
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["xtick.labelsize"] = tick_size
    plt.rcParams["ytick.labelsize"] = tick_size
    plt.rcParams["axes.labelsize"] = label_size
    plt.rcParams["axes.titlesize"] = title_size
    plt.rcParams["legend.fontsize"] = legend_size
    plt.rcParams["legend.frameon"] = False

    # Flexoki colors
    if use_flexoki:
        override_mpl_colors()

    # Set up Jupyter notebook
    try:
        import IPython

        IPython.display.set_matplotlib_formats("retina")
    except ImportError:
        pass


def show_nticks(ax: plt.Axes, x: bool = True, y: bool = False, n: int = 10):
    """In-place modifies Matplotlib axes to show only n ticks."""
    if x:
        xticks = ax.xaxis.get_major_ticks()
        if len(xticks) > n:
            ax.xaxis.set_major_locator(plt.MaxNLocator(n, steps=[1, 2, 5, 10]))
    if y:
        yticks = ax.yaxis.get_major_ticks()
        if len(yticks) > n:
            ax.yaxis.set_major_locator(plt.MaxNLocator(n, steps=[1, 2, 5, 10]))

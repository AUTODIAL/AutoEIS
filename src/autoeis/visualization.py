"""
Collection of functions for visualizing EIS data and results.

.. currentmodule:: autoeis.visualization

.. autosummary::
   :toctree: generated/

    draw_circuit
    plot_bode
    plot_nyquist
    plot_impedance_combo
    plot_linKK_residuals
    plot_nyquist
    set_plot_style
    show_nticks

"""

import logging
import os
import re
from collections.abc import Iterable

import arviz
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
import rich
import seaborn as sns
from matplotlib.axes import Axes
from pandas.io.formats.style import Styler
from rich.console import Console
from rich.table import Table

log = logging.getLogger(__name__)


__all__ = [
    "draw_circuit",
    "plot_bode",
    "plot_nyquist",
    "plot_impedance_combo",
    "plot_linKK_residuals",
    "plot_nyquist",
    "print_summary_statistics",
    "print_inference_results",
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
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    fig : lcapy.figure
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
    *,
    fmt: str = ".-",
    markersize: int = 6,
    color: str = None,
    alpha: float = 1,
    label: str = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Plots EIS data in Nyquist plot.

    Parameters
    ----------
    Z: np.ndarray[complex]
        Impedance data.
    fmt: str, optional
        Format of the markers in the plot. Default is "o-".
    markersize: int, optional
        Size of the markers in the plot. Default is 6.
    color: str, optional
        Color of the markers in the plot. Default is None.
    alpha: float, optional
        Transparency of the markers in the plot. Default is 1.
    label: str, optional
        Label for the plot. Default is None.
    ax: plt.Axes, optional
        Axes to plot on. Default is None.

    Returns
    -------
    plt.Axes
        Axes object of the Nyquist plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")

    # Remove color from fmt if present
    if fmt[0] in ["b", "g", "r", "c", "m", "y", "k", "w"]:
        color = fmt[0]
        fmt = fmt[1:]

    ax.plot(Z.real, -Z.imag, fmt, c=color, markersize=markersize, label=label, alpha=alpha)
    ax.set_xlabel("Re(Z)")
    ax.set_ylabel("-Im(Z)")
    ax.axis("equal")

    if label is not None:
        ax.legend()

    return ax


def plot_bode(
    freq: np.ndarray,
    Z: np.ndarray,
    *,
    fmt=".-",
    markersize=6,
    deg: bool = True,
    ax: np.ndarray[plt.Axes] = None,
    label: str = None,
    log: bool = False,
    alpha: float = 1,
) -> np.ndarray:
    """Plots the Bode plot for the impedance data.

    Parameters
    ----------
    freq: np.ndarray
        Frequencies corresponding to the impedance data.
    Z: np.ndarray
        Impedance data.
    fmt: str, optional
        Format of the markers in the plot. Default is ".-".
    markersize: int, optional
        Size of the markers in the plot. Default is 6.
    deg: bool, optional
        If True, plots the Bode plot in degrees. Default is True.
    ax: np.ndarray, optional
        Array of Axes to plot on. Default is None.
    label: str, optional
        Label for the data series. Default is None.
    log: bool, optional
        If True, plots the y-axis of |Z| vs. freq in log scale. Default is False.
    alpha: float, optional
        Transparency of the markers in the plot. Default is 1.

    Returns
    -------
    np.ndarray
        Array of Axes objects of the Bode plot.
    """
    if ax is None:
        fig, ax = plt.subplots(ncols=2, figsize=(8, 3.25), layout="constrained")

    if not isinstance(ax, np.ndarray) or not isinstance(ax[0], plt.Axes):
        raise AssertionError("Incompatible 'ax': Must be two-long subplot axes. "
                             "Use `fig, ax = plt.subplots(ncols=2)` and pass 'ax'.")  # fmt: off

    # Plot magnitude |Z|
    ax[0].plot(freq, np.abs(Z), fmt, label=label, markersize=markersize, alpha=alpha)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel(r"$|Z|$")
    ax[0].grid(True)
    if label:
        ax[0].legend()

    # Plot phase φ
    ax[1].plot(
        freq, np.angle(Z, deg=deg), fmt, label=label, markersize=markersize, alpha=alpha
    )
    ax[1].set_xscale("log")
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel(rf"$\phi$ ({'deg' if deg else 'rad'})")
    ax[1].grid(True)
    if label:
        ax[1].legend()

    if log:
        ax[0].set_yscale("log")
    ax[0].figure.get_layout_engine().set(wspace=0.1)

    return ax


def plot_impedance_combo(
    freq: np.ndarray,
    Z: np.ndarray,
    *,
    fmt: str = ".-",
    markersize: int = 6,
    ax: Iterable[plt.Axes] = None,
    label: str = None,
    log: bool = False,
) -> list[plt.Axes]:
    """Plots EIS data in Nyquist and Bode plots.

    Parameters
    ----------
    freq: np.ndarray[float]
        Frequencies corresponding to the impedance data.
    Z: np.ndarray[complex]
        Impedance data.
    fmt: str, optional
        Format of the markers in the plot. Default is ".-".
    markersize: int, optional
        Size of the markers in the plots. Default is 10.
    ax: Iterable[plt.Axes], optional
        Iterable of axes (must be of length 3) to plot on. Default is None.
    label: str, optional
        Label for the plot. Default is None.
    log: bool, optional
        If True, plots the y-axis of |Z| vs. freq in log scale. Default is False.

    Returns
    -------
    list[plt.Axes]
        List of axes objects of the Nyquist and Bode plots.
    """
    if ax is None:
        fig, ax = plt.subplots(ncols=3, figsize=(12, 3.25), layout="constrained")
    else:
        msg = "Incompatible 'ax'. Use fig, ax = plt.subplots(ncols=2), and pass 'ax'."
        assert len(ax) == 3 and all(isinstance(a, Axes) for a in ax), msg

    plot_nyquist(Z=Z, label=label, ax=ax[0], fmt=fmt, markersize=markersize)
    plot_bode(freq, Z, ax=ax[1:], fmt=fmt, markersize=markersize, log=log)
    ax2 = convert_to_twinx(ax[1:])

    return np.array([ax[0], *ax2])


def plot_linKK_residuals(
    freq: np.ndarray[float],
    res_real: np.ndarray[float],
    res_imag: np.ndarray[float],
    ax: plt.Axes = None,
) -> plt.Axes:
    """Plots the residuals of the linear Kramers-Kronig validation.

    Parameters
    ----------
    freq: np.ndarray[float]
        Frequencies corresponding to the residuals.
    res_real: np.ndarray[float]
        Real part of the residuals.
    res_imag: np.ndarray[float]
        Imaginary part of the residuals.
    ax: plt.Axes, optional
        Axes to plot on. Default is None.

    Returns
    -------
    plt.Axes
        Axes object of the residuals plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(freq, res_real, label="delta Re")
    ax.plot(freq, res_imag, label="delta Im")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("delta %")
    ax.set_xscale("log")
    ax.set_title("Lin-KK validation")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend()
    return ax


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


def print_inference_results(circuits: pd.DataFrame, return_table=True) -> Styler | Table:
    """Prints the inference results in a pretty format, excluding unncessary
    columns, highlighting the best performing circuits.

    Parameters
    ----------
    circuits : pd.DataFrame
        Circuits dataframe with inference results

    Returns
    -------
    pd.io.formats.style.Styler | Table
        Styled table with the inference results
    """
    circuits = circuits.copy(deep=True)

    # Rank the circuits based on WAIC
    if "WAIC (real)" in circuits.columns:
        circuits["WAIC (sum)"] = circuits["WAIC (real)"] + circuits["WAIC (imag)"]
    else:
        circuits["WAIC (sum)"] = circuits["WAIC (mag)"] + circuits["WAIC (phase)"]
    circuits.sort_values(by=["WAIC (sum)"], ascending=True, inplace=True, ignore_index=True)

    cols_to_hide = [
        "Parameters",
        "InferenceResult",
        "converged",
        "divergences",
        "Z_pred",
        "WAIC (sum)",
        "R^2 (real)",
        "R^2 (imag)",
        "MAPE (real)",
        "MAPE (imag)",
    ]
    df = circuits.style.hide(cols_to_hide, axis=1)
    fmt = {
        "WAIC (real)" if "WAIC (real)" in circuits.columns else "WAIC (mag)": "{:.2e}",
        "WAIC (imag)" if "WAIC (imag)" in circuits.columns else "WAIC (phase)": "{:.2e}",
        "R^2 (ravg)": "{:.3f}",
        "R^2 (iavg)": "{:.3f}",
        "MAPE (ravg)": "{:.2e}",
        "MAPE (iavg)": "{:.2e}",
    }
    df.format(fmt)

    # Create a rich Table to pretty print the results
    table = Table(title="Inference results", show_header=True, header_style="bold")

    # Add columns to the table
    columns = [
        "Circuit",
        "WAIC (real)" if "WAIC (real)" in circuits.columns else "WAIC (mag)",
        "WAIC (imag)" if "WAIC (imag)" in circuits.columns else "WAIC (phase)",
        "R2 (re)",
        "R2 (im)",
        "MAPE (re)",
        "MAPE (im)",
        "Np",
    ]
    for column in columns:
        table.add_column(column, justify="right")

    # Fill the table with data
    for i, row in df.data.iterrows():
        table.add_row(
            row["circuitstring"],
            f"{row['WAIC (real)' if 'WAIC (real)' in circuits.columns else 'WAIC (mag)']:.2e}",
            f"{row['WAIC (imag)' if 'WAIC (imag)' in circuits.columns else 'WAIC (phase)']:.2e}",
            f"{row['R^2 (ravg)']:.3f}",
            f"{row['R^2 (iavg)']:.3f}",
            f"{row['MAPE (ravg)']:.2e}",
            f"{row['MAPE (iavg)']:.2e}",
            f"{row['n_params']}",
        )

    return table if return_table else df


def override_mpl_colors(override_named_colors: bool = True):
    """Override matplotlib's default colors with Flexoki colors."""
    # Define the Flexoki-Light color scheme based on the provided table
    # Original sequence: red, orange, yellow, green, cyan, blue, purple, magenta
    flexoki_light_colors = {
        "red": mcolors.to_rgb("#D14D41"),
        "blue": mcolors.to_rgb("#4385BE"),
        "green": mcolors.to_rgb("#879A39"),
        "orange": mcolors.to_rgb("#DA702C"),
        "purple": mcolors.to_rgb("#8B7EC8"),
        "yellow": mcolors.to_rgb("#D0A215"),
        "cyan": mcolors.to_rgb("#3AA99F"),
        "magenta": mcolors.to_rgb("#CE5D97"),
    }

    # Override default named colors
    if override_named_colors:
        for k, v in flexoki_light_colors.items():
            mcolors.ColorConverter.cache[(k, None)] = v
            mcolors.ColorConverter.colors[k] = v
            mcolors.ColorConverter.cache[(k[0], None)] = v  # k[0] is the short name
            mcolors.ColorConverter.colors[k[0]] = v

    # Define the Flexoki-Light style
    flexoki_light_style = {
        "axes.prop_cycle": mpl.cycler(
            color=list(flexoki_light_colors.values()),
            linestyle=["-", "--", "-.", ":"] * 2,
        ),
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
    use_arviz: bool = True,
    use_seaborn: bool = True,
    use_flexoki: bool = True,
    console_width: int = 100,
):
    """Modifies the default arviz/matplotlib config for prettier plots.

    Parameters
    ----------
    use_arviz: bool, optional
        If True, use arviz's default plotting style. Default is True.
    use_seaborn: bool, optional
        If True, use seaborn's default plotting style. Default is True.
    use_flexoki: bool, optional
        If True, use Flexoki's default plotting style. Default is True.
    console_width: int, optional
        Width of the console in characters. Only set when in VS Code
        Interactive mode, otherwise automaticall deterimined. Default is 100.
    """
    # Arviz
    if use_arviz:
        arviz.style.use("arviz-viridish")

    # Seaborn
    if use_seaborn:
        sns.set_style("ticks")

    # Matplotlib
    label_size = 11
    tick_size = label_size - 1
    ax_title_size = label_size + 1
    fig_title_size = ax_title_size + 2
    legend_size = label_size - 1

    plt.rcParams["font.family"] = "sans-serif"
    fonts = ["Helvetica", "Arial", "Verdana", "Tahoma", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = fonts
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["xtick.labelsize"] = tick_size
    plt.rcParams["ytick.labelsize"] = tick_size
    plt.rcParams["axes.labelsize"] = label_size
    plt.rcParams["axes.titlesize"] = ax_title_size
    plt.rcParams["figure.titlesize"] = fig_title_size
    plt.rcParams["legend.fontsize"] = legend_size
    plt.rcParams["legend.frameon"] = False

    # Flexoki colors
    if use_flexoki:
        override_mpl_colors()

    # Set up Jupyter notebook
    try:
        import IPython
        import matplotlib_inline.backend_inline

        # IPython.display.set_matplotlib_formats("retina")
        matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
    except ImportError:
        pass

    # Set rich console width when in VS Code Interactive Window
    if os.getenv("VSCODE_PID") is not None:
        rich.get_console().width = console_width


def show_nticks(ax: plt.Axes, x: bool = True, y: bool = False, n: int = 10):
    """In-place modifies Matplotlib axes to show only ``n`` ticks.

    Parameters
    ----------
    ax: plt.Axes
        Axes to modify.
    x: bool, optional
        If True, applies the filter to the x-axis. Default is True.
    y: bool, optional
        If True, applies the filter to the y-axis. Default is False.
    n: int, optional
        Number of ticks to show. Default is 10.
    """
    if x:
        xticks = ax.xaxis.get_major_ticks()
        if len(xticks) > n:
            ax.xaxis.set_major_locator(plt.MaxNLocator(n, steps=[1, 2, 5, 10]))
    if y:
        yticks = ax.yaxis.get_major_ticks()
        if len(yticks) > n:
            ax.yaxis.set_major_locator(plt.MaxNLocator(n, steps=[1, 2, 5, 10]))


def convert_to_twinx(ax: np.ndarray[plt.Axes]) -> tuple[plt.Axes]:
    """Convert a 2-column subplots ax object into a single-column with twinx.
    Left axis is always red, right axis is always blue.

    Parameters
    ----------
    ax : numpy.ndarray
        Array of 2-column subplot Axes objects.

    Returns
    -------
    tuple[plt.Axes]
        Two Axes objects with the same data from the original subplots, but
        now combined into a single-column plot with twinx.
    """
    # Check that the input is a 2-column axes object
    if ax.size != 2:
        raise ValueError("Input 'ax' object must have 2 subplots.")

    # Extract the first subplot as the base for the new single-column plot
    ax1 = ax[0]

    # Get the data and labels from the second column (right plot)
    ax2 = ax1.twinx()  # Create a twin axis sharing the x-axis

    # Copy over the data from the second subplot to the right axis
    for line in ax[1].get_lines():
        ax2.plot(
            line.get_xdata(),
            line.get_ydata(),
            color="b",  # Set twin axis plot to blue
            label=line.get_label(),
            linestyle=line.get_linestyle(),
            marker=line.get_marker(),
        )

    # Set labels and titles from the original plots
    ax1.set_xlabel(ax[0].get_xlabel())
    ax1.set_ylabel(ax[0].get_ylabel(), color="r")  # Set left axis label to red
    ax2.set_ylabel(ax[1].get_ylabel(), color="b")  # Set right axis label to blue

    # Remove the right subplot to clean up the figure
    ax[1].remove()

    # Turn off grid for the 2nd axis
    ax2.grid(False)

    # Set tick parameters to match the axis colors
    ax1.tick_params(axis="y", colors="r")  # Left axis red ticks
    ax2.tick_params(axis="y", colors="b")  # Right axis blue ticks

    # Ensure the lines match the designated axis colors
    for l1 in ax[0].get_lines():
        l1.set_color("r")  # Set left axis lines to red
    for l2 in ax2.get_lines():
        l2.set_color("b")  # Set right axis lines to blue

    return ax1, ax2

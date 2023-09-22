import arviz
import matplotlib.pyplot as plt


def plot_eis_data(Re_Z, Im_Z, freq, saveto=None):
    """Plot EIS data in Nyquist and Bode plots."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Nyquist plot
    axes[0].scatter(Re_Z, -Im_Z, s=1.5)
    axes[0].set_xlabel(r"$Re(Z) / \Omega$")
    axes[0].set_ylabel(r"$-Im(Z) / \Omega$")
    axes[0].set_title("Non-filtered")

    # Bode plot (magnitude) <- Re(Z)
    ax1 = axes[1]
    ax1.scatter(freq, Re_Z, s=1.5, color='blue', label=r'$Re(Z)$')
    ax1.set_xscale("log")
    ax1.set_xlabel("freq (Hz)")
    ax1.set_ylabel(r"$Re(Z) / \Omega$")
    ax1.legend(loc='upper left')

    # Bode plot (phase) <- Im(Z)
    ax2 = ax1.twinx()  # instantiate a second y-axis sharing the same x-axis
    ax2.scatter(freq, -Im_Z, s=1.5, color='red', label=r'$-Im(Z)$')
    ax2.set_ylabel(r"$-Im(Z) / \Omega$")
    ax2.legend(loc='upper right')

    axes[1].set_title("Non-filtered")

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
    ax.legend()

    if saveto is not None:
        fig.savefig(saveto, dpi=300)
    
    return fig, ax


def set_plotting_style(use_arviz=True) -> None:
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

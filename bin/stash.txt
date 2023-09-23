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


def plot_eis(
    freq: np.ndarray,
    impedance: np.ndarray = "",
    reals: np.ndarray = "",
    imags: np.ndarary = "",
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
        ax.scatter(np.log10(frequencies), magnitude, c="b", label="magnitude")
        ax.set_xlabel("log(freq)")
        ax.set_ylabel("magnitude")
        ax.set_title("Bode plot")
        # Phase plot
        ax2 = ax.twinx()
        ax2.scatter(np.log10(frequencies), phase, c="r", alpha=0.7, marker="v", label="phase")
        ax2.set_ylabel("Phase")
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        fig.show()

    return fig

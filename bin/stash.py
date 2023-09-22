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

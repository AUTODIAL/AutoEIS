import numpy as np
import autoeis as ae

def update_circuits_with_posterior_scores(
    circuits_df,
    results,
    freq,
    Z,
    score_fn,
    threshold=79,
    save_plot=True,
    verbose=True
):
    """
    Integrate posterior quality evaluation into circuits DataFrame.

    Args:
        circuits_df (pd.DataFrame): Original circuits table.
        results (list): List of AutoEIS InferenceResult.
        freq (np.ndarray): Frequency array.
        Z (np.ndarray): Impedance data.
        score_fn (callable): Your posterior scoring function, should return (total_score, bad_param_list).
        threshold (int): Score threshold for bad parameter filtering.
        save_plot (bool): Whether to generate KDE/histogram plots.
        verbose (bool): Whether to print per-circuit result.

    Returns:
        pd.DataFrame: Updated circuits DataFrame.
    """
    scores = []
    bad_params_all = []

    if verbose:
        print("Evaluating posterior distributions...\n")

    for result in results:
        if result.converged:
            if verbose:
                ae.visualization.print_summary_statistics(result.mcmc, result.circuit)

            total_score, bad_params = score_fn(
                result.mcmc, threshold=threshold, save_plot=save_plot
            )

            if verbose:
                print(f"Circuit: {result.circuit}")
                print(f"Score: {total_score:.2f}")
                print(f"Bad parameters: {bad_params if bad_params else 'None'}\n")
        else:
            total_score, bad_params = 0, ["Not Converged"]
            if verbose:
                print(f"Circuit {result.circuit} did not converge.\n")

        scores.append(total_score)
        bad_params_all.append(", ".join(bad_params) if bad_params else "None")

    # ---------- Write inference results to new DataFrame columns ----------
    circuits_df = circuits_df.copy()
    circuits_df["InferenceResult"] = results
    circuits_df["Posterior Score"] = scores
    circuits_df["Bad Parameters"] = bad_params_all

    # ---------- Compute fitness metrics using AutoEIS ----------
    circuits_df = ae.core.compute_fitness_metrics(circuits_df, freq, Z)

    # ---------- Custom ranking logic ----------
    # True if the circuit contains bad parameters, False otherwise
    circuits_df["HasBad"] = circuits_df["Bad Parameters"].str.strip().str.lower() != "none"

    # Sort so that circuits without bad parameters come first,
    # and within each group, sort by Posterior Score (descending)
    circuits_df.sort_values(
        by=["HasBad", "Posterior Score"],
        ascending=[True, False],
        inplace=True,
        ignore_index=True
    )

    # Assign consecutive ranks: 1, 2, 3, ...
    circuits_df["Posterior Score Rank"] = np.arange(1, len(circuits_df) + 1)

    # Optional: Remove the helper column
    circuits_df.drop(columns=["HasBad"], inplace=True)

    return circuits_df
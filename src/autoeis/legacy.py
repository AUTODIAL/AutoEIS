import logging
import os
import time

import arviz as az
import dill
import jax.numpy as jnp  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, Predictive
from tqdm.auto import tqdm

import autoeis.visualization as viz
from autoeis import utils

log = logging.getLogger(__name__)


def circuit_to_function(circuit: str, use_jax=True) -> callable:
    """Converts a circuit string to a callable function."""
    circuit_df = pd.DataFrame([circuit], columns=["circuitstring"])
    eqn = generate_mathematical_expression(circuit_df)["Mathematical expressions"][0]
    if use_jax:
        eqn = eqn.replace("np", "jnp")

    def _fn(X, F):
        assert utils.count_parameters(circuit) == len(X), "Invalid number of parameters."
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
                circuit = circuit.replace(test_results_2[m], "(1/(X*(2*1j*np.pi*F)**(Y)))")

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


def ohmic_resistance_filter_legacy(
    circuits: pd.DataFrame, ohmic_resistance: float, rtol=0.5
) -> pd.DataFrame:
    """Filters the circuits whose ohmic resistance doesn't match a desired value."""
    for i in range(len(circuits["circuitstring"])):
        # Find the series elements
        series_circuit = find_series_elements(circuit=circuits["circuitstring"][i])
        # Find the series resistors
        find_R = re.compile(r"R[0-9]")
        series_resistors = find_R.findall(series_circuit)
        # Initiate a list to store series resistors' values for future comparison
        R_values_list = []
        for j in range(len(series_resistors)):
            value_R_p = re.compile(f"{series_resistors[j]} = [0-9]*\.[0-9]*")
            values_R_withid = value_R_p.findall("".join(circuits["Resistors"][i]))
            value_R_p2 = re.compile(r"[0-9]*\.[0-9]*")
            for k in range(len(values_R_withid)):
                R_value = value_R_p2.findall(values_R_withid[k])
                R_values_list.append(R_value)
        if R_values_list == []:
            circuits.drop([i], inplace=True)
        else:
            value_identify_list = []
            for m in range(len(R_values_list)):
                if (
                    float(R_values_list[m][0]) < ohmic_resistance * 0.5
                    or float(R_values_list[m][0]) > ohmic_resistance * 1.5
                ):
                    value_identify_list.append(False)
                else:
                    value_identify_list.append(True)
            if True not in value_identify_list:
                circuits.drop([i], inplace=True)

    circuits.reset_index(drop=True, inplace=True)
    return circuits


def apply_heuristic_rules_legacy(
    circuits: pd.DataFrame, ohmic_resistance: float
) -> pd.DataFrame:
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


def model_evaluation(results):
    # TODO: Relying on the column index is not a good idea, refactor.
    evaluation_results = results[results.columns[[0, 17, 18, 19, 20, 23, 25]]]

    # FIXME: Remove next line once confirmed that `loc` is correctly used.
    # evaluation_results["Consistency"] = pd.to_numeric(evaluation_results["Consistency"], errors="coerce")
    evaluation_results.loc[:, "Consistency"] = pd.to_numeric(
        evaluation_results["Consistency"], errors="coerce"
    )
    evaluation_results.loc[evaluation_results["Consistency"].isna(), "Consistency"] = np.inf

    def absdiff(x):
        return np.inf if np.isinf(x) else np.abs(x - 1)

    def custom_sort(x):
        return -1000 if x == "F" else x

    # FIXME: Remove next line once confirmed that `loc` is correctly used.
    # evaluation_results["Consistency"] = evaluation_results["Consistency"].apply(absdiff)
    evaluation_results.loc[:, "Consistency"] = evaluation_results["Consistency"].apply(
        absdiff
    )
    # FIXME: Remove next line once confirmed that `loc` is correctly used.
    # evaluation_results["Posterior_shape"] = evaluation_results["Posterior_shape"].apply(custom_sort)
    evaluation_results.loc[:, "Posterior_shape"] = evaluation_results[
        "Posterior_shape"
    ].apply(custom_sort)

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


def perform_bayesian_inference(
    eis_data: pd.DataFrame,
    ecms: pd.DataFrame,
    saveto: str = None,
    plot: bool = False,
    draw_ecm=False,
    seed: int = None,
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

    # NOTE: R2, MSE, RMSE, and MAPE are calculated between GT and GEP, prior to BI
    # Create a list to store the R2 value of each ECM
    R2_list = []
    R2_real_list = []
    R2_imag_list = []
    # Create a list to store fitting quality metrics of each ECM
    MSE_list = []
    RMSE_list = []
    MAPE_list = []
    # Create a list to store simulated ECM data
    # NOTE: ...still prior to BI
    ECMs_data = []
    # NOTE: Now, we're in the BI territory
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

    # Set the seed for reproducibility (if not set, use current time in nanoseconds)
    seed = seed or time.time_ns() % 2**32
    rng_key = random.PRNGKey(seed)
    rng_key, rng_subkey = random.split(rng_key)

    # BI parts
    values = ecms["Variables_values"]
    names = ecms["Variables_names"]
    expressions_strs = ecms["Mathematical expressions"]
    circuit_names = ecms["Combined Circuits"]

    for i in tqdm(range(len(ecms["Combined Circuits"])), disable=True):
        circuit_name_i = circuit_names[i]
        try:
            value_i = list(map(float, values[i]))
        except ValueError:
            value_i = eval(values[i])
        name_i = names[i]
        expression_str_i = expressions_strs[i].replace("np.", "jnp.")

        def function_i(X, F):
            return eval(expression_str_i)

        log.info(f"Circuit {i}: {circuit_name_i}")
        log.info(f"Elements: ({name_i})\nValues: ({value_i})")

        if plot and draw_ecm:
            viz.draw_circuit(circuit_name_i)

        Zsim = function_i(value_i, freq)
        ECMs_data.append(Zsim)

        log.info("Julia circuit's fitting")

        r2_value = utils.r2_score(Z, Zsim)
        log.info(f"R² = {r2_value}")
        R2_list.append(r2_value)

        r2_real = utils.r2_score(Zreal, Zsim.real)
        log.info(f"R² (Re) = {r2_real}")
        R2_real_list.append(r2_real)
        r2_imag = utils.r2_score(Zimag, Zsim.imag)
        log.info(f"R² (Im) = {r2_imag}")
        R2_imag_list.append(r2_imag)

        MSE_value = utils.mse_score(Z, Zsim)
        log.info(f"MSE = {MSE_value}")
        MSE_list.append(MSE_value)

        RMSE_value = utils.rmse_score(Z, Zsim)
        log.info(f"RMSE = {RMSE_value}")
        RMSE_list.append(RMSE_value)

        MAPE_value = utils.mape_score(Z, Zsim)
        log.info(f"MAPE = {MAPE_value}")
        MAPE_list.append(MAPE_value)

        if plot:
            fig, ax = plt.subplots()
            viz.plot_nyquist(Z=Zsim, fmt="o", color="r", label="simulated", ax=ax)
            viz.plot_nyquist(Z=Z, fmt="-", color="b", label="experiment", ax=ax)
            if saveto is not None:
                fpath = os.path.join(saveto, f"nyquist_simulated_{i}.png")
                fig.savefig(fpath, dpi=300)

        def model_i(values=value_i, func=function_i, true_data=eis_data):
            true_freq = np.asarray(true_data["freq"])
            true_Zreal = np.asarray(true_data["Zreal"])
            true_Zimag = np.asarray(true_data["Zimag"])
            # Define sampling variables
            variables_list = []
            for j in range(len(name_i)):
                name = name_i[j]
                value = values[j]
                if "n" in name:
                    free_variable = numpyro.sample(f"{name}", dist.Uniform(0, 1))
                    variables_list.append(free_variable)
                else:
                    free_variable = numpyro.sample(f"{name}", dist.LogNormal(2.5, 1.7))
                    real_variable = value * free_variable
                    variables_list.append(real_variable)
            # Define error term
            true_obs = true_Zreal + true_Zimag * 1j
            mu = func(variables_list, true_freq)
            error_term = numpyro.sample("err", dist.HalfNormal())
            numpyro.sample("obs", dist.HalfNormal(error_term), obs=abs(true_obs - mu))

        # ?: Why 200? These are just to plot the prior distributions
        prior_predictive = Predictive(model_i, num_samples=200)
        prior_prediction = prior_predictive(rng_subkey)
        rng_key, rng_subkey = random.split(rng_key)
        Prior_predictions.append(prior_prediction)

        # ?: Why 10,000?
        # NOTE: use num_chains > 1 to enable parallel sampling
        kernel = NUTS(model_i, target_accept_prob=0.8)
        num_samples = 10000
        mcmc_i = MCMC(kernel, num_warmup=1000, num_samples=num_samples, num_chains=1)
        mcmc_i.run(rng_subkey, values=value_i, func=function_i, true_data=eis_data)

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
        # ?: Why 100? This has to do the with the number of samples done in prior_predictive
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
                posterior_HDI.figure.savefig(
                    f"posterior_predictions_with_HDI_{i}.png", dpi=300
                )

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
            sep_mape_real = utils.mape_score(Zreal, BI_data.real)
            sep_mape_real_list.append(sep_mape_real)
            sep_r2_real = utils.r2_score(Zreal, BI_data.real)
            sep_r2_real_list.append(sep_r2_real)

        avg_mape_real = np.array(sep_mape_real_list).mean()
        avg_r2_real = np.array(sep_r2_real_list).mean()
        log.info(f"Posterior fit (real): MAPE = {avg_mape_real}, R² = {avg_r2_real}")
        Posterior_r2_real.append(avg_r2_real)
        Posterior_mape_real.append(avg_mape_real)

        if plot:
            ax.plot(
                freq,
                BI_data.real,
                marker=".",
                ms=15,
                color="grey",
                alpha=0.5,
                label="predictive",
            )
            ax.plot(
                freq, Zreal, "--", marker="o", c="b", alpha=0.9, ms=8, label="ground truth"
            )
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
            sep_mape_imag = utils.mape_score(Zimag, BI_data.imag)
            sep_mape_imag_list.append(sep_mape_imag)
            sep_r2_imag = utils.r2_score(Zimag, BI_data.imag)
            sep_r2_imag_list.append(sep_r2_imag)

        avg_mape_imag = np.array(sep_mape_imag_list).mean()
        avg_r2_imag = np.array(sep_r2_imag_list).mean()
        log.info(f"Posterior fit (imag): MAPE = {avg_mape_imag}, R² = {avg_r2_imag}")
        Posterior_r2_imag.append(avg_r2_imag)
        Posterior_mape_imag.append(avg_mape_imag)
        if plot:
            ax.plot(
                freq,
                -BI_data.imag,
                marker=".",
                ms=15,
                color="grey",
                alpha=0.5,
                label="predictive",
            )
            ax.plot(
                freq, -Zimag, "--", marker="o", c="b", alpha=0.9, ms=8, label="ground truth"
            )
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
            sep_mape = utils.mape_score(Z, BI_data)
            sep_mape_list.append(sep_mape)
            sep_r2 = utils.r2_score(Z, BI_data)
            sep_r2_list.append(sep_r2)

        # ?: Why commented out? because mse doesn't make much sense for complex values
        # ?: What other metrics can we use to quantify error in complex values?
        # avg_mse = np.array(sep_mse_list).mean()
        avg_mape = np.array(sep_mape_list).mean()
        avg_r2 = np.array(sep_r2_list).mean()
        log.info(f"Posterior fit: MAPE = {avg_mape}, R² = {avg_r2}")
        Posterior_r2.append(avg_r2)
        Posterior_mape.append(avg_mape)

        if plot:
            ax.plot(
                BI_data.real,
                -BI_data.imag,
                marker=".",
                ms=15,
                color="grey",
                alpha=0.5,
                label="predictions",
            )
            ax.plot(
                Zreal,
                -Zimag,
                "--",
                marker="o",
                c="b",
                alpha=0.9,
                ms=8,
                label="grount truth",
            )
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
        # NOTE: We're making sure the figure is not blank!
        if any(
            len(result[0].lines[0].get_xydata().T[0]) == 2 for result in posterior_dist[:]
        ):
            posterior_mark = "F"
        else:
            # NOTE: What other criteria can we use to evaluate the posterior distribution?
            posterior_mark = posterior_evaluation(posterior_dist)
        posterior_shape.append(posterior_mark)

        # NOTE: Another metric to evaluate the posterior distribution (consistency)
        r_hats = []
        for i in range(len(name_i)):
            r_hats.append(
                summary(mcmc_i.get_samples(), prob=0.94, group_by_chain=False)[
                    f"{name_i[i]}"
                ]["r_hat"]
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

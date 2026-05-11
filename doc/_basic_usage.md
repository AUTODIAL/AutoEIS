# Basic usage

:::{warning}
The envelope function, `perform_full_analysis` has some issues since it was doing too much all at once. For now, we've deprecated the function until it's made robust. We recommend using the step-by-step approach since it gives more control. That said, since a one-stop-shop function is what many users, especially experimentlists, would like, we're working on making it robust. We'll update this page once the function is ready.
:::

To use AutoEIS, you can either perform the circuit generation and Bayesian inference step by step or use the `perform_full_analysis` function to perform the whole process automatically. The following is a minimal example of how to use the `perform_full_analysis` function.

```python
import autoeis as ae

# Load and visualize the test dataset
freq, Z = ae.io.load_test_dataset()
ae.visualization.plot_impedance_combo(freq, Z)

# Perform automated EIS analysis
circuits = ae.perform_full_analysis(freq, Z, iters=24, parallel=True)

# Print summary of the inference for each circuit model
for i, row in circuits.iterrows():
    circuit = row["circuit"]
    mcmc = row["InferenceResult"].mcmc
    if row["converged"]:
        ae.visualization.print_summary_statistics(mcmc, circuit)

# Print summary of all circuit models
ae.visualization.print_inference_results(circuits)
```

## Posterior quality diagnostics

After Bayesian inference, AutoEIS can score each posterior parameter
distribution for Gaussian-like behavior. The diagnostic combines Q-Q plot
linearity, skewness, and KDE mode count, then labels each parameter as
`Normal`, `Multi-Peak`, `Left-Skewed`, `Right-Skewed`, `Touch-Bound`, or
`Invalid`.

```python
result = results[0]

# Return one row per inferred parameter.
quality = result.assess_posterior_quality(threshold=79)
print(quality[["parameter", "score", "shape", "status"]])

# The same API is also available as a module-level function.
quality = ae.diagnostics.assess_posterior_quality(result.mcmc, threshold=79)

# Optional: save KDE + histogram plots for each parameter.
quality = result.assess_posterior_quality(
    threshold=79,
    save_plots=True,
    save_dir="posterior_quality",
)

# Add posterior scores back to a circuits dataframe and rank models.
circuits = ae.diagnostics.add_posterior_quality(circuits, results, threshold=79)
```

## ECM topology vectors

ECM topology utilities are available from `ae.ecm`. They operate on the same
AutoEIS circuit strings, but produce normalized topology trees and compact
vectors for tracking or comparing model structure.

```python
circuit = "R1-[P2,R3]-C4"

# Simplify mergeable R/C/L branches without changing the existing parser API.
simplified = ae.ecm.simplify_ecm("R1-R2-[C1-C2,R3]-R4")

# Convert ECM topology to an integer vector and back.
ecm_vector = ae.ecm.ecm_to_vector(circuit)
roundtrip = ae.ecm.vector_to_ecm(ecm_vector)

# Convert numeric EIS values with matching series/parallel syntax.
eis_vector = ae.ecm.eis_to_vector("10-[2.5,30]-1e-6")
values = ae.ecm.extract_eis_values(eis_vector)

# Add vectors to a circuits dataframe.
circuits = ae.ecm.add_ecm_vectors(circuits)
```

:::{seealso}
While the above example should work out of the box, it is recommended to use AutoEIS in a step by step fashion to have more control over the analysis process. Furthermore, you'll learn more about the inner workings of AutoEIS this way. An example notebook that demonstrates how to use AutoEIS in more details can be found [here](https://github.com/AUTODIAL/AutoEIS/blob/develop/examples/autoeis_demo.ipynb).
:::

:::{note}
Apart from the functions used in the example notebook, there are more functionalities in AutoEIS that are not yet documented. Until we add more examples on how to use these features, you can find the full list of functions in the [API reference](modules) section.
:::

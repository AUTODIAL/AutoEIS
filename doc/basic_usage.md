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

:::{seealso}
While the above example should work out of the box, it is recommended to use AutoEIS in a step by step fashion to have more control over the analysis process. Furthermore, you'll learn more about the inner workings of AutoEIS this way. An example notebook that demonstrates how to use AutoEIS in more details can be found [here](https://github.com/AUTODIAL/AutoEIS/blob/develop/examples/autoeis_demo.ipynb).
:::

:::{note}
Apart from the functions used in the example notebook, there are more functionalities in AutoEIS that are not yet documented. Until we add more examples on how to use these features, you can find the full list of functions in the [API reference](modules) section.
:::

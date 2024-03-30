# Basic usage

To use AutoEIS, you can either perform the circuit generation and Bayesian inference step by step or use the `perform_full_analysis` function to perform the whole process automatically. The following is a minimal example of how to use the `perform_full_analysis` function.

```python
import autoeis as ae

# Load and visualize the test dataset
freq, Z = ae.io.load_test_dataset()
ae.visualization.plot_impedance_combo(freq, Z)

# Perform automated EIS analysis
circuits = ae.perform_full_analysis(freq, Z, iters=24, parallel=True)
print(circuits)

# Print summary of the results
mcmcs, status = circuits["MCMC (chain)"], circuits["MCMC (status)"]
for mcmc, stat, circuit in zip(mcmcs, status, circuits.circuitstring):
    if stat == 0:
        ae.visualization.print_summary_statistics(mcmc, circuit)
```

:::{seealso}
While the above example should work out of the box, it is recommended to use AutoEIS in a step by step fashion to have more control over the analysis process. Furthermore, you'll learn more about the inner workings of AutoEIS this way. An example notebook that demonstrates how to use AutoEIS in more details can be found [here](https://github.com/AUTODIAL/AutoEIS/blob/develop/examples/autoeis_demo.ipynb).
:::

:::{note}
Apart from the functions used in the example notebook, there are more functionalities in AutoEIS that are not yet documented. Until we add more examples on how to use these features, you can find the full list of functions in the [API reference](modules) section.
:::

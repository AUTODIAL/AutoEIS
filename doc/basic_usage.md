# Basic usage
To use AutoEIS, you can either perform the circuit generation and Bayesian inference step by step or use the `perform_full_analysis` function to perform the whole process automatically. The following is an example of how to use the `perform_full_analysis` function.

### Import AutoEIS

```python
import autoeis as ae
```

### Load EIS measurements

```python
# Load test dataset shipped with AutoEIS
Z, freq = ae.io.load_test_dataset()
```

### Automated EIS analysis

```python
circuits = ae.perform_full_analysis(Z, freq, iters=100, parallel=True)
print(circuits)
```

An example notebook that demonstrates how to use AutoEIS in more details can be found [here](https://github.com/AUTODIAL/AutoEIS/blob/develop/examples/autoeis_demo.ipynb).

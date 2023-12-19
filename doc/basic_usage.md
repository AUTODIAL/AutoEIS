# Basic usage
To use AutoEIS, you can either perform the circuit generation and Bayesian inference step by step or use the `perform_full_analysis` function to perform the whole process automatically. The following is an example of how to use the `perform_full_analysis` function.

### Import required modules

```python
import numpy as np
import autoeis as ae
```

### Load EIS measurements

```python
path_data = "assets/test_data.txt"
freq, Zreal, Zimag = np.loadtxt(path_data, skiprows=1, unpack=True, usecols=(0, 1, 2))
# Convert to complex impedance (the file contains -Im(Z) hence the minus sign)
Z = Zreal - 1j*Zimag
```

### Automated EIS analysis

```python
circuits = ae.perform_full_analysis(Z, freq, iters=100, parallel=True)
print(circuits)
```

An example notebook that demonstrates how to use AutoEIS in more details can be found [here](https://github.com/AUTODIAL/AutoEIS/blob/develop/examples/autoeis_demo.ipynb).

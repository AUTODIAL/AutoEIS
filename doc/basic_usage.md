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
df = ae.io.load_eis_data(path_data)
```

### Fetch frequency and impedance

```python
freq = df["freq/Hz"].to_numpy()
Re_Z = df["Re(Z)/Ohm"]).to_numpy()
Im_Z = -df["-Im(Z)/Ohm"].to_numpy()
Z = Re_Z + Im_Z * 1j
```

### Automated EIS analysis

```python
results = ae.perform_full_analysis(Z, freq, iters=100, saveto="results", draw_ecm=True)
```

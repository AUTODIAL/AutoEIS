# AutoEis.py
------------
AutoEis is a Python package to automatically propose statistical plausible equivalent circuit models (ECMs) for electrochemical impedance spectroscopy (EIS) analysis. The package is designed for researchers and practitioners in the fields of electrochemical analysis, including but not limited to explorations of electrocatalysis, battery design, and investigations of material degradations.

Please be aware that the current version is only the beta test and has not been formally realized. If you find any bug and any suggestions, please [file an issue](https://github.com/AUTODIAL/Auto_Eis/issues) or directly [pull requests](https://github.com/AUTODIAL/Auto_Eis/pulls). We would really appreciate any contributions from our commmunity. 

## Installation
---------------
The easiest way to install this package is using pip install from [pypi](https://pypi.org/project/AutoEis/)
```bash
pip install AutoEis
```

## Dependencies
---------------
The circuits generation is realized based on the julia package [equivalentcircuit.jl](https://github.com/MaximeVH/EquivalentCircuits.jl) designed by MaximeVH. It requires a installation of [julia programming language](https://julialang.org/)

AutoEis requires:
-   **Python programming language (>=3.7, <3.11)**
- - NumPy (>=1.20)
- - Matplotlib (>=3.3)
- - Pandas (>=1.1)
- - impedance (>=1.4)
- - regex (>=2.2)
- - arviz (>=2.2.1)
- - numpyro (=0.10.1)
- - dill (>=0.3.4)
- - PyJulia (>=0.5.7)
- - IPython (>=7.19.0)
- - jax (>=0.3.9)

*Note: If you operating system is Windows, after install jax, it is required to install jaxlib as well. However, the installation of jaxlib on Windows is not pypi-supported. You may need to visit [this repository](https://github.com/cloudhan/jax-windows-builder) to find the version corresponding to your python version, and then install it using wheel.

-   **Julia programming language (>=1.7.0)**
- - equivalentcircuit (>=0.1.3)
- - CSV
- - DataFrame
- - Pandas
- - PyCall
- - DelimitedFiles
- - StringEncodings

## Workflow
------------
The schematic workflow of AutoEis is shown below:
![Workflow](https://github.com/AUTODIAL/Auto_Eis/blob/main/AutoEis_workflow.png)
It contains: data pre-processing, ECM generation, circuit post-filtering, Bayesian inference and model evaluation process. Through this workflow, AutoEis can prioritize the statistical optimal ECM and also retains suboptimal models with lower priority for subsequent expert inspection.

## Usage
-------------
To enable the interaction between python and julia, please set the julia executable pathway at first. The common pathway of the Julia executable path depends on the operating system you are using. Here are the common default locations for each supported OS:

- Windows: *C:\Users\<username>\AppData\Local\Julia-<version>\bin*
- macOS: */Applications/Julia-<version>.app/Contents/Resources/julia/bin*
- Linux: */usr/local/julia-<version>/bin*

Please note that <version> refers to the specific version of Julia you have installed, and <username> is the name of the current user on Windows. To confirm the location of your Julia executable path, you can open a command prompt or terminal and enter the command which julia (for Unix-based systems) or where julia (for Windows). This will display the full path of the Julia executable file.

```bash
# import AutoEIS and its dependencies
import AutoEis as ae

# define the path of julia program
j = ae.set_julia (r"D:\Julia-1.7.2\bin\julia.exe")
```
Then by calling the function `initialize_julia()`, AutoEis will automatically install julia's dependencies. *(This step is only required at your first time using AutoEis)*
```bash
# initialize the julia environment and download the dependencies
ae.initialize_julia()
```
Now you are good to load your data and perform all analysis with one function
```bash
# set the parameter of plots
ae.set_parameter()

# Load the data
data_path = "Test_data.txt"
df = ae.load_data(data_path)
frequencies = ...
reals = ...
imags = ...
measurements = reals + 1j*imags

# Perform automated ECM generation and evaluation
ae.EIS_auto(impedance=measurements,freq=frequencies,data_path=data_path,iter_number=100,plot_ECM=False)
```
- `impedance` : the measured electrochemical impedance
- `freq`: the measured frequencies
- `data_path`: the pathway of loaded data (this path will be used for the results storage)
- `iter_number`: the numbers of ECM generation to be performed (default = 100)
- `plot_ECM`: to plot ECM or not (*Note: To enable this parameter, a [LaTex compiler](https://www.latex-project.org/get/) is required.*) 
  
An example that demonstrate how to use AutoEis is attached [here](https://github.com/AUTODIAL/Auto_Eis/blob/main/Example_AutoEIS.ipynb). 

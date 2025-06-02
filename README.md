[![DOI](https://joss.theoj.org/papers/10.21105/joss.06256/status.svg)](https://doi.org/10.21105/joss.06256)
![example workflow](https://github.com/AUTODIAL/AutoEIS/actions/workflows/nightly.yml/badge.svg)

> [!NOTE]
> AutoEIS is now published in the Journal of Open Source Software (JOSS). You can find the paper [here](https://doi.org/10.21105/joss.06256). If you find AutoEIS useful, please consider citing it in your work.
>
> > Sadeghi et al., (2025). AutoEIS: Automated equivalent circuit modeling from electrochemical impedance spectroscopy data using statistical machine learning. _Journal of Open Source Software_, 10(109), 6256, https://doi.org/10.21105/joss.06256
>
> > Zhang, Runze, et al. "Editors’ choice—AutoEIS: automated bayesian model selection and analysis for electrochemical impedance spectroscopy." _Journal of The Electrochemical Society_ 170.8 (2023): 086502. https://doi.org/10.1149/1945-7111/aceab2

> [!TIP]
> _Want to get notified about major announcements/new features?_ Please click on "Watch" -> "Custom" -> Check "Releases". Starring the repository alone won't notify you when we make a new release. This is particularly useful since we're actively working on adding new features/improvements to AutoEIS. Currently, we might issue a new release every month, so rest assured that you won't be spammed.

# AutoEIS

## What is AutoEIS?

AutoEIS (Auto ee-eye-ess) is a Python package that automatically proposes statistically plausible equivalent circuit models (ECMs) for electrochemical impedance spectroscopy (EIS) analysis. The package is designed for researchers and practitioners in the fields of electrochemical analysis, including but not limited to explorations of electrocatalysis, battery design, and investigations of material degradation.

## Contributing

AutoEIS is still under development and the API might change. If you find any bugs or have any suggestions, please file an [issue](https://github.com/AUTODIAL/AutoEIS/issues) or directly submit a [pull request](https://github.com/AUTODIAL/AutoEIS/pulls). We would greatly appreciate any contributions from the community. Please refer to the [contributing guide](https://github.com/AUTODIAL/AutoEIS/doc/contributing.md).

## Installation

### Pip

Open a terminal (or command prompt on Windows) and run the following command:

```bash
pip install -U autoeis
```

Julia dependencies will be automatically installed at first import. It's recommended that you have your own Julia installation, but if you don't, Julia itself will also be installed automatically.

> **How to install Julia?** If you decided to have your own Julia installation (recommended), the official way to install Julia is via [juliaup](https://github.com/JuliaLang/juliaup). [Juliaup](https://github.com/JuliaLang/juliaup) provides a command line interface to automatically install Julia (optionally multiple versions side by side). Working with [juliaup](https://github.com/JuliaLang/juliaup) is straightforward; Please follow the instructions on its GitHub [page](https://github.com/JuliaLang/juliaup).

## Usage

Visit our [example notebooks](https://autodial.github.io/AutoEIS/examples.html) page to learn how to use AutoEIS.

> [!WARNING]
> The examples are designed to be run interactively, so you should use a Jupyter notebook-like environment like Jupyter Lab, IPython Notebook, or VSCode. The examples may not work as expected if you run them in a non-interactive environment like a Python REPL. For a smooth experience, please use a supported environment.

## Workflow

The schematic workflow of AutoEIS is shown below:

![AutoEIS workflow](https://raw.githubusercontent.com/AUTODIAL/AutoEIS/develop/assets/workflow.png)

It includes: data pre-processing, ECM generation, circuit post-filtering, Bayesian inference, and the model evaluation process. Through this workflow, AutoEis can prioritize the statistically optimal ECM and also retain suboptimal models with lower priority for subsequent expert inspection. A detailed workflow can be found in the [paper](https://iopscience.iop.org/article/10.1149/1945-7111/aceab2/meta).

# Acknowledgement

Thanks to Prof. Jason Hattrick-Simpers, Dr. Robert Black, Dr. Debashish Sur, Dr. Parisa Karimi, Dr. Brian DeCost, Dr. Kangming Li, and Prof. John R. Scully for their guidance and support. Also, thanks to Dr. Shijing Sun, Prof. Keryn Lian, Dr. Alvin Virya, Dr. Austin McDannald, Dr. Fuzhan Rahmanian, and Prof. Helge Stein for their feedback and discussions. Special shoutout to Prof. John R. Scully and Dr. Debashish Sur for letting us use their corrosion data to showcase the functionality of AutoEIS—your help has been invaluable!

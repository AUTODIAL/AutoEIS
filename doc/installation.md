# Installation
Follow the instructions below to install AutoEIS and its dependencies:

## AutoEIS
Open a terminal (or command prompt on Windows) and run the following command:

```bash
pip install -U autoeis
```

## Julia dependencies

### Julia language
The official way to install Julia is via [juliaup](https://github.com/JuliaLang/juliaup). [Juliaup](https://github.com/JuliaLang/juliaup) provides a command line interface to automatically install Julia (optionally multiple versions side by side). Working with [juliaup](https://github.com/JuliaLang/juliaup) is straightforward; Please follow the instructions on its GitHub [page](https://github.com/JuliaLang/juliaup).

:::{admonition} Minimum Julia version
:class: warning
AutoEIS requires Julia version 1.9 or higher. This strict requirement is due to many optimizations introduced in Julia 1.9 that significantly reduce the startup time of `EquivalentCircuits.jl`, the backend of AutoEIS.
:::

### EquivalentCircuits.jl
The circuit generation in AutoEIS is done using the Julia package [EquivalentCircuits.jl](https://github.com/MaximeVH/EquivalentCircuits.jl). AutoEIS provides a helper function to automatically install the required Julia dependencies. Open a terminal (or command prompt on Windows) and run the following command:

```shell
python -m autoeis install
```

:::{admonition} About shared environments
:class: note
AutoEIS doesn't pollute your global Julia environment. Instead, it creates a new shared environment called `autoeis-VERSION_NUMBER` (`VERSION_NUMBER` is the AutoEIS version) and installs the required packages there. This way, you can safely use AutoEIS without worrying about breaking your global Julia environment. Shared environments are stored in the `~/.julia/environments` directory on Unix-based systems and `%USERPROFILE%\.julia\environments` on Windows.
:::

## Verify the installation
If all steps were completed successfully, you should now be able to use AutoEIS. To confirm that AutoEIS is installed correctly, running the following command in a terminal (or command prompt on Windows) should print the version number:

```shell
python -c "import autoeis; print(autoeis.__version__)"
```

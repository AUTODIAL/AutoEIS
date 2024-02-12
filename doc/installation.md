# Installation

Open a terminal (or command prompt on Windows) and run the following command:

```bash
pip install -U autoeis
```

Julia dependencies will be automatically installed at first import. It's recommended that you have your own Julia installation, but if you don't, Julia itself will also be installed automatically.

:::{admonition} How to install Julia
:class: note
If you decided to have your own Julia installation (recommended), the official way to install Julia is via [juliaup](https://github.com/JuliaLang/juliaup). [Juliaup](https://github.com/JuliaLang/juliaup) provides a command line interface to automatically install Julia (optionally multiple versions side by side). Working with [juliaup](https://github.com/JuliaLang/juliaup) is straightforward; Please follow the instructions on its GitHub [page](https://github.com/JuliaLang/juliaup).
:::

:::{admonition} Minimum Julia version
:class: warning
AutoEIS requires Julia version 1.9 or higher. This strict requirement is due to many optimizations introduced in Julia 1.9 that significantly reduce the startup time of `EquivalentCircuits.jl`, the backend of AutoEIS.
:::

:::{admonition} About shared environments
:class: note
AutoEIS doesn't pollute your global Julia environment. Instead, it creates a new environment with the same name as your Python virtual environment (if you're in on!) and installs the required packages there. This way, you can safely use AutoEIS without worrying about breaking your global Julia environment. The Julia environment is stored in the same folder as your Python virtual environment. For instance, if you're using the Anaconda Python distribution and the name of your Python virtual environment is `myenv`, the path to the Julia environment is `~/anaconda3/envs/myenv/julia_env` on Unix-based systems and `%USERPROFILE%\anaconda3/envs/myenv/julia_env` on Windows.
:::

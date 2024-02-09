import shutil

import juliapkg

from autoeis.version import __equivalent_circuits_jl_version__


def install():
    """Installs Julia dependencies for AutoEIS."""
    assert_julia_installed()
    kwargs = {"name": "EquivalentCircuits", "uuid": "da5bd070-f609-4e16-a30d-de86b3faa756"}
    if __equivalent_circuits_jl_version__.startswith("v"):
        kwargs["version"] = __equivalent_circuits_jl_version__
    else:
        kwargs["rev"] = __equivalent_circuits_jl_version__
    pkg_spec = juliapkg.PkgSpec(**kwargs)
    juliapkg.add(pkg_spec)
    juliapkg.resolve()


def init_julia():
    """Initializes Julia and returns the Main module."""
    assert_julia_installed()
    import juliacall

    return juliacall.Main


def import_package(package_name, Main):
    """Imports a package in Julia and returns the module."""
    Main.seval(f"using {package_name}")
    return eval(f"Main.{package_name}")


def import_backend(Main):
    """Imports EquivalentCircuits package from Julia."""
    assert_backend_installed()
    return import_package("EquivalentCircuits", Main)


def assert_julia_installed():
    """Asserts that Julia is installed."""
    msg = "Julia not found. Visit https://github.com/JuliaLang/juliaup and install Julia."
    assert shutil.which("julia"), msg


def assert_backend_installed():
    """Asserts that EquivalentCircuits.jl is installed."""
    assert_julia_installed()
    Main = init_julia()
    msg = "EquivalentCircuits.jl not installed."
    assert import_package("EquivalentCircuits", Main), msg

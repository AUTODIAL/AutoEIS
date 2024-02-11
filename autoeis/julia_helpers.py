import shutil

from autoeis.utils import get_logger
from autoeis.version import __equivalent_circuits_jl_version__

log = get_logger(__name__)
def install(ec_path=None):
    """Installs Julia dependencies for AutoEIS."""
    assert_julia_installed()
    import juliapkg

    kwargs = {"name": "EquivalentCircuits", "uuid": "da5bd070-f609-4e16-a30d-de86b3faa756"}
    if ec_path is not None:
        kwargs["path"] = ec_path
        kwargs["dev"] = True
    else:
        if __equivalent_circuits_jl_version__.startswith("v"):
            kwargs["version"] = __equivalent_circuits_jl_version__
        else:
            kwargs["rev"] = __equivalent_circuits_jl_version__
            kwargs["url"] = "https://github.com/ma-sadeghi/EquivalentCircuits.jl"
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


def import_backend(Main=None):
    """Imports EquivalentCircuits package from Julia."""
    Main = init_julia() if Main is None else Main
    assert_backend_installed()
    return import_package("EquivalentCircuits", Main)


def assert_julia_installed():
    """Asserts that Julia is installed."""
    msg = "Julia not found. Visit https://github.com/JuliaLang/juliaup and install Julia."
    if shutil.which("julia") is None:
        raise ImportError(msg)


def assert_backend_installed(Main=None):
    """Asserts that EquivalentCircuits.jl is installed."""
    assert_julia_installed()
    Main = init_julia() if Main is None else Main
    msg = "EquivalentCircuits.jl not installed."
    if import_package("EquivalentCircuits", Main) is None:
        raise ImportError(msg)

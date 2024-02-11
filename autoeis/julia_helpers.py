from juliapkg.find_julia import ju_find_julia_noinstall
from juliapkg.state import STATE

from .utils import get_logger
from .version import __equivalent_circuits_jl_version__

log = get_logger(__name__)


def install(ec_path=None, install_julia=True):
    """Installs Julia dependencies for AutoEIS."""
    if not install_julia:
        is_julia_installed(error=True)
    if not is_julia_installed(error=False):
        log.warning("Julia not found in PATH, installing Julia...")
    # The next line automatically installs Julia if it is not found
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
    is_julia_installed()
    import juliacall

    return juliacall.Main


def import_package(package_name, Main, error=False):
    """Imports a package in Julia and returns the module."""
    from juliacall import JuliaError

    try:
        Main.seval(f"using {package_name}")
        return eval(f"Main.{package_name}")
    except JuliaError as e:
        if error:
            raise e
    return None


def import_backend(Main=None):
    """Imports EquivalentCircuits package from Julia."""
    is_julia_installed(error=True)
    Main = init_julia() if Main is None else Main
    is_backend_installed(error=True)
    return import_package("EquivalentCircuits", Main)


def is_julia_installed(error=False):
    """Asserts that Julia is installed."""
    julia_installed_sys = ju_find_julia_noinstall()
    julia_installed_exe = "executable" in STATE
    if julia_installed_sys or julia_installed_exe:
        return True
    msg = "Julia not found. Visit https://github.com/JuliaLang/juliaup and install Julia."
    if error:
        raise ImportError(msg)
    return False


def is_backend_installed(Main=None, error=False):
    """Asserts that EquivalentCircuits.jl is installed."""
    is_julia_installed(error=True)
    Main = init_julia() if Main is None else Main
    if import_package("EquivalentCircuits", Main) is not None:
        return True
    msg = "EquivalentCircuits.jl not found, run 'python -m autoeis install'"
    if error:
        raise ImportError(msg)
    return False

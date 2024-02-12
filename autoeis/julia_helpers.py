import juliapkg
from juliapkg.deps import can_skip_resolve
from juliapkg.find_julia import find_julia

from .utils import get_logger, suppress_output
from .version import __equivalent_circuits_jl_version__

log = get_logger(__name__)


def install_julia():
    """Installs Julia using juliapkg."""
    # Importing juliacall automatically installs Julia using juliapkg
    import juliacall


def install_backend(ec_path=None):
    """Installs Julia dependencies for AutoEIS."""
    is_julia_installed(error=True)

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
    is_julia_installed(error=True, install=False)
    from juliacall import Main

    return Main


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
    Main = init_julia() if Main is None else Main
    is_backend_installed(error=True, install=False)
    return import_package("EquivalentCircuits", Main)


def is_julia_installed(error=False, install=False):
    """Asserts that Julia is installed."""
    # Look for system-wide Julia executable
    try:
        find_julia()
        return True
    except Exception:
        pass
    # Look for local Julia executable (e.g., installed by juliapkg)
    if can_skip_resolve():
        return True
    if install:
        log.warning("Julia not found, installing Julia...")
        with suppress_output():
            install_julia()
        return True
    msg = "Julia not found. Visit https://github.com/JuliaLang/juliaup and install Julia."
    if error:
        raise ImportError(msg)
    return False


def is_backend_installed(Main=None, error=False, install=False):
    """Asserts that EquivalentCircuits.jl is installed."""
    Main = init_julia() if Main is None else Main
    if import_package("EquivalentCircuits", Main, error=False) is not None:
        return True
    if install:
        log.warning("EquivalentCircuits.jl not found, installing...")
        with suppress_output():
            install_backend()
        return True
    msg = "EquivalentCircuits.jl not found, run 'python -m autoeis install'"
    if error:
        raise ImportError(msg)
    return False


def ensure_julia_deps_ready():
    """Ensures Julia and EquivalentCircuits.jl are installed."""
    is_julia_installed(error=True, install=True)
    is_backend_installed(error=True, install=True)

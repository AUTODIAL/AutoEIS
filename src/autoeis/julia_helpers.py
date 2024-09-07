import logging
import shutil
from pathlib import Path

import juliapkg
from juliapkg.deps import can_skip_resolve
from juliapkg.find_julia import find_julia

from .utils import suppress_output
from .version import __equivalent_circuits_jl_version__

log = logging.getLogger(__name__)


def install_julia(quiet=True):
    """Installs Julia using juliapkg."""
    # Importing juliacall automatically installs Julia using juliapkg
    if quiet:
        with suppress_output():
            import juliacall
    else:
        import juliacall


def install_backend(ec_path: Path = None, quiet=True):
    """Installs Julia dependencies for AutoEIS.

    Parameters
    ----------
    ec_path : Path, optional
        Path to the local copy of EquivalentCircuits. Default is None. If None,
        the remote version will be used.
    """
    is_julia_installed(error=True)

    # TODO: No longer needed since dependencies are specified in juliapkg.json
    # kwargs = {"name": "EquivalentCircuits", "uuid": "da5bd070-f609-4e16-a30d-de86b3faa756"}
    # if ec_path is not None:
    #     kwargs["path"] = str(ec_path)
    #     kwargs["dev"] = True
    # else:
    #     if __equivalent_circuits_jl_version__.startswith("v"):
    #         kwargs["version"] = __equivalent_circuits_jl_version__
    #     else:
    #         kwargs["rev"] = __equivalent_circuits_jl_version__
    #         kwargs["url"] = "https://github.com/ma-sadeghi/EquivalentCircuits.jl"
    # pkg_spec = juliapkg.PkgSpec(**kwargs)
    # juliapkg.add(pkg_spec)

    if quiet:
        with suppress_output():
            juliapkg.resolve()
    else:
        juliapkg.resolve()


def init_julia(quiet=False):
    """Initializes Julia and returns the Main module.

    Raises
    ------
    ImportError
        If Julia is not installed.
    """
    is_julia_installed(error=True)
    if not can_skip_resolve():
        log.warning("Julia is installed, but needs to be resolved...")
    if quiet:
        with suppress_output():
            from juliacall import Main
    else:
        from juliacall import Main

    return Main


def import_package(package_name, Main, error=False):
    """Imports a package in Julia and returns the module.

    Parameters
    ----------
    package_name : str
        Name of the Julia package to import.
    Main : juliacall.Main
        Julia Main module.
    error : bool, optional
        If True, raises an error if the package is not found. Default is False.

    Returns
    -------
    module
        The imported Julia module.

    Raises
    ------
    ImportError
        If the package is not found and error is True.
    """
    from juliacall import JuliaError

    try:
        Main.seval(f"using {package_name}")
        return eval(f"Main.{package_name}")
    except JuliaError as e:
        if error:
            raise e
    return None


def import_backend(Main=None):
    """Imports EquivalentCircuits package from Julia.

    Parameters
    ----------
    Main : juliacall.Main, optional
        Julia Main module. Default is None.

    Returns
    -------
    module
        The imported Julia module.

    Raises
    ------
    ImportError
        If Julia is not installed or the package is not found.
    """
    Main = init_julia() if Main is None else Main
    is_backend_installed(Main=Main, error=True)
    return import_package("EquivalentCircuits", Main)


def is_julia_installed(error=False):
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
    msg = "Julia not found. Visit https://github.com/JuliaLang/juliaup and install Julia."
    if error:
        raise ImportError(msg)
    return False


def is_backend_installed(Main=None, error=False):
    """Asserts that EquivalentCircuits.jl is installed.

    Parameters
    ----------
    Main : juliacall.Main, optional
        Julia Main module. Default is None. If None, the Main module will be
        initialized using `init_julia()`.
    error : bool, optional
        If True, raises an error if the package is not found. Default is False.
    install : bool, optional
        If True, installs the package if it is not found. Default is False.

    Returns
    -------
    bool
        True if the package is installed, False otherwise.

    Raises
    ------
    ImportError
        If Julia is not installed or the package is not found and error is True.
    """
    Main = init_julia() if Main is None else Main
    if import_package("EquivalentCircuits", Main, error=False) is not None:
        return True
    msg = "EquivalentCircuits.jl not found, run 'python -m autoeis install'"
    if error:
        raise ImportError(msg)
    return False


def ensure_julia_deps_ready(quiet=True, retry=True):
    """Ensures Julia and EquivalentCircuits.jl are installed."""

    def _ensure_julia_deps_ready(quiet):
        if not is_julia_installed(error=False):
            log.warning("Julia not found, installing Julia...")
            install_julia(quiet=quiet)
        Main = init_julia(quiet=quiet)
        if not is_backend_installed(Main=Main, error=False):
            log.warning("Julia dependencies not found, installing EquivalentCircuits.jl...")
            install_backend(quiet=quiet)

    def _reset_julia_env(quiet):
        remove_julia_env()
        if quiet:
            with suppress_output():
                juliapkg.resolve(force=True)
        else:
            juliapkg.resolve(force=True)

    try:
        _ensure_julia_deps_ready(quiet)
    except Exception:
        if retry:
            _reset_julia_env(quiet)
            _ensure_julia_deps_ready(quiet)
            return
        raise


def remove_julia_env():
    """Removes the active Julia environment directory.

    Notes
    -----
    When Julia or its dependencies are corrupted, this is a possible fix.
    """
    path_julia_env = Path(juliapkg.project())

    if path_julia_env.exists():
        log.warning(f"Removing Julia environment directory: {path_julia_env}")
        shutil.rmtree(path_julia_env)
    else:
        log.warning("Julia environment directory not found.")

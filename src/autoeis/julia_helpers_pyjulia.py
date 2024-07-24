# This code is based on the file julia_helpers.py from the PySR project.
# It has been adapted to suit the requirements of AutoEIS.
# The original file is under the Apache 2.0 license.
# Acknowledge the original authors and license when utilizing this code.
# Original repository: https://github.com/MilesCranmer/PySR
# Commit reference: 976f8d8 dated 2023-09-16.

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path

from julia.api import JuliaError

from .version import __equivalent_circuits_jl_version__, __version__

MAX_RETRIES = 2
juliainfo = None
julia_initialized = False
julia_kwargs_at_initialization = None
julia_activated_env = None

log = logging.getLogger(__name__)


# TODO: For virtualenvs see https://github.com/JuliaPy/PyCall.jl?tab=readme-ov-file#python-virtual-environments
def install(
    julia_project=None, quiet=False, precompile=None, offline=False
):  # pragma: no cover
    """Install all required dependencies for EquivalentCircuits.jl."""
    import julia

    _julia_version_assertion()
    # Set JULIA_PROJECT so that we install in the autoeis environment
    processed_julia_project, is_shared = _process_julia_project(julia_project)
    _set_julia_project_env(processed_julia_project, is_shared)

    if precompile is False:
        os.environ["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

    if offline:
        os.environ["JULIA_PKG_OFFLINE"] = "true"

    try:
        julia.install(quiet=quiet)
    except julia.tools.PyCallInstallError:
        # Attempt to reset PyCall.jl's build:
        subprocess.run(
            [
                "julia",
                "-e",
                f'ENV["PYTHON"] = "{sys.executable}"; import Pkg; Pkg.build("PyCall")',
            ],
        )
        # Try installing again:
        julia.install(quiet=quiet)

    Main, init_log = init_julia(julia_project, quiet=quiet, return_aux=True)
    io_arg = _get_io_arg(quiet)

    if precompile is None:
        precompile = init_log["compiled_modules"]

    if not precompile:
        Main.eval('ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0')

    if is_shared:
        # Install EquivalentCircuits.jl:
        _add_ec_to_julia_project(Main, io_arg)

    Main.eval("using Pkg")
    Main.eval(f"Pkg.instantiate({io_arg})")

    if precompile:
        Main.eval(f"Pkg.precompile({io_arg})")

    if not quiet:
        log.warning(
            "It is recommended to restart Python after installing AutoEIS "
            "dependencies, so that the Julia environment is properly initialized."
        )


def init_julia(julia_project=None, quiet=False, julia_kwargs=None, return_aux=False):
    """Initialize julia binary, turning off compiled modules if needed."""
    global julia_initialized
    global julia_kwargs_at_initialization
    global julia_activated_env

    if julia_kwargs is None:
        julia_kwargs = {"optimize": 3}

    from julia.core import JuliaInfo, UnsupportedPythonError

    _julia_version_assertion()
    processed_julia_project, is_shared = _process_julia_project(julia_project)
    _set_julia_project_env(processed_julia_project, is_shared)

    try:
        info = JuliaInfo.load(julia="julia")
    except FileNotFoundError:
        _raise_julia_not_found()

    if not info.is_pycall_built():
        _raise_import_error()

    from julia.core import Julia

    try:
        Julia(**julia_kwargs)
    except UnsupportedPythonError:
        # HACK: When switching virtualenvs, we also get UnsupportedPythonError.
        # Check if it resolves by recompiling PyCall, if not turn off compiled_modules
        _recompile_pycall()
        try:
            Julia(**julia_kwargs)
        except UnsupportedPythonError:
            # Static python binary, so we turn off pre-compiled modules.
            julia_kwargs = {**julia_kwargs, "compiled_modules": False}
            Julia(**julia_kwargs)
            log.warning(
                "Your system's Python library is static (e.g., conda), "
                "so precompilation will be turned off. For a dynamic library, "
                "try using `pyenv` and installing with `--enable-shared`: "
                "https://github.com/pyenv/pyenv/blob/master/plugins/python-build/README.md"
            )

    using_compiled_modules = ("compiled_modules" not in julia_kwargs) or julia_kwargs[
        "compiled_modules"
    ]

    from julia import Main as _Main

    Main = _Main

    if julia_activated_env is None:
        julia_activated_env = processed_julia_project

    if julia_initialized and julia_kwargs_at_initialization is not None:
        # Check if the kwargs are the same as the previous initialization
        init_set = set(julia_kwargs_at_initialization.items())
        new_set = set(julia_kwargs.items())
        set_diff = new_set - init_set
        # Remove the `compiled_modules` key, since it is not a user-specified kwarg:
        set_diff = {k: v for k, v in set_diff if k != "compiled_modules"}
        if len(set_diff) > 0:
            log.warning(
                f"Julia has already started. The new Julia options {set_diff} "
                "will be ignored."
            )

    if julia_initialized and julia_activated_env != processed_julia_project:
        Main.eval("using Pkg")

        io_arg = _get_io_arg(quiet)
        # Can't pass IO to Julia call as it evaluates to PyObject, so just directly
        # use Main.eval:
        Main.eval(
            f'Pkg.activate("{_escape_filename(processed_julia_project)}",'
            f"shared = Bool({int(is_shared)}), "
            f"{io_arg})"
        )

        julia_activated_env = processed_julia_project

    if not julia_initialized:
        julia_kwargs_at_initialization = julia_kwargs

    julia_initialized = True
    if return_aux:
        return Main, {"compiled_modules": using_compiled_modules}
    return Main


def import_backend(Main):
    """Load EquivalentCircuits.jl, verify version and return a reference."""
    ec = import_package("EquivalentCircuits", Main=Main)
    # FIXME: Currently don't know how to assert branch name if installed from GitHub
    if __equivalent_circuits_jl_version__.startswith("v"):
        _backend_version_assertion(Main)
    return ec


def import_package(pkg_name, Main):
    """Load a Julia package and return a reference to it."""
    # HACK: On Windows, for some reason, the first two imports fail!
    for _ in range(MAX_RETRIES):
        try:
            Main.eval(f"using {pkg_name}")
            break
        except Exception:
            pass
    else:
        # Import failed, raise proper error
        try:
            Main.eval(f"using {pkg_name}")
        except (JuliaError, RuntimeError) as e:
            _raise_import_error(root=e)
    return importlib.import_module(f"julia.{pkg_name}")


def add_to_path(path):
    if not os.path.exists(path):
        raise ValueError(f"The provided path '{path}' does not exist.")

    if path not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + path


def _recompile_pycall():
    import julia

    try:
        os.environ["PYCALL_JL_RUNTIME_PYTHON"] = sys.executable
        julia.install(quiet=True)
    except julia.tools.PyCallInstallError:
        pass


def _raise_import_error(root: Exception = None):
    """Raise ImportError if Julia dependencies are not installed."""
    raise ImportError(
        "Required dependencies are not installed or built. Run the "
        "following command: import autoeis; autoeis.julia_helpers.install()."
    ) from root


def _raise_julia_not_found(root: Exception = None):
    """Raise FileNotFoundError if Julia is not installed."""
    raise FileNotFoundError(
        "Julia is not installed in your PATH. Please install Julia "
        "and add it to your PATH."
    ) from root


def _load_juliainfo():
    """Execute julia.core.JuliaInfo.load(), and store as juliainfo."""
    global juliainfo

    if juliainfo is None:
        from julia.core import JuliaInfo

        try:
            juliainfo = JuliaInfo.load(julia="julia")
        except FileNotFoundError:
            _raise_julia_not_found()

    return juliainfo


def _get_julia_env_dir():
    """Find the Julia environments' directory."""
    try:
        julia_env_dir_str = subprocess.run(
            ["julia", "-e using Pkg; print(Pkg.envdir())"],
            capture_output=True,
            env=os.environ,
        ).stdout.decode()
    except FileNotFoundError:
        env_path = os.environ["PATH"]
        _raise_julia_not_found()
    return Path(julia_env_dir_str)


def _set_julia_project_env(julia_project, is_shared):
    """Set JULIA_PROJECT environment variable."""
    if is_shared:
        if _is_julia_version_greater_eq(version=(1, 7, 0)):
            os.environ["JULIA_PROJECT"] = "@" + str(julia_project)
        else:
            julia_env_dir = _get_julia_env_dir()
            os.environ["JULIA_PROJECT"] = str(julia_env_dir / julia_project)
    else:
        os.environ["JULIA_PROJECT"] = str(julia_project)


def _get_io_arg(quiet):
    """Return Julia-compatible IO arg that suppresses output if quiet=True."""
    io = "devnull" if quiet else "stderr"
    io_arg = f"io={io}" if _is_julia_version_greater_eq(version=(1, 6, 0)) else ""
    return io_arg


def _process_julia_project(julia_project):
    if julia_project is None:
        is_shared = True
        processed_julia_project = f"autoeis-{__version__}"
    elif julia_project[0] == "@":
        is_shared = True
        processed_julia_project = julia_project[1:]
    else:
        is_shared = False
        processed_julia_project = Path(julia_project)
    return processed_julia_project, is_shared


def _is_julia_version_greater_eq(juliainfo=None, version=(1, 6, 0)):
    """Check if Julia version is greater than specified version."""
    if juliainfo is None:
        juliainfo = _load_juliainfo()
    current_version = (
        juliainfo.version_major,
        juliainfo.version_minor,
        juliainfo.version_patch,
    )
    return current_version >= version


def _add_ec_to_julia_project(Main, io_arg):
    """Install EquivalentCircuits.jl and dependencies to the Julia project."""
    Main.eval("using Pkg")
    Main.eval(f"Pkg.Registry.update({io_arg})")
    kwargs = {"name": "EquivalentCircuits"}
    if __equivalent_circuits_jl_version__.startswith("v"):
        kwargs["version"] = __equivalent_circuits_jl_version__
    else:
        kwargs["rev"] = __equivalent_circuits_jl_version__
    Main.ec_spec = Main.PackageSpec(**kwargs)
    Main.eval(f"Pkg.add([ec_spec], {io_arg})")


def _escape_filename(filename):
    """Turn a path into a string with correctly escaped backslashes."""
    str_repr = str(filename)
    str_repr = str_repr.replace("\\", "\\\\")
    return str_repr


def _julia_version_assertion():
    """Check if Julia version is greater than 1.9"""
    if not _is_julia_version_greater_eq(version=(1, 9, 0)):
        raise NotImplementedError(
            "AutoEIS requires Julia 1.9.0 or greater. "
            "Please update your Julia installation."
        )


def _backend_version_assertion(Main):
    """Check if EquivalentCircuits.jl version is correct."""
    try:
        backend_version = Main.eval("string(pkgversion(EquivalentCircuits))")
        expected_backend_version = __equivalent_circuits_jl_version__
        if backend_version != expected_backend_version:  # pragma: no cover
            log.warning(
                f"AutoEIS backend (EquivalentCircuits.jl) version {backend_version} "
                f"does not match expected version {expected_backend_version}. "
                "Things may break. Please update your AutoEIS installation with "
                "`import autoeis; autoeis.julia_helpers.install()`."
            )
    except JuliaError:  # pragma: no cover
        log.warning(
            "You seem to have an outdated version of EquivalentCircuits.jl. "
            "Things may break. Please update your AutoEIS installation with "
            "`import autoeis; autoeis.julia_helpers.install()`."
        )


def _update_julia_project(Main, is_shared, io_arg):
    try:
        if is_shared:
            _add_ec_to_julia_project(Main, io_arg)
        Main.eval("using Pkg")
        Main.eval(f"Pkg.resolve({io_arg})")
    except (JuliaError, RuntimeError) as e:
        _raise_import_error(root=e)

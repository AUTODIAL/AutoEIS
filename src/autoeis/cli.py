from pathlib import Path
import click

# Add this global switch
julia_enabled = False

if julia_enabled:
    from .julia_helpers import install_backend, install_julia
else:
    # dummy fallback to avoid import errors
    def install_julia(*args, **kwargs):
        raise RuntimeError("Julia functionality is disabled.")

    def install_backend(*args, **kwargs):
        raise RuntimeError("Julia functionality is disabled.")

@click.group("autoeis")
@click.pass_context
def autoeis_installer(context):
    ctx = context

@click.option(
    "--ec-path",
    default=None,
    type=str,
    help="Installs a local copy of EquivalentCircuits instead of the remote version.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Prints the installation process to the console.",
)
@autoeis_installer.command("install", help="Install Julia dependencies for AutoEIS.")
def install_cli(ec_path, verbose):
    if not julia_enabled:
        raise RuntimeError("Julia functionality is disabled.")

    if ec_path is not None:
        # Expand ~ in path if present
        ec_path = Path(ec_path).expanduser()

    install_julia(quiet=not verbose)
    install_backend(ec_path=ec_path, quiet=not verbose)

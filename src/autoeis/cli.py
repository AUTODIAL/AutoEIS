from pathlib import Path

import click

from .julia_helpers import install_backend, install_julia


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
    if ec_path is not None:
        # Expand ~ in path if present
        ec_path = Path(ec_path).expanduser()
    install_julia(quiet=not verbose)
    install_backend(ec_path=ec_path, quiet=not verbose)

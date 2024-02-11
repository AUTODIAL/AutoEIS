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
@autoeis_installer.command("install", help="Install Julia dependencies for AutoEIS.")
def install_cli(ec_path):
    install_julia()
    install_backend(ec_path=ec_path)

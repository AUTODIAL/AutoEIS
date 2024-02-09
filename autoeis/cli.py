import click

from autoeis.julia_helpers import install


@click.group("autoeis")
@click.pass_context
def autoeis_installer(context):
    ctx = context


@autoeis_installer.command("install", help="Install Julia dependencies for AutoEIS.")
def install_cli():
    install()

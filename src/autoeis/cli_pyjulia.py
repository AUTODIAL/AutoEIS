import click

from autoeis.julia_helpers import install


@click.group("autoeis")
@click.pass_context
def autoeis_installer(context):
    ctx = context


@autoeis_installer.command("install", help="Install Julia dependencies for AutoEIS.")
@click.option(
    "-p",
    "julia_project",
    "--project",
    default=None,
    type=str,
    help="Install in a specific Julia project (e.g., a local copy of EquivalentProject.jl).",
    metavar="PROJECT_DIRECTORY",
)
@click.option("-q", "--quiet", is_flag=True, default=False, help="Disable logging.")
@click.option(
    "--precompile",
    "precompile",
    flag_value=True,
    default=None,
    help="Force precompilation of Julia libraries.",
)
@click.option(
    "--no-precompile",
    "precompile",
    flag_value=False,
    default=None,
    help="Disable precompilation.",
)
def install_cli(julia_project, quiet, precompile):
    install(julia_project=julia_project, quiet=quiet, precompile=precompile)

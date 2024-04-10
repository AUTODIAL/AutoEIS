import shutil
from datetime import date

from autoeis.version import __version__

# Copy notebooks to the root of the documentation
shutil.rmtree("examples", ignore_errors=True)
shutil.copytree("../examples", "examples", dirs_exist_ok=True)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "AutoEIS"
copyright = f"{date.today().year}, AutoEIS developers"
author = "Runze Zhang, Amin Sadeghi, Jason Hattrick-Simpers"
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# For Sphinx not to complain about missing heading levels
suppress_warnings = ["myst.header"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "myst_parser",  # already activated by myst_nb
    # "myst_nb",  # overrides nbsphinx
    "sphinx_copybutton",
    "nbsphinx",
    # 'autodoc2',
    # 'numpydoc',
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc2_packages = [
    "../autoeis",
]

# myst_nb config
# nb_execution_timeout = 600
# nb_execution_mode = "cache"

# nbsphinx config
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]
nbsphinx_execute = "always"
nbsphinx_prompt_width = "0"
nbsphinx_allow_errors = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_title = ''
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo-light-mode.png",
    "dark_logo": "logo-dark-mode.png",
}

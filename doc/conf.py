from datetime import date


def get_version_from_file(file_path):
    version = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('__version__'):
                # Assuming the line is in the form of '__version__ = "0.0.17"'
                version = line.split('=')[1].strip().strip('"').strip("'")
                break
    return version


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AutoEIS'
copyright = f"{date.today().year}, AutoEIS developers"
author = 'Runze Zhang, Amin Sadeghi, Jason Hattrick-Simpers'
version = get_version_from_file("../autoeis/version.py")
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# For Sphinx not to complain about missing heading levels
suppress_warnings = ["myst.header"]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'myst_parser',
    'sphinx_copybutton',
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

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc2_packages = [
    "../autoeis",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
# html_title = ''
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo-light-mode.png",
    "dark_logo": "logo-dark-mode.png",
}

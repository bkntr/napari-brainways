# Configuration file for the Sphinx documentation builder.

# -- Project information

from importlib.metadata import version as importlib_version

project = "napari-brainways"
copyright = "2023, Ben Kantor"
author = "Ben Kantor"

release = importlib_version("brainways_reg_model")
version = importlib_version("brainways_reg_model")

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
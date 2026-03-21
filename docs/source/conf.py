# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import logging
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve
import sphinx_rtd_theme

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "_ext"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
logger = logging.getLogger(__name__)

project = "SpaceTravLR"
author = "Koushul & Ally"
copyright = f"{datetime.now():%Y}, Koushul & Ally @ jishnulab.org"

# -- Generate timeline from YAML ---------------------------------------------
try:
    from generate_timeline import generate_timeline
    generate_timeline()
except Exception as e:
    logger.warning(f"Failed to generate timeline from YAML: {e}")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = [
#     "sphinx_rtd_theme",
#     "sphinx.ext.autodoc",
#     "sphinx.ext.napoleon",
#     "sphinx.ext.mathjax",
#     # "sphinx.ext.intersphinx",
#     "sphinx.ext.autosummary",
#     # "sphinxcontrib.bibtex",
#     # "sphinx.ext.doctest",
#     # "sphinx.ext.coverage",
#     # "sphinx.ext.githubpages",
#     "edit_on_github",
#     # "sphinx_autodoc_typehints",
#     "nbsphinx",
# ]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    # "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "myst_nb",
    "sphinx_design",  # for cards
    "sphinx_tippy",
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'special-members': '__init__',
}
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

master_doc = "index"
pygments_style = "tango"
pygments_dark_style = "monokai"

# master_doc = "index"
# pygments_style = "sphinx"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

nitpicky = False

# # bibliography
# bibtex_bibfiles = ["references.bib"]
# bibtex_reference_style = "author_year"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
    "light_logo": "img/logo.png",
    "dark_logo": "img/logo.png"

}


html_show_sphinx = False
html_show_sourcelink = False
html_title = "SpaceDocs"

# autodoc + napoleon
autosummary_generate = True
autodoc_member_order = "alphabetical"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True


# myst
nb_execution_mode = "off"
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

# hover
tippy_anchor_parent_selector = "div.content"
tippy_enable_mathjax = True
# no need because of sphinxcontrib-bibtex
tippy_enable_doitips = False
linkcheck_report_timeouts_as_broken = True

# # -- Basic notebooks and those stored under /vignettes and /perspectives --

# notebooks_url = "https://github.com/Koushul/SpaceTravLR_notebooks/raw/master/"
# notebooks = []
# notebook = [
#     "germinal_center.ipynb",
# ]
# notebooks.extend(notebook)

# notebook = [
#     "germinal_center.ipynb"
# ]
# notebooks.extend([f"vignettes/{nb}" for nb in notebook])

# notebook = ["Perspectives.ipynb", "Perspectives_parameters.ipynb"]
# notebooks.extend([f"perspectives/{nb}" for nb in notebook])

# # -- Retrieve all notebooks --

# for nb in notebooks:
#     url = notebooks_url + nb
#     try:
#         urlretrieve(url, nb)
#     except URLError as e:
#         logger.error(f"Unable to retrieve notebook: `{url}`. Reason: `{e}`")

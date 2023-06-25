# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dance'
copyright = '2022, OmicsML'
author = 'DANCE Team'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinxcontrib.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

intersphinx_mapping = {
    'anndata': ('https://anndata.readthedocs.io/en/stable/', None),
    'mudata': ('https://mudata.readthedocs.io/en/stable/', None),
    'scanpy': ('https://scanpy.readthedocs.io/en/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
}

autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Napoleon settings -------------------------------------------------------
napoleon_include_init_with_doc = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []

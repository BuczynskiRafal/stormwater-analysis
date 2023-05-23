import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Stormwater analysis"
copyright = "2023, Rafał Buczyński"
author = "Rafał Buczyński"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []  # type: ignore

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

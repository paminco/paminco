# https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/configuring.html

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import paminco

# -- Project information -----------------------------------------------------

project = 'paminco'
copyright = '2021, Philipp Warode, Per Joachims'
author = 'Philipp Warode, Per Joachims'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.viewcode',
    "sphinx.ext.autosummary",
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    # 'sphinx_automodapi.smart_resolver',
    # 'sphinx_automodapi.automodapi',
    'numpydoc',
    'myst_nb',
    'sphinx_copybutton',
    'sphinx_panels',
    'sphinx_togglebutton',
    
    'sphinxcontrib.contentui',
    # 'sphinx.ext.napoleon',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "deflist",
    "html_admonition",
    "html_image",
    "substitution",
]
myst_substitutions = {
  "packagename": "paminco"
}


autosummary_generate = True
numpydoc_show_class_members = False
autosectionlabel_maxdepth = 1
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"
# html_logo = "_static/logo1.svg"
html_theme_options = {
    "logo_link": "index",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/paminco/paminco",
            "icon": "fab fa-github-square",
        },
    ],
    "show_prev_next": False,
    "collapse_navigation": False,
    "navigation_depth": 2,
}

# Add any paths that contain custom stat    ic files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    "css/paminco.css",
    "css/user_guide.css",
    "css/api.css",
]


# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    'python': ('http://docs.python.org/3', None),
    'numpy': ("https://numpy.org/doc/stable/", None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ("https://matplotlib.org/stable", None),
    'pandas': ("https://pandas.pydata.org/pandas-docs/stable", None),
}

# -- NBSphinx ----------------------------------------------------------------
# nbsphinx_input_prompt = 'In [%s]:'
# nbsphinx_prolog = """
# .. raw:: html

#     <style>
#         p {
#             margin-top: 1rem;
#         }
        
#         .nbinput .prompt,
#         .nboutput .prompt {
#             display: none;
#         }
#     </style>
# """

# -- Copybutton  -------------------------------------------------------------
# To disable the copy button on input/output of notebook-code-cells
copybutton_selector = ",".join(
    [
        f"div.highlight-{css_class} div.highlight pre"
        for css_class in ("python", "ipython3", "default", "bash")
    ]
)

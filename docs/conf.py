# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../python'))

# -- Project information -----------------------------------------------------
project = 'LibAccInt'
copyright = '2026, Bryce M. Westheimer'
author = 'Bryce M. Westheimer'
version = '0.1.0'
release = '0.1.0-alpha.2'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'myst_parser',
    'breathe',
    'sphinx_rtd_theme',
]

# Auto-generate autosummary stubs
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = None
html_favicon = None

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

html_context = {
    'display_github': True,
    'github_user': 'brycewestheimer',
    'github_repo': 'libaccint-public',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# -- Extension configuration -------------------------------------------------

# Breathe configuration (Doxygen XML integration)
breathe_projects = {
    'libaccint': '../build/docs/doxygen/xml'
}
breathe_default_project = 'libaccint'
breathe_default_members = ('members', 'undoc-members')

# Napoleon configuration (Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}
autodoc_typehints = 'description'

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Todo extension
todo_include_todos = True

# -- Options for MathJax -----------------------------------------------------
mathjax3_config = {
    'tex': {
        'macros': {
            'Real': r'\mathbb{R}',
            'bra': [r'\left\langle #1 \right|', 1],
            'ket': [r'\left| #1 \right\rangle', 1],
            'braket': [r'\left\langle #1 \middle| #2 \right\rangle', 2],
            'zeta': r'\zeta',
            'eta': r'\eta',
            'rho': r'\rho',
            'ERI': r'\left(\mu\nu\middle|\lambda\sigma\right)',
        }
    }
}

# -- Custom CSS --------------------------------------------------------------


def setup(app):
    app.add_css_file('custom.css')

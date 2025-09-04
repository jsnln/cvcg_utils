# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cvcg_utils'
copyright = '2025, jsnln'
author = 'jsnln'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

highlight_language = 'python'
pygments_style = 'sphinx'

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.todo',
    'sphinx.ext.apidoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "classic"
html_theme = "sphinx_rtd_theme"
# html_theme = 'alabaster'
html_static_path = ['_static']

todo_include_todos = True

autodoc_mock_imports = ['matplotlib', 'pytorch3d', 'tqdm', 'drtk', 'diff_gaussian_rasterization', 'e3nn', 'torch', 'imageio']

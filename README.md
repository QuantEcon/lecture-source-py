
# Source files for "Lectures in Quantitative Economics" -- Python version

This repository contains the `rst` source files for each python lecture in the [Quantitative Economics](https://lectures.quantecon.org/) lecture series. Additionally the repository holds assets such as code snippets, figures and includes that are referenced from the source files.

You can quickly and easily build and run the lectures as Jupyter Notebooks with the help of a [QuantEcon](https://quantecon.org/) built extension [sphinxcontrib.jupyter](https://github.com/QuantEcon/sphinxcontrib-jupyter).


## Installation

1) Download and install [Anacoda](https://www.anaconda.com/distribution/) for your platform (Windows/Linux/Mac OS).

2) Download or clone this repository.

3) Run `make setup`

*The make setup command checks for and installs the [quantecon package](https://pypi.org/project/quantecon/) and the [sphinxcontrib.jupyter extension](https://pypi.org/project/sphinxcontrib-jupyter/) for [Sphinx](https://www.sphinx-doc.org/). All other dependencies are included with Anaconda including the [Jupyter Notebook](https://jupyter.org/) viewer.*


## Building notebooks

Run `make notebooks`

*The make notebooks command converts the `rst` source files into `ipynb` notebooks using Sphinx and the sphinxcontrib.jupyter extension. The `ipynb` files are stored in a temporary `_build` directory.*


## Viewing notebooks

Run `make view`

Additionally you can view a lecture directly:

Example `make view lecture=about_py`

*The make view command launches a local instance of Jupyter Notebook viewer. A new web browser will open displaying a web based interface for viewing and interacting with the notebook files.*

## Editing notebooks

Edit the source `rst` files in your favorite text editor and run `make notebooks`.

If you have previously run `make view` and have your web browser with Jupyter Notebook still running you can simply refresh the page to view your updates. Alternatively, run `make view` again to relaunch Jupyter Notebook in your web browser.

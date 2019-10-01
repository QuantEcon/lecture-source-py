
# Lectures in Quantitative Economics: Source Files

### Python version

This repository contains

* the `rst` source files for each python lecture in [Quantitative Economics](https://lectures.quantecon.org/), in directory `source/rst`

* supporting Python code in `source/_static/code/`

* supporting figures, PDFs and other static assets in `source/_static`.


## Installation

1) Download and install [Anaconda](https://www.anaconda.com/distribution/) for your platform (Windows/Linux/Mac OS).

2) Download or clone this repository.

3) Enter your local copy of the repository and run `make setup`.

The `make setup` command checks for and installs

* the [quantecon package](https://quantecon.org/quantecon-py) and

* the [sphinxcontrib.jupyter extension](https://github.com/QuantEcon/sphinxcontrib-jupyter) for [Sphinx](https://www.sphinx-doc.org/).

Other dependencies are included with Anaconda.




## Building notebooks

To transform the `rst` files in to `ipynb` files, enter the repo and run `make notebooks`.

The resulting `ipynb` files are stored in a temporary `_build` directory at the root level of the repository.


## Viewing notebooks

Run `make view`

Additionally you can view a particular lecture directly:

* Example: `make view lecture=about_py`

The `make view` command launches a local instance of Jupyter and points it at
the contents of the `_build` directory.


## Workflow

Standard workflow for editing, say, `lqcontrol.rst` in the master branch is

1. Enter your local copy of `lecture-source-py` and type `git pull` to get the latest version
1. Run `make notebooks`
1. Run `make view lecture=lqcontrol` to see `lqcontrol.ipynb` in Jupyter
    * or just `make view` and then navigate to `lqcontrol.ipynb` in the browser window that pops up
1.  Edit `lqcontrol.rst` in your favorite text editor
1. Run `make notebooks` again to generate a new version of `lqcontrol.ipynb`
    * the build system uses caching so this should be quick
    * you might need to open a new terminal window to run this command
1. Return to `lqcontrol.ipynb` in Jupyter and reload the page
1. Go to step 4 and repeat as necessary.

Finally, add, commit and push your changes using `git`.


## Converting notebooks to RST files

Sometimes it's convenient to write a lecture as a notebook and then convert to
RST

This guide is provided by TJS and requires pandoc 2.6 or newer

(Use `pandoc --version` to test)

1.  This step is necessary only if you want to strip out dollar signs from maths

    *  `python latex_space_strip.py  [myinputfile.ipynb] -o [myoutputfile.ipynb]`

2.  To convert, use

    *  `pandoc [myfilenamenew.pynb] -f ipynb+tex_math_dollars -t rst -s -o [newfilename.rst]`


## Style Guide - Writing Conventions

### Mathematical Notation

Matrices always use square brackets. Use `\begin{bmatrix} ... \end{bmatrix}`

Sequences use curly brackets, such as `\{ x_t \}_{t=0}^{\infty}`

The use of align environments can be done using the `\begin{algined} ... \end{aligned}` as it is not a full math environment and works within the equation wrapping of sphinx.

"Independent and identically distributed" is abbreviated to "IID".

The headings should not use math-environment.

Labels must be written in all small alphabetical letters. Any special character should be avoided in labels except "dash" i.e "-"

All the cite key must use the default google scholar bibtex conventions.

### Emphasis and Definitions

Use **bold** for definitions and _italic_ for emphasis. For example,

* A **closed set** is a set whose complement is open.
* All consumers have _identical_ endowments.

### Titles and Headings
* Capitalization of all words for all titles.
  > Example “How it Works: Data, Variables and Names”

### Adding References
#### Adding a Citation to a Lecture

To add a reference to the text of a QuantEcon lecture you need to use the `:cite:<bibtex-label>` directive.

For example

```
:cite:`StokeyLucas1989`, chapter 2
```

is rendered rendered in HTML and LaTex as:

> [SLP89], chapter 2

#### Adding a new reference to QuantEcon

To add a new reference to the project, a bibtex entry needs to be added to `QuantEcon.lectures/_static/quant-econ.bib`.

### Sphinx and Restructured Text

#### Editing
The syntax of the source files is reStructuredText.

[Here is a nice primer](http://sphinx-doc.org/rest.html) on how to write reStructuredText files.

[Here is the documentation](http://jinja.pocoo.org/docs/dev/) for the Jinja template syntax.

### Helpful Links
* [A nice Sphinx tutorial](http://sphinx-doc.org/tutorial.html)
* [Another rst primer](http://docutils.sourceforge.net/docs/user/rst/quickstart.html)

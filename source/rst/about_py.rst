.. _about_py:

.. include:: /_static/includes/header.raw

.. index::
    single: python

******************************************
About Python
******************************************

.. contents:: :depth: 2

.. epigraph::

   "Python has gotten sufficiently weapons grade that we don’t descend into R
   anymore. Sorry, R people. I used to be one of you but we no longer descend
   into R." -- Chris Wiggins



Overview
============

In this lecture we will

* outline what Python is
* showcase some of its abilities
* compare it to some other languages.

At this stage, it's **not** our intention that you try to replicate all you see.

We will work through what follows at a slow pace later in the lecture series.

Our only objective for this lecture is to give you some feel of what Python is, and what it can do.



What's Python?
============================

`Python <https://www.python.org>`_ is a general-purpose programming language conceived in 1989 by Dutch programmer `Guido van Rossum <https://en.wikipedia.org/wiki/Guido_van_Rossum>`_.

Python is free and open source, with development coordinated through the `Python Software Foundation <https://www.python.org/psf/>`_.

Python has experienced rapid adoption in the last decade and is now one of the most popular programming languages.

Common Uses
------------

:index:`Python <single: Python; common uses>` is a general-purpose language used in almost all application domains such as

* communications

* web development

* CGI and graphical user interfaces

* game development

* multimedia, data processing, security, etc., etc., etc.

Used extensively by Internet services and high tech companies including

* `Google <https://www.google.com/>`_

* `Dropbox <https://www.dropbox.com/>`_

* `Reddit <https://www.reddit.com/>`_

* `YouTube <https://www.youtube.com/>`_

* `Walt Disney Animation <https://pydanny-event-notes.readthedocs.org/en/latest/socalpiggies/20110526-wda.html>`_.

Python is very beginner-friendly and is often used to `teach computer science and programming <http://cacm.acm.org/blogs/blog-cacm/176450-python-is-now-the-most-popular-introductory-teaching-language-at-top-us-universities/fulltext>`_.

For reasons we will discuss, Python is particularly popular within the scientific community with users including NASA, CERN and practically all branches of academia.

It is also `replacing familiar tools like Excel <https://news.efinancialcareers.com/us-en/3002556/python-replaced-excel-banking>`_ in the fields of finance and banking.


Relative Popularity
----------------------

The following chart, produced using Stack Overflow Trends, shows one measure of the relative popularity of Python

.. figure:: /_static/lecture_specific/about_py/python_vs_matlab.png

The figure indicates not only that Python is widely used but also that adoption of Python has accelerated significantly since 2012.

We suspect this is driven at least in part by uptake in the scientific
domain, particularly in rapidly growing fields like data science.

For example, the popularity of `pandas <http://pandas.pydata.org/>`_, a library for data analysis with Python has exploded, as seen here.

(The corresponding time path for MATLAB is shown for comparison)

.. figure:: /_static/lecture_specific/about_py/pandas_vs_matlab.png

Note that pandas takes off in 2012, which is the same year that we see
Python's popularity begin to spike in the first figure.

Overall, it's clear that

* Python is `one of the most popular programming languages worldwide <http://spectrum.ieee.org/computing/software/the-2017-top-programming-languages>`__.

* Python is a major tool for scientific computing, accounting for a rapidly rising share of scientific work around the globe.




Features
----------

Python is a `high-level language <https://en.wikipedia.org/wiki/High-level_programming_language>`_ suitable for rapid development.

It has a relatively small core language supported by many libraries.

Other features of Python:

* multiple programming styles are supported (procedural, object-oriented, functional, etc.) 

* it is interpreted rather than compiled.



Syntax and Design
-----------------

.. index::
    single: Python; syntax and design

One nice feature of Python is its elegant syntax --- we'll see many examples later on.

Elegant code might sound superfluous but in fact it's highly beneficial because it makes the syntax easy to read and easy to remember.

Remembering how to read from files, sort dictionaries and other such routine tasks means that you don't need to break your flow in order to hunt down correct syntax.

Closely related to elegant syntax is an elegant design.

Features like iterators, generators, decorators and list comprehensions make Python highly expressive, allowing you to get more done with less code.

`Namespaces <https://en.wikipedia.org/wiki/Namespace>`_ improve productivity by cutting down on bugs and syntax errors.




Scientific Programming
============================

.. index::
    single: scientific programming

Python has become one of the core languages of scientific computing.

It's either the dominant player or a major player in

* `machine learning and data science <http://scikit-learn.org/stable/>`_
* `astronomy <http://www.astropy.org/>`_
* `artificial intelligence <https://wiki.python.org/moin/PythonForArtificialIntelligence>`_
* `chemistry <http://chemlab.github.io/chemlab/>`_
* `computational biology <http://biopython.org/wiki/Main_Page>`_
* `meteorology <https://pypi.org/project/meteorology/>`_

Its popularity in economics is also beginning to rise.

This section briefly showcases some examples of Python for scientific programming.

* All of these topics will be covered in detail later on.


Numerical Programming
------------------------

.. index::
    single: scientific programming; numeric

Fundamental matrix and array processing capabilities are provided by the excellent `NumPy <http://www.numpy.org/>`_ library.

NumPy provides the basic array data type plus some simple processing operations.

For example, let's build some arrays

.. code-block:: python3

    import numpy as np                     # Load the library

    a = np.linspace(-np.pi, np.pi, 100)    # Create even grid from -π to π
    b = np.cos(a)                          # Apply cosine to each element of a
    c = np.sin(a)                          # Apply sin to each element of a



Now let's take the inner product

.. code-block:: python3

    b @ c

The number you see here might vary slightly but it's essentially zero.

(For older versions of Python and NumPy you need to use the `np.dot <http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html>`_ function)



.. index:
    single: SciPy

The `SciPy <http://www.scipy.org>`_ library is built on top of NumPy and provides additional functionality.

.. _tuple_unpacking_example:

For example, let's calculate :math:`\int_{-2}^2 \phi(z) dz` where :math:`\phi` is the standard normal density.

.. code-block:: python3

    from scipy.stats import norm
    from scipy.integrate import quad

    ϕ = norm()
    value, error = quad(ϕ.pdf, -2, 2)  # Integrate using Gaussian quadrature
    value


SciPy includes many of the standard routines used in

* `linear algebra <http://docs.scipy.org/doc/scipy/reference/linalg.html>`_

* `integration <http://docs.scipy.org/doc/scipy/reference/integrate.html>`_

* `interpolation <http://docs.scipy.org/doc/scipy/reference/interpolate.html>`_

* `optimization <http://docs.scipy.org/doc/scipy/reference/optimize.html>`_

* `distributions and random number generation <http://docs.scipy.org/doc/scipy/reference/stats.html>`_

* `signal processing <http://docs.scipy.org/doc/scipy/reference/signal.html>`_

See them all `here <http://docs.scipy.org/doc/scipy/reference/index.html>`_.



Graphics
--------------------

.. index::
    single: Matplotlib

The most popular and comprehensive Python library for creating figures and graphs is `Matplotlib <http://matplotlib.org/>`_, with functionality including

* plots, histograms, contour images, 3D graphs, bar charts etc.

* output in many formats (PDF, PNG, EPS, etc.)

* LaTeX integration

Example 2D plot with embedded LaTeX annotations

.. figure:: /_static/lecture_specific/about_py/qs.png

Example contour plot

.. figure:: /_static/lecture_specific/about_py/bn_density1.png

Example 3D plot

.. figure:: /_static/lecture_specific/about_py/career_vf.png

More examples can be found in the `Matplotlib thumbnail gallery <http://matplotlib.org/gallery.html>`_.

Other graphics libraries include

* `Plotly <https://plot.ly/python/>`_
* `Bokeh <http://bokeh.pydata.org/en/latest/>`_
* `VPython <http://www.vpython.org/>`_ --- 3D graphics and animations



Symbolic Algebra
--------------------

It's useful to be able to manipulate symbolic expressions, as in Mathematica or Maple.

.. index::
    single: SymPy

The `SymPy <http://www.sympy.org/>`_ library provides this functionality from within the Python shell.

.. code-block:: python3

    from sympy import Symbol

    x, y = Symbol('x'), Symbol('y')  # Treat 'x' and 'y' as algebraic symbols
    x + x + x + y


We can manipulate expressions

.. code-block:: python3

    expression = (x + y)**2
    expression.expand()


solve polynomials

.. code-block:: python3

    from sympy import solve

    solve(x**2 + x + 2)


and calculate limits, derivatives and integrals

.. code-block:: python3

    from sympy import limit, sin, diff

    limit(1 / x, x, 0)


.. code-block:: python3

    limit(sin(x) / x, x, 0)


.. code-block:: python3

    diff(sin(x), x)


The beauty of importing this functionality into Python is that we are working within 
a fully fledged programming language. 

We can easily create tables of derivatives, generate LaTeX output, add that output 
to figures and so on.


Statistics
--------------------

Python's data manipulation and statistics libraries have improved rapidly over
the last few years.

Pandas
^^^^^^
.. index::
    single: Pandas

One of the most popular libraries for working with data is `pandas <http://pandas.pydata.org/>`_.

Pandas is fast, efficient, flexible and well designed.

Here's a simple example, using some dummy data generated with Numpy's excellent 
``random`` functionality.

.. code-block:: python3

    import pandas as pd
    np.random.seed(1234)

    data = np.random.randn(5, 2)  # 5x2 matrix of N(0, 1) random draws
    dates = pd.date_range('28/12/2010', periods=5)

    df = pd.DataFrame(data, columns=('price', 'weight'), index=dates)
    print(df)


.. code-block:: python3

    df.mean()



Other Useful Statistics Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. index::
    single: statsmodels

* `statsmodels <http://statsmodels.sourceforge.net/>`_ --- various statistical routines

.. index::
    single: scikit-learn

* `scikit-learn <http://scikit-learn.org/>`_ --- machine learning in Python (sponsored by Google, among others)

.. index::
    single: pyMC

* `pyMC <http://pymc-devs.github.io/pymc/>`_ --- for Bayesian data analysis

.. index::
    single: pystan

* `pystan <https://pystan.readthedocs.org/en/latest/>`_ Bayesian analysis based on `stan <http://mc-stan.org/>`_


Networks and Graphs
--------------------

Python has many libraries for studying graphs.

.. index::
    single: NetworkX

One well-known example is `NetworkX <http://networkx.github.io/>`_. 
Its features include, among many other things:

* standard graph algorithms for analyzing networks

* plotting routines

Here's some example code that generates and plots a random graph, with node color determined by shortest path length from a central node.

.. code-block:: ipython

  import networkx as nx
  import matplotlib.pyplot as plt
  %matplotlib inline
  np.random.seed(1234)

  # Generate a random graph
  p = dict((i, (np.random.uniform(0, 1), np.random.uniform(0, 1)))
           for i in range(200))
  g = nx.random_geometric_graph(200, 0.12, pos=p)
  pos = nx.get_node_attributes(g, 'pos')

  # Find node nearest the center point (0.5, 0.5)
  dists = [(x - 0.5)**2 + (y - 0.5)**2 for x, y in list(pos.values())]
  ncenter = np.argmin(dists)

  # Plot graph, coloring by path length from central node
  p = nx.single_source_shortest_path_length(g, ncenter)
  plt.figure()
  nx.draw_networkx_edges(g, pos, alpha=0.4)
  nx.draw_networkx_nodes(g,
                         pos,
                         nodelist=list(p.keys()),
                         node_size=120, alpha=0.5,
                         node_color=list(p.values()),
                         cmap=plt.cm.jet_r)
  plt.show()


Cloud Computing
--------------------------------

.. index::
    single: cloud computing

Running your Python code on massive servers in the cloud is becoming easier and easier.

.. index::
    single: cloud computing; anaconda enterprise

A nice example is `Anaconda Enterprise <https://www.anaconda.com/enterprise/>`_.

See also

.. index::
    single: cloud computing; amazon ec2

* `Amazon Elastic Compute Cloud <http://aws.amazon.com/ec2/>`_

.. index::
    single: cloud computing; google app engine

* The `Google App Engine <https://cloud.google.com/appengine/>`_ (Python, Java, PHP or Go)

.. index::
    single: cloud computing; pythonanywhere

* `Pythonanywhere <https://www.pythonanywhere.com/>`_

.. index::
    single: cloud computing; sagemath cloud

* `Sagemath Cloud <https://cloud.sagemath.com/>`_


Parallel Processing
--------------------------------

.. index::
    single: parallel computing

Apart from the cloud computing options listed above, you might like to consider

.. index::
    single: parallel computing; ipython

* `Parallel computing through IPython clusters <http://ipython.org/ipython-doc/stable/parallel/parallel_demos.html>`_.

.. index::
    single: parallel computing; starcluster

* The `Starcluster <http://star.mit.edu/cluster/>`_ interface to Amazon's EC2.

.. index::
    single: parallel computing; copperhead

.. index::
    single: parallel computing; pycuda

* GPU programming through `PyCuda <https://wiki.tiker.net/PyCuda>`_, `PyOpenCL <https://mathema.tician.de/software/pyopencl/>`_, `Theano <http://deeplearning.net/software/theano/>`_ or similar.



.. _intfc:



Other Developments
------------------------

There are many other interesting developments with scientific programming in Python.

Some representative examples include

.. index::
    single: scientific programming; Jupyter

* `Jupyter <http://jupyter.org/>`_ --- Python in your browser with interactive code cells,  embedded images and other useful features.

.. index::
    single: scientific programming; Numba

* `Numba <http://numba.pydata.org/>`_ --- Make Python run at the same speed as native machine code!


.. index::
    single: scientific programming; Blaze

* `Blaze <http://blaze.pydata.org/>`_ --- a generalization of NumPy.


.. index::
    single: scientific programming; PyTables

* `PyTables <http://www.pytables.org>`_ --- manage large data sets.

.. index::
    single: scientific programming; CVXPY

* `CVXPY <https://github.com/cvxgrp/cvxpy>`_ --- convex optimization in Python.




Learn More
============


* Browse some Python projects on `GitHub <https://github.com/trending?l=python>`_.

* Read more about `Python's history and rise in popularity <https://www.welcometothejungle.com/en/articles/btc-python-popular>`_ .

* Have a look at `some of the Jupyter notebooks <http://nbviewer.jupyter.org/>`_ people have shared on various scientific topics.

.. index::
    single: Python; PyPI

* Visit the `Python Package Index <https://pypi.org/>`_.

* View some of the questions people are asking about Python on `Stackoverflow <http://stackoverflow.com/questions/tagged/python>`_.

* Keep up to date on what's happening in the Python community with the `Python subreddit <https://www.reddit.com:443/r/Python/>`_.

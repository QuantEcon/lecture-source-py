.. _heavy_tails:

.. include:: /_static/includes/header.raw

.. highlight:: python3

**************************
Heavy-Tailed Distributions
**************************

.. index::
    single: Heavy-tailed distributions

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon
  !pip install --upgrade yfinance



Overview
========


Most commonly used probability distributions in classical statistics and
the natural sciences have either bounded support or light tails.  

When a distribution is light-tailed, extreme observations are rare and 
draws tend not to deviate too much from the mean.  

Having internalized these kinds of distributions, many researchers and
practitioners use rules of thumb such as "outcomes more than four or five
standard deviations from the mean can safely be ignored."

However, some distributions encountered in economics have far more probability
mass in the tails than distributions like the normal distribution.

With such **heavy-tailed** distributions, what would be regarded as extreme
outcomes for someone accustomed to thin tailed distributions occur relatively
frequently.

Examples of heavy-tailed distributions observed in economic and financial
settings include

* the income distributions and the wealth distribution (see, e.g., :cite:`pareto1896cours`, :cite:`benhabib2018skewed`),

* the firm size distribution (:cite:`axtell2001zipf`, :cite:`gabaix2016power`}),

* the distribution of returns on holding assets over short time horizons (:cite:`mandelbrot1963variation`, :cite:`rachev2003handbook`), and

* the distribution of city sizes (:cite:`rozenfeld2011area`, :cite:`gabaix2016power`).

These heavy tails turn out to be important for our understanding economic outcomes and their impact.

As one example, the heaviness of the tail in the wealth distribution is one
natural measure of inequality. 

It matters for taxation and redistribution
policies, as well as for flow-on effects for productivity growth, business
cycles, and political economy 

* see, e.g., :cite:`acemoglu2002political`, :cite:`glaeser2003injustice`, :cite:`bhandari2018inequality` or :cite:`ahn2018inequality`).  

This lecture formalizes some of the concepts introduced above and reviews the
key ideas.

Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

    from scipy.stats import cauchy



Visual Comparisons
==================


One way to build intuition on the difference between light and heavy tails is
to plot independent draws and compare them side-by-side.  

A Simulation
-------------

The figure below shows a simulation.

The top two subfigures each show 120 independent draws from the normal distribution, which is light-tailed.

The bottom subfigure shows 120 independent draws from `the Cauchy distribution <https://en.wikipedia.org/wiki/Cauchy_distribution>`__, which is heavy-tailed.


.. _light_heavy_fig1:

.. figure::
    /_static/lecture_specific/heavy_tails/light_heavy_fig1.png

In the top subfigure, the standard deviation of the normal distribution is 2,
and the draws are clustered around the mean.  

In the middle subfigure, the standard deviation is increased to 12 and, as expected, the amount of dispersion rises.  

The bottom subfigure, with the Cauchy draws, shows a
different pattern: tight clustering around the mean for the great majority of
observations, combined with a few sudden large deviations from the mean.

This is typical of a heavy-tailed distribution.



Heavy Tails in Asset Returns
----------------------------

Next let's look at some financial data.

Our aim is to plot the daily change in the price of Amazon (AMZN) stock for
the period 1st January 2015 to 1st November 2019, 

This equates to daily returns if we set dividends aside.  

The code below produces the desired plot using Yahoo financial data via the ``yfinance`` library.

.. code-block:: python3

    import yfinance as yf
    import pandas as pd

    s = yf.download('AMZN', '2015-1-1', '2019-11-1')['Adj Close']
    
    r = s.pct_change()
    
    fig, ax = plt.subplots()
    
    ax.plot(r, linestyle='', marker='o', alpha=0.5, ms=4)
    ax.vlines(r.index, 0, r.values, lw=0.2)
    
    ax.set_ylabel('returns', fontsize=12)
    ax.set_xlabel('date', fontsize=12)
    
    plt.show()


Five of the 1217 observations are more than 5 standard
deviations from the mean.  

Overall, the figure is suggestive of heavy tails,
although not to the same degree as the Cauchy distribution the 
figure above.  

If, however, one takes tick-by-tick data rather
daily data, the heavy-tailedness of the distribution increases further.

.. See, e.g., https://www.researchgate.net/figure/Financial-returns-of-high-frequency-trading-observations-per-minute-from-Bank-of_fig1_330700816

.. Ideally, get AMZN tick-by-tick data



Exercises
=========


Exercise 1
----------

Replicate :ref:`this figure <light_heavy_fig1>` .  Use ``np.random.seed(11)``
to set the seed.


Solutions
=========


Exercise 1
----------

.. code-block:: python3


    n = 120
    np.random.seed(11)

    fig, axes = plt.subplots(3, 1, figsize=(6, 12))

    for ax in axes:
        ax.set_ylim((-120, 120))

    s_vals = 2, 12

    for ax, s in zip(axes[:2], s_vals):
        data = np.random.randn(n) * s
        ax.plot(list(range(n)), data, linestyle='', marker='o', alpha=0.5, ms=4)
        ax.vlines(list(range(n)), 0, data, lw=0.2)
        ax.set_title(f"draws from $N(0, \sigma^2)$ with $\sigma = {s}$", fontsize=11)

    ax = axes[2]
    distribution = cauchy()
    data = distribution.rvs(n)
    ax.plot(list(range(n)), data, linestyle='', marker='o', alpha=0.5, ms=4)
    ax.vlines(list(range(n)), 0, data, lw=0.2)
    ax.set_title(f"draws from the Cauchy distribution", fontsize=11)

    plt.subplots_adjust(hspace=0.25)

    plt.show()


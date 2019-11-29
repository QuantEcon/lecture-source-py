.. include:: /_static/includes/header.raw

.. highlight:: python3

**********************************
Kesten Processes and Firm Dynamics
**********************************

.. index::
    single: Linear State Space Models

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon
  !pip install --upgrade yfinance


Overview
========

:doc:`Previously <linear_models>` we learned about linear stochastic processes
(linear state space models).

Now we generalize these linear models slightly by allowing the multiplicative component to be stochastic.  

Such processes are known as Kesten processes after German--American mathematician Harry Kesten (1931--2019)

Although simple to write down, Kesten processes are interesting for at least two reasons:

1. A number of significant economic processes are or can be described as Kesten processes.

2. Kesten processes generate interesting dynamics, including, in some cases, heavy-tailed cross-sectional distributions.


We will discuss these issues as we go along.

Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

The following two lines can be added to avoid an annoying FutureWarning, and prevent a specific compatibility issue between pandas and matplotlib from causing problems down the line:

.. code-block:: ipython
    
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()


Kesten Processes
=================

.. index::
    single: Kesten processes; heavy tails


A **Kesten process** is a stochastic process of the form

.. math::
    :label: kesproc

    x_{t+1} = a_{t+1} x_t + \eta_{t+1}
    \quad \text{with } x_0 \text{ given}

where :math:`\{a_t\}_{t \geq 1}` and :math:`\{\eta_t\}_{t \geq 1}` are IID
sequences.

We will focus on the nonnegative scalar case, where :math:`x_t` takes values
in :math:`\mathbb R_+`.

In particular, we will assume that 

* the initial condition :math:`x_0` is nonnegative,

* :math:`\{a_t\}_{t \geq 1}` is a nonnegative IID stochastic process and

* :math:`\{\eta_t\}_{t \geq 1}` is another nonnegative IID stochastic process, independent of the first.




Example: GARCH Volatility
-------------------------


The GARCH model is common in financial settings, where time series such as asset returns exhibit time varying volatility.

For example, consider the following plot of daily returns on the Nasdaq
Composite Index for the period 1st January 2006 to 1st November 2019. 

.. _ndcode:

.. code-block:: python3

    import yfinance as yf
    import pandas as pd

    s = yf.download('^IXIC', '2006-1-1', '2019-11-1')['Adj Close']
    
    r = s.pct_change()
    
    fig, ax = plt.subplots()
    
    ax.plot(r, alpha=0.7)
    
    ax.set_ylabel('returns', fontsize=12)
    ax.set_xlabel('date', fontsize=12)
    
    plt.show()


Notice how the series exhibits bursts of volatility (high variance) and then
settles down again.

GARCH models can replicate this feature.

The GARCH(1, 1) volatility process takes the form

.. math::
    :label: garch11v

    \sigma_{t+1}^2 = \alpha_0 + \sigma_t^2 (\alpha_1 \xi_{t+1}^2 + \beta)
    
where :math:`\{\xi_t\}` is IID with :math:`\mathbb E \xi_t^2 = 1` and all parameters are positive.  

Returns on a given asset are then modeled as

.. math::
    :label: garch11r

    r_t = \sigma_t \zeta_{t+1}

where :math:`\{\zeta_t\}` is again IID and independent of :math:`\{\xi_t\}`.

Notice that the volatility sequence :math:`\{\sigma_t\}`, which drives the dynamics, is a Kesten process.


Example: Wealth Dynamics
------------------------

Suppose that a given household saves a fixed fraction :math:`s` of its current wealth in every period.

The household earns labor income :math:`y_t` at the start of time :math:`t`.

Wealth then evolves according to 

.. math::
    :label: wealth_dynam

    w_{t+1} = R_{t+1} s w_t  + y_{t+1}
    
where :math:`\{R_t\}` is the gross rate of return on assets.

If :math:`\{R_t\}` and :math:`\{y_t\}` are both IID, then :eq:`wealth_dynam`
is a Kesten process.


Stationarity
------------

In earlier lectures on :doc:`Markov chains <finite_markov>` and :doc:`linear state space models <linear_models>`, we introduced the notion of a stationary distribution.

In the present context, we can define a stationary distribution as follows:

The distribution :math:`F^*` on :math:`\mathbb R` is called **stationary** for the
Kesten process :eq:`kesproc` if

.. math::
    :label: kp_stationary0

    x_t \sim F^* 
    \quad \implies \quad 
    a_{t+1} x_t + \eta_{t+1} \sim F^*

In other words, if the current state :math:`x_t` has distribution :math:`F^*`,
then so does the next period state :math:`x_{t+1}`.

We can write this alternatively as

.. math::
    :label: kp_stationary

    F^*(y) = \int \mathbb P\{ a_{t+1} x + \eta_{t+1} \leq y\} F^*(dx)
    \quad \text{for all } y \geq 0.

The left hand side is the distribution of the next period state when the
current state is drawn from :math:`F^*`.

The equality in :eq:`kp_stationary` states that this distribution is unchanged.


Cross-Sectional Interpretation
------------------------------

There is an important cross-sectional interpretation of stationary distributions, discussed previously but worth repeating here.

Suppose, for example, that we are interested in the wealth distribution --- that is, the current distribution of wealth across households in a given country.

Suppose further that 

* the wealth of each household evolves independently according to
  :eq:`wealth_dynam`,

* :math:`F^*` is a stationary distribution for this stochastic process and

* there are many households.

Then :math:`F^*` is a steady state for the wealth distribution.

To see this, suppose that :math:`F^*` is the current wealth distribution.

What is the fraction of households with wealth less than :math:`y` next
period?

To obtain this, we sum the probability that wealth is less than :math:`y` tomorrow given that current wealth is :math:`w`, weighted by the fraction of households with wealth :math:`w`.  

.. If we randomly select a household and update it via :eq:`wealth_dynam`, then, by the definition of stationarity, we draw its new wealth according to :math:`F^*`

Noting that the fraction of households with wealth in interval :math:`dw` is :math:`F^*(dw)`, we get


.. math::

    \int \mathbb P\{ R_{t+1} s w  + y_{t+1} \leq y\} F^*(dw)

Since :math:`F^*` is stationary for the wealth process, this is just
:math:`F^*(y)`.

Hence the fraction of households with wealth in :math:`[0, y]` is the same
next period as it is this period.

Since :math:`y` was chosen arbitrarily, the distribution is unchanged.


Conditions for Stationarity
---------------------------

The Kesten process :math:`x_{t+1} = a_{t+1} x_t + \eta_{t+1}` does not always
have a stationary distribution.

For example, if :math:`a_t \equiv \eta_t \equiv 1` for all :math:`t`, then
:math:`x_t = x_0 + t`, which diverges to infinity.

To prevent this kind of divergence, we require that :math:`\{a_t\}` is
strictly less than 1 most of the time.

In particular, if 

.. math::
    :label: kp_stat_cond

    \mathbb E \ln a_t < 0
    \quad \text{and} \quad
    \mathbb E \eta_t < \infty

then a unique stationary distribution exists on :math:`\mathbb R_+`.

* See, for example, theorem 2.1.3 of :cite:`buraczewski2016stochastic`, which provides slightly weaker conditions.

As one application of this result, we see that the wealth process
:eq:`wealth_dynam` will have a unique stationary distribution whenever
:math:`\mathbb E \ln R_t  + \ln s < 0`.


Heavy Tails
===========

Under certain conditions, the stationary distribution of a Kesten process has
a Pareto tail.

(See our :doc:`earlier lecture <heavy_tails>`  on heavy-tailed distributions for a discussion of Pareto tails.)

This fact is highly significant for economics because of the prevalence of
Pareto-tailed distributions.

The Kesten-Goldie Theorem
-------------------------

To state the conditions under which the stationary distribution of a Kesten process has a Pareto tail, we first recall that a random variable is called **nonarithmetic** if its distribution is not concentrated on :math:`t \mathbb Z` for any :math:`t \geq 0`.

For example, any random variable with a density is nonarithmetic.

The famous Kesten--Goldie Theorem (see, e.g., :cite:`buraczewski2016stochastic`, theorem 2.4.4) states that if

1. The stationarity conditions in :eq:`kp_stat_cond` hold,

2. The random variable :math:`a_t` is positive with probability one and nonarithmetic.

3. :math:`\mathbb P\{a_t x + \eta_t = x\} < 1` for all :math:`x \in \mathbb R_+`.

4. There exists a positive constant :math:`\alpha` such that

.. math::

    \mathbb E a_t^\alpha = 1,
        \quad
    \mathbb E \eta_t^\alpha < \infty,
        \quad \text{and} \quad
    \mathbb E [a_t^{\alpha+1} ] < \infty

then the stationary distribution of the Kesten process has a Pareto tail with
tail index :math:`\alpha`.

More precisely, if :math:`F^*` is the unique stationary distribution and :math:`x^* \sim F^*`, then 

.. math::

    \lim_{x \to \infty} x^\alpha \mathbb P\{x^* > x\} = c
    
for some positive constant :math:`c`.


Intuition
---------

Later we will illustrate the Kesten-Goldie Theorem using rank-size plots.

Prior to doing so, we can give the following intuition for the conditions.

Two important conditions are that :math:`\mathbb E \ln a_t < 0`, so the model
is stationary, and :math:`\mathbb E a_t^\alpha = 1` for some :math:`\alpha >
0`.

The first condition implies that the distribution of :math:`a_t` has a large amount of probability mass below 1.

The second condition implies that the distribution of :math:`a_t` has at least some probability mass at or above 1.

The first condition gives us existence of the stationary condition. 

The second condition means that the current state can be expanded by :math:`a_t`.

If this occurs for several concurrent periods, the effects compound each other, since :math:`a_t` is multiplicative.

This leads to spikes in the time series, which fill out the extreme right hand tail of the distribution.

The spikes in the time series are visible in the following simulation, which generates of 10 paths when :math:`a_t` and :math:`b_t` are lognormal.


.. code:: ipython3

    μ = -0.5
    σ = 1.0
    
    def kesten_ts(ts_length=100):
        x = np.zeros(ts_length)
        for t in range(ts_length-1):
            a = np.exp(μ + σ * np.random.randn())
            b = np.exp(np.random.randn())
            x[t+1] = a * x[t] + b
        return x
    
    fig, ax = plt.subplots()
    
    num_paths = 10
    np.random.seed(12)
    
    for i in range(num_paths):
        ax.plot(kesten_ts())
        
    ax.set(xlabel='time', ylabel='$x_t$')    
    plt.show()



Application: Firm Dynamics
==========================


As noted in our :doc:`lecture on heavy tails <heavy_tails>`, for common measures of firm size such as revenue or employment, the US firm size distribution exhibits a Pareto tail (see, e.g., :cite:`axtell2001zipf`, :cite:`gabaix2016power`).

Let us try to explain this rather striking fact using the Kesten--Goldie Theorem.

Gibrat's Law
------------

It was postulated many years ago by Robert Gibrat :cite:`gibrat1931inegalites` that firm size evolves according to a simple rule whereby size next period is proportional to current size.

This is now know as Gibrat's law of proportional growth.

We can express this idea by stating that a suitably defined measure 
:math:`s_t` of firm size obeys

.. math::
    :label: firm_dynam_gb

    \frac{s_{t+1}}{s_t} = a_{t+1} 
    
for some positive IID sequence :math:`\{a_t\}`.

One implication of Gibrat's law is that the growth rate of individual firms
does not depend on their size.

However, over the last few decades, research contradicting Gibrat's law has
accumulated in the literature.

For example, it is commonly found that, on average,

1. small firms grow faster than large firms (see, e.g., :cite:`evans1987relationship` and :cite:`hall1987relationship`) and

2. the growth rate of small firms is more volatile than that of large firms :cite:`dunne1989growth`.

On the other hand, Gibrat's law is generally found to be a reasonable
approximation for large firms :cite:`evans1987relationship`.


We can accommodate these empirical findings by modifying :eq:`firm_dynam_gb`
to

.. math::
    :label: firm_dynam

    s_{t+1} = a_{t+1} s_t + b_{t+1}
    

where :math:`\{a_t\}` and :math:`\{b_t\}` are both IID and independent of each
other.

In the exercises you are asked to show that :eq:`firm_dynam` is more
consistent with the empirical findings presented above than Gibrat's law in
:eq:`firm_dynam_gb`.


Heavy Tails
-----------

So what has this to do with heavy tails?

The answer is that :eq:`firm_dynam` is a Kesten process.

If the conditions of the Kesten-Goldie Theorem are satisfied, then the firm
size distribution is predicted to have heavy tails --- which is exactly what
we see in the data.

In the exercises below we explore this idea further, generalizing the firm
size dynamics and examining the corresponding rank-size plots.

We also try to illustrate why the Pareto tail finding is significant for
quantitative analysis.


Exercises
=========

Exercise 1
----------

Simulate and plot 15 years of daily returns (consider each year as having 250
working days) the GARCH(1, 1) process in :eq:`garch11v`--:eq:`garch11r`.

Take :math:`\xi_t` and :math:`\zeta_t` to be independent and standard normal.

Set :math:`\alpha_0 = 0.00001, \alpha_1 = 0.1, \beta = 0.9` and :math:`\sigma_0 = 0`.
    
Compare visually with the Nasdaq Composite Index returns :ref:`shown above <ndcode>`.

While the time path differs, you should see bursts of high volatility.


Exercise 2
----------


In our discussion of firm dynamics, it was claimed that :eq:`firm_dynam` is more consistent with the empirical literature than Gibrat's law in :eq:`firm_dynam_gb`.

(The empirical literature was reviewed immediately above :eq:`firm_dynam`.) 

In what sense is this true (or false)?

.. Exercise 3

    .. math::
        :label: firm_dynam_ee

        s_{t+1} = e_{t+1} \mathbb 1 \{s_t < \bar s\}
            + (a_{t+1} s_t + b_{t+1}) \mathbb 1 \{s_t \geq \bar s\}

    Generate rank-size plot.



Solutions
=========


Exercise 1
----------

Here is one solution:


.. code:: ipython3

    α_0 = 1e-5
    α_1 = 0.1
    β = 0.9
    
    years = 15
    days = years * 250
    
    def garch_ts(ts_length=days):
        σ2 = 0
        r = np.zeros(ts_length)
        for t in range(ts_length-1):
            ξ = np.random.randn()
            σ2 = α_0 + σ2 * (α_1 * ξ**2 + β)
            r[t] = np.sqrt(σ2) * np.random.randn()
        return r
    
    fig, ax = plt.subplots()
    
    np.random.seed(12)
    
    ax.plot(garch_ts(), alpha=0.7)
        
    ax.set(xlabel='time', ylabel='$\\sigma_t^2$')    
    plt.show()


Exercise 2
----------

The empirical findings are that


1. small firms grow faster than large firms  and

2. the growth rate of small firms is more volatile than that of large firms.

Also, Gibrat's law is generally found to be a reasonable approximation for
large firms than for small firms 

The claim is that the dynamics in :eq:`firm_dynam` are more consistent with
points 1-2 than Gibrat's law.

To see why, we rewrite :eq:`firm_dynam` in terms of growth dynamics:

.. math::
    :label: firm_dynam_2

    \frac{s_{t+1}}{s_t} = a_{t+1} + \frac{b_{t+1}}{s_t}

Taking :math:`s_t = s` as given, the mean and variance of firm growth are

.. math::

    \mathbb E a
    + \frac{\mathbb E b}{s}
    \quad \text{and} \quad
    \mathbb V a
    + \frac{\mathbb V b}{s^2}
    
Both of these decline with firm size :math:`s`, consistent with the data.

Moreover, the law of motion :eq:`firm_dynam_2` clearly approaches Gibrat's law
:eq:`firm_dynam_gb` as :math:`s_t` gets large.


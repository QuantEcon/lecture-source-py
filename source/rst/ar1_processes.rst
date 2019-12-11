.. _ar1:

.. include:: /_static/includes/header.raw

.. highlight:: python3

*************
AR1 Processes
*************

.. index::
    single: Autoregressive processes

.. contents:: :depth: 2


Overview
========

In this lecture we are going to study a very simple class of stochastic
models called AR(1) processes.

These simple models are used again and again in economic research to represent the dynamics of series such as

* labor income
* dividends
* productivity
* etc.

AR(1) processes can take negative values but are easily converted into positive processes by a transformation such as exponentiation.

We are going to study AR(1) processes partly because they are useful and
partly because they help us understand important concepts.

Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

The AR(1) Model
===============

The **AR(1) model** (autoregressive model of order 1) takes the form

.. math::
    :label: can_ar1

    X_{t+1} = a X_t + b + c W_{t+1}

where :math:`a, b, c` are scalar-valued parameters.

This law of motion generates a time series :math:`\{ X_t\}` as soon as we
specify an initial condition :math:`X_0`.

This is called the **state process** and the state space is :math:`\mathbb R`.

To make things even simpler, we will assume that

* the process :math:`\{ W_t \}` is IID and standard normal,
* the initial condition :math:`X_0` is drawn from the normal distribution :math:`N(\mu_0, \sigma_0^2)` and
* the initial condition :math:`X_0` is independent of :math:`\{ W_t \}`.



Moving Average Representation
-----------------------------

Iterating backwards from time :math:`t`, we obtain

.. math::

   X_t = a X_{t-1} + b +  c W_t 
           = a^2 X_{t-2} + a b + a c W_{t-1} + b + c W_t 
           = \cdots

If we work all the way back to time zero, we get

.. math::
    :label: ar1_ma

       X_t = a^t X_0 + b \sum_{j=0}^{t-1} a^j +
            c \sum_{j=0}^{t-1} a^j  W_{t-j} 

Thus, we see that :math:`X_t` is a well defined random variable that depends
on 

* the parameters,
* the initial condition :math:`X_0` and 
* the shocks :math:`W_0, \ldots W_t`.

Throughout, the symbol :math:`\psi_t` will be used to refer to the
density of this random variable :math:`X_t`.


Distribution Dynamics
---------------------

One of the nice things about this model is that it's so easy to trace out the sequence of distributions :math:`\{ \psi_t \}` corresponding to the time
series :math:`\{ X_t\}`.

To see this, we first note that :math:`X_t` is normally distributed for each :math:`t`.

This is immediate form :eq:`ar1_ma`, since linear combinations of independent
normal random variables are normal.


Given that :math:`X_t` is normally distributed, we know the full distribution
:math:`\psi_t` if we can work out the first two moments.

Let :math:`\mu_t` and :math:`\sigma_t` denote the mean and standard deviation
of :math:`X_t` respectively.

We can pin down these values from :eq:`ar1_ma` or we can use the following
recursive expressions:

.. math::
    :label: dyn_tm

    \mu_{t+1} = a \mu_t + b
    \quad \text{and} \quad
    \sigma^2_{t+1} = a^2 \sigma^2_t + c^2

These expressions are obtained from :eq:`can_ar1` by taking, respectively, the expectation
and variance of both sides of the equality.

The following code uses these laws of motion to track the sequence of marginal
distributions :math:`\{ \psi_t \}`.

The parameters are

.. code-block:: python3

    a, b, c = 0.9, 0.1, 0.5

    mu, sigma = -3.0, 0.4  # initial condition
    sim_length = 10

    grid = np.linspace(-5, 7, 120)

    from scipy.stats import norm

    fig, ax = plt.subplots()

    for t in range(sim_length):
        mu = a * mu + b
        sigma = np.sqrt(a**2 * sigma**2 + c**2)
        ax.plot(grid, norm.pdf(grid, loc=mu, scale=sigma), 
                label=f"$\psi_{t}$",
                alpha=0.7)

    ax.legend(bbox_to_anchor=[1.05,1],loc=2,borderaxespad=1)

    plt.show()



Stationarity and Asymptotic Stability
=====================================


Notice that, in the figure above, the sequence :math:`\{ \psi_t \}` seems to be converging to a limiting distribution.

This is even clearer if we project forward further into the future:


.. code-block:: python3

    def plot_density_seq(ax, mu_0=-3.0, sigma_0=0.4, sim_length=60):
        mu, sigma = mu_0, sigma_0
        for t in range(sim_length):
            mu = a * mu + b
            sigma = np.sqrt(a**2 * sigma**2 + c**2)
            ax.plot(grid, norm.pdf(grid, loc=mu, scale=sigma),  alpha=0.5)

    fig, ax = plt.subplots()
    plot_density_seq(ax)
    plt.show()


This is indeed the case.

Moreover, the limit does not depend on the initial condition.

For example, this density sequence converges to the same limit.

.. code-block:: python3

    fig, ax = plt.subplots()
    plot_density_seq(ax, mu_0=3.0)
    plt.show()

In fact it's easy to show that such convergence will occur, regardless of the initial condition, whenever :math:`|a| < 1`.

To see this, we just have to look at the dynamics of the first two moments, as
given in :eq:`dyn_tm`.

When :math:`|a| < 1`, these sequence converge to the respective limits

.. math::
    :label: mu_sig_star

    \mu_* := \frac{b}{1-a}
    \quad \text{and} \quad
    \sigma^2_* = \frac{c^2}{1 - a^2}

(See our :doc:`lecture on one dimensional dynamics <scalar_dynam>` for background.)

Hence 

.. math::
    :label: ar1_psi_star

    \psi_t \to \psi_* = N(\mu_*, \sigma^2_*)
    \quad \text{as }
    t \to \infty

We can confirm this works for the sequence above using the following code.


.. code-block:: python3

    fig, ax = plt.subplots()
    plot_density_seq(ax, mu_0=3.0)

    mu_star = b / (1 - a)
    sigma_star = np.sqrt(c**2 / (1 - a**2))
    psi_star = norm.pdf(grid, loc=mu_star, scale=sigma_star)
    ax.plot(grid, psi_star, 'k-', lw=2, label="$\psi_*$")
    ax.legend()

    plt.show()


Stationary Distributions
------------------------

In our lecture on :doc:`finite Markov chains <finite_markov>` and also in our
lecture on :doc:`inventory dynamics <inventory_dynamics>`, we discussed the
notion of stationary distributions.

In general, a stationary distribution is any distribution that is a fixed
point of the update rule for distributions.

A different way to put this, specialized to the current setting, is: a
density :math:`\psi` on :math:`\mathbb R` is **stationary** for the AR(1) process if

.. math::
    X_t \sim \psi
    \quad \implies \quad 
    a X_t + b + c W_{t+1} \sim \psi

The distribution :math:`\psi_*` in :eq:`ar1_psi_star` has this property ---
checking this is an exercise.

(Of course, we are assuming that :math:`|a| < 1` so that :math:`\psi_*` is
well defined.)

In fact, it can be shown that no other distribution on :math:`\mathbb R` has this property.


Ergodicity
==========

In :doc:`our lecture on finite Markov chains <finite_markov>`, we explored the
notion of ergodicity.

In that setting we found that, under some conditions that implied stability,
averages over time series converge to expectations under the stationary
distribution.

The same phenomenon can be observed here.


Indeed, it can be proved that, whenever :math:`|a| < 1`, we have

.. math::
    :label: ar1_ergo

    \frac{1}{m} \sum_{t = 1}^m h(X_t)  \to 
    \int h(x) \psi^*(x) dx
        \quad \text{as } m \to \infty

provided that the integral on the right hand side is finite.

In :eq:`ar1_ergo`, convergence holds with probability one.

The textbook by :cite:`MeynTweedie2009` is a classic reference on this topic.

For example, if we consider the indicator function :math:`h(x) = x`, we get

.. math::

    \frac{1}{m} \sum_{t = 1}^m X_t  \to 
    \int x \psi^*(x) dx
        \quad \text{as } m \to \infty

In other words, the time series sample mean converges to the mean of the
stationary distribution.


Exercises
=========

Exercise 1
----------

Let :math:`k` be a natural number. 

The :math:`k`-th central moment of a  random variable is defined as 

.. math::

    M_k := \mathbb E [ (X - \mathbb E X )^k ]

When that random variable is :math:`N(\mu, \sigma^2)`, it is know that

.. math::

    M_k = 
    \begin{cases}
        0 & \text{ if } k \text{ is even} \\
        \sigma^k (k-1)!! & \text{ if } k \text{ is odd} 
    \end{cases}

Here :math:`n!!` is the double factorial.

According to :eq:`ar1_ergo`, we should have, for any :math:`k \in \mathbb N`,

.. math::

    \frac{1}{m} \sum_{t = 1}^m 
        (X_t - \mu_* )^k
        \approx M_k

when :math:`m` is large.

Confirm this by simulation at a range of :math:`k` using the default parameters from the lecture.


Exercise 2
-----------

Write your own version of a one dimensional `kernel density
estimator <https://en.wikipedia.org/wiki/Kernel_density_estimation>`__,
which estimates a density from a sample.

Write it as a class that takes the data :math:`X` and bandwidth
:math:`h` when initialized and provides a method :math:`f` such that

.. math::


       f(x) = \frac{1}{hn} \sum_{i=1}^n 
       K \left( \frac{x-X_i}{h} \right)

For :math:`K` use the Gaussian kernel (:math:`K` is the standard normal
density).

Write the class so that the bandwidth defaults to Silverman’s rule (see
the “rule of thumb” discussion on `this
page <https://en.wikipedia.org/wiki/Kernel_density_estimation>`__). Test
the class you have written by going through the steps

1. simulate data :math:`X_1, \ldots, X_n` from distribution :math:`\phi`
2. plot the kernel density estimate over a suitable range
3. plot the density of :math:`\phi` on the same figure

for distributions :math:`\phi` of the following types

-  `beta
   distribution <https://en.wikipedia.org/wiki/Beta_distribution>`__
   with :math:`\alpha = \beta = 2`
-  `beta
   distribution <https://en.wikipedia.org/wiki/Beta_distribution>`__
   with :math:`\alpha = 2` and :math:`\beta = 5`
-  `beta
   distribution <https://en.wikipedia.org/wiki/Beta_distribution>`__
   with :math:`\alpha = \beta = 0.5`

Use :math:`n=100`.

Make a comment on your results. (Do you think this is a good estimator
of these distributions?)

Exercise 3
----------

In class we discussed the following fact: For the :math:`AR(1)` process

.. math::  X_{t+1} = a X_t + b + c \xi_{t+1} 

with :math:`\{ \xi_t \}` iid and standard normal,

.. math::

    
       \psi_t = N(\mu, s^2) \implies \psi_{t+1} 
       = N(a \mu + b, a^2 s^2 + c^2) 

Confirm this, at least approximately, by simulation. Let

-  :math:`a = 0.9`
-  :math:`b = 0.0`
-  :math:`c = 0.1`
-  :math:`\mu = -3`
-  :math:`s = 0.2`

First, plot :math:`\psi_t` and :math:`\psi_{t+1}` using the true
distributions described above.

Second, plot :math:`\psi_{t+1}` on the same figure (in a different
color) as follows:

1. Generate :math:`n` draws of :math:`X_t` from the :math:`N(\mu, s^2)`
   distribution
2. Update them all using the rule
   :math:`X_{t+1} = a X_t + b + c \xi_{t+1}`
3. Use the resulting sample of :math:`X_{t+1}` values to produce a
   density estimate via kernel density estimation.

Try this for several different values of :math:`n` and confirm that the
simulation based estimate of :math:`\psi_{t+1}` does converge to the
theoretical distribution.

Solutions
=========


Exercise 1
----------

.. code-block:: python3

    from numba import njit
    from scipy.special import factorial2

    @njit
    def sample_moments_ar1(k, m=100_000, mu_0=0.0, sigma_0=1.0, seed=1234):
        np.random.seed(seed)
        sample_sum = 0.0
        x = mu_0 + sigma_0 * np.random.randn()
        for t in range(m):
            sample_sum += (x - mu_star)**k
            x = a * x + b + c * np.random.randn()
        return sample_sum / m

    def true_moments_ar1(k):
        if k % 2 == 0:
            return sigma_star**k * factorial2(k - 1)
        else:
            return 0

    k_vals = np.arange(6) + 1
    sample_moments = np.empty_like(k_vals)
    true_moments = np.empty_like(k_vals)

    for k_idx, k in enumerate(k_vals):
        sample_moments[k_idx] = sample_moments_ar1(k)
        true_moments[k_idx] = true_moments_ar1(k)

    fig, ax = plt.subplots()
    ax.plot(k_vals, true_moments, label="true moments")
    ax.plot(k_vals, sample_moments, label="sample moments")
    ax.legend()

    plt.show()



Exercise 2
-----------

Here is one solution:

.. code:: ipython3

    K = norm.pdf
    
    class KDE:
        
        def __init__(self, x_data, h=None):
    
            if h is None:
                c = x_data.std()
                n = len(x_data)
                h = 1.06 * c * n**(-1/5)
            self.h = h
            self.x_data = x_data
    
        def f(self, x):
            if np.isscalar(x):
                return K((x - self.x_data) / self.h).mean() * (1/self.h)
            else:
                y = np.empty_like(x)
                for i, x_val in enumerate(x):
                    y[i] = K((x_val - self.x_data) / self.h).mean() * (1/self.h)
                return y

.. code:: ipython3

    
    def plot_kde(ϕ, x_min=-0.2, x_max=1.2):
        x_data = ϕ.rvs(n)
        kde = KDE(x_data)
        
        x_grid = np.linspace(-0.1, 1.1, 100)
        fig, ax = plt.subplots()
        ax.plot(x_grid, kde.f(x_grid), label="estimate")
        ax.plot(x_grid, ϕ.pdf(x_grid), label="true density")
        ax.legend()
        plt.show()

.. code:: ipython3

    from scipy.stats import beta
    
    n = 100
    parameter_pairs= (2, 2), (2, 5), (0.5, 0.5)
    for α, β in parameter_pairs:
        plot_kde(beta(α, β))

We see that the kernel density estimator is effective when the underlying
distribution is smooth but less so otherwise.

Exercise 3
----------

Here is our solution

.. code:: ipython3

    a = 0.9
    b = 0.0
    c = 0.1
    μ = -3
    s = 0.2

.. code:: ipython3

    μ_next = a * μ + b
    s_next = np.sqrt(a**2 * s**2 + c**2)

.. code:: ipython3

    ψ = lambda x: K((x - μ) / s)
    ψ_next = lambda x: K((x - μ_next) / s_next)

.. code:: ipython3

    ψ = norm(μ, s)
    ψ_next = norm(μ_next, s_next)

.. code:: ipython3

    n = 1000
    x_draws = ψ.rvs(n)
    x_draws_next = a * x_draws + b + c * np.random.randn(n)
    kde = KDE(x_draws_next)
    
    x_grid = np.linspace(μ - 1, μ + 1, 100)
    fig, ax = plt.subplots()
    
    ax.plot(x_grid, ψ.pdf(x_grid), label="$\psi_t$")
    ax.plot(x_grid, ψ_next.pdf(x_grid), label="$\psi_{t+1}$")
    ax.plot(x_grid, kde.f(x_grid), label="estimate of $\psi_{t+1}$")
    
    ax.legend()
    plt.show()


The simulated distribution approximately coincides with the theoretical
distribution, as predicted.

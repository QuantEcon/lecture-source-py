
.. include:: /_static/includes/header.raw

.. highlight:: python3

******************
Inventory Dynamics
******************

.. index::
    single: Markov process, inventory

.. contents:: :depth: 2

Overview
========

In this lecture we will study the time path of inventories for firms with 
so-called S-s inventory dynamics.

Such firms allow inventory to fall until it goes below some level :math:`s`.

At that point, they order in sufficent quantities to bring their inventory up
to capacity :math:`S`.

Such policies are common in practice and also optimal in certain
circumstances.

A review and a macroeconomic application can be found in :cite:`caplin1985variability`.

Here our main aim is to learn more about simulation and Markov dynamics.

One change from the :doc:`lecture on finite Markov chains <finite_markov>` is
that the state space will be a continuum.

Let's start with some imports

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

    from numba import njit, jitclass, float64


Sample Paths 
============

Consider a firm with inventory :math:`X_t`.

The firm waits until :math:`X_t \leq s` and then restocks up to :math:`S` units.

It faces stochastic demand :math:`D_{t+1}`, where :math:`\{D_t\}` is
IID with common density :math:`q`.

Inventory follows

.. math::

    X_{t+1} = 
        \begin{cases}
          ( S - D_{t+1})^+ & \quad \text{if } X_t \leq s \\
          ( X_t - D_{t+1} )^+ &  \quad \text{if } X_t > s
        \end{cases}

Here :math:`a^+ := \max\{a, 0\}`.

In what follows, we will assume that :math:`q` is lognormal, so that

.. math::  

    D_t = \exp(\mu + \sigma Z_t)

where :math:`\mu` and :math:`\sigma` are parameters and :math:`\{Z_t\}` is IID
and standard normal.

Here's a class that stores parameters and generates time paths for inventory.


.. code-block:: python3

    firm_data = [
       ('s', float64),          # restock trigger level
       ('S', float64),          # capacity
       ('mu', float64),         # shock location parameter
       ('sigma', float64)       # shock scale parameter
    ]


    @jitclass(firm_data)
    class InventoryModel:

        def __init__(self, s=10, S=100, mu=1.0, sigma=0.5):

            self.s, self.S, self.mu, self.sigma = s, S, mu, sigma

        def update(self, x):
            "Update the state from t to t+1 given current state x."

            Z = np.random.randn()
            D = np.exp(self.mu + self.sigma * Z)
            if x <= self.s:
                return max(self.S - D, 0)
            else:
                return max(x - D, 0)

        def sim_inventory_path(self, x_init, sim_length):

            X = np.empty(sim_length)
            X[0] = x_init

            for t in range(sim_length-1):
                X[t+1] = self.update(X[t])
            return X

        
Let's run a first simulation, of a single path:

.. code:: ipython3

    firm = InventoryModel()

    s, S = firm.s, firm.S
    sim_length = 100
    x_init = 50

    X = firm.sim_inventory_path(x_init, sim_length)

    fig, ax = plt.subplots()
    bbox = (0., 1.02, 1., .102)
    legend_args = {'ncol': 3,
                   'bbox_to_anchor': bbox,
                   'loc': 3,
                   'mode': 'expand'}
    
    ax.plot(X, label="inventory")
    ax.plot(s * np.ones(sim_length), 'k--', label="$s$")
    ax.plot(S * np.ones(sim_length), 'k-', label="$S$")
    ax.set_ylim(0, S+10)
    ax.set_xlabel("time")
    ax.legend(**legend_args)
    
    plt.show()


Now let's simulate multiple paths in order to build a more complete picture of
the probabilities of different outcomes:

.. code:: ipython3

    sim_length=200
    fig, ax = plt.subplots(figsize=(9, 6))
    bbox = (0., 1.02, 1., .102)
    legend_args = {'ncol': 3,
                   'bbox_to_anchor': bbox,
                   'loc': 3,
                   'mode': 'expand'}
    
    ax.plot(s * np.ones(sim_length), 'k--', label="$s$")
    ax.plot(S * np.ones(sim_length), 'k-', label="$S$")
    ax.set_ylim(0, S+10)
    ax.legend(**legend_args)
    
    for i in range(400):
        X = firm.sim_inventory_path(x_init, sim_length)
        ax.plot(X, 'b', alpha=0.2, lw=0.5)
    
    plt.show()


Marginal Distributions
======================

Now let’s look at the marginal distribution :math:`\psi_T` of :math:`X_T` for some
fixed :math:`T`.

We will do this by generating many draws of :math:`X_T` given initial
condition :math:`X_0`.

With these draws of :math:`X_T` we can build up a picture of its distribution :math:`\psi_T` 

Here's one visualization, with :math:`T=50`.

.. code:: ipython3

    T = 50
    M = 200  # Number of draws
    
    ymin, ymax = 0, S + 10
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    
    for ax in axes:
        ax.grid(alpha=0.4)
    
    ax = axes[0]
    
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('$X_t$', fontsize=16)
    ax.vlines((T,), -1.5, 1.5)
    
    ax.set_xticks((T,))
    ax.set_xticklabels((r'$T$',))
    
    sample = np.empty(M)
    for m in range(M):
        X = firm.sim_inventory_path(x_init, 2 * T)
        ax.plot(X, 'b-', lw=1, alpha=0.5)
        ax.plot((T,), (X[T+1],), 'ko', alpha=0.5)
        sample[m] = X[T+1]
    
    axes[1].set_ylim(ymin, ymax)
    
    axes[1].hist(sample, 
                 bins=16, 
                 density=True, 
                 orientation='horizontal', 
                 histtype='bar',
                 alpha=0.5)
    
    plt.show()

We can build up a clearer picture by drawing more samples

.. code:: ipython3

    T = 50
    M = 50_000
    
    fig, ax = plt.subplots()
    
    sample = np.empty(M)
    for m in range(M):
        X = firm.sim_inventory_path(x_init, T+1)
        sample[m] = X[T]
    
    ax.hist(sample, 
             bins=36, 
             density=True, 
             histtype='bar',
             alpha=0.75)
    
    plt.show()

Note that the distribution is bimodal

* Most firms have restocked twice but a few have restocked only once (see figure with paths above).

* Firms in the second category have lower inventory.

We can also approximate the distribution using a kernel density estimator. 

Kernel density estimators can be thought of as smoothed histograms.

They are preferable to histograms when the distribution being estimated is likely to be smooth.

We will use a kernel density estimator from Scikit Learn.

.. code:: ipython3

    from sklearn.neighbors import KernelDensity
    
    def plot_kde(sample, ax, label=''):
        
        xmin, xmax = 0.9 * min(sample), 1.1 * max(sample)
        xgrid = np.linspace(xmin, xmax, 200)
        kde = KernelDensity(kernel='gaussian').fit(sample[:, None])
        log_dens = kde.score_samples(xgrid[:, None])
    
        ax.plot(xgrid, np.exp(log_dens), label=label)

.. code:: ipython3

    fig, ax = plt.subplots()
    plot_kde(sample, ax)
    plt.show()



Exercises
=========

Exercise 1
----------

This model is asymptotically stationary, with a unique stationary
distribution.

(See :doc:`our lecture on finite state Markov chains <finite_markov>` for
background --- the state space is now continuous but the fundamental concepts
are the same.)

In particular, the sequence of marginal distributions :math:`\{\psi_t\}`
is converging to a unique limiting distribution that does not depend on
initial conditions.

Although we will not prove this here, we can investigate it using simulation.

Your task is to generate and plot the sequence :math:`\{\psi_t\}` at times
:math:`t = 10, 50, 200, 300, 400` using the code above.

You should see convergence. Try different initial conditions to verify
that, in the long run, the distribution is invariant across initial
conditions.

Exercise 2
----------

Using simulation, calculate the probability that we need to order twice or more in the first
50 periods?

Solutions
=========

Exercise 1
----------

Here’s one possible solution:

.. code:: ipython3

    M = 50_000  # Number of paths
    T = 501
    X = np.empty((M, T))

.. code:: ipython3

    for m in range(M):
        X[m, :] = firm.sim_inventory_path(x_init, T)
    
    fig, ax = plt.subplots()
    sample_dates = 10, 50, 200, 300, 400
    
    for t in sample_dates:
        plot_kde(X[:, t], ax, label=f't = {t}')
    
    ax.set_xlabel('inventory')
    ax.set_ylabel('probability')
    ax.legend()
    plt.show()

Notice that by :math:`t=300` or :math:`t=400` the densities are barely
changing.

We have reached a good approximation of the stationary density.

You can convince yourself that initial conditions don’t matter by
testing a few of them.

.. code:: ipython3

    for m in range(M):
        X[m, :] = firm.sim_inventory_path(80, T)
    
    fig, ax = plt.subplots()
    sample_dates = 10, 50, 200, 300, 400
    
    for t in sample_dates:
        plot_kde(X[:, t], ax, label=f't = {t}')
    
    ax.set_xlabel('inventory')
    ax.set_ylabel('probability')
    ax.legend()
    plt.show()


Exercise 2
----------

Here is one solution.

.. code:: ipython3

    sim_length = 50
    M = 100_000  
    counter = 0
    
    for m in range(M):
        X = firm.sim_inventory_path(x_init, sim_length)
        if sum(X <= s) > 1:
            counter += 1
            
    print(f"Frequency of at least two stock outs = {counter / M}")



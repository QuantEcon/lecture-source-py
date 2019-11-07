.. _parallel:

.. include:: /_static/includes/header.raw

***************
Parallelization
***************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon


**Co-author: Matthew McKay**

  
Overview
========


The growth of CPU clock speed (i.e., single thread execution speed) has slowed
dramatically in recent years, due to inherent physical limitations on the
construction of individual chips.

Chip designers and computer programmers have responded by seeking a different
path to fast execution: parallelization.

On the part of hardware makers, this has lead to rapid growth in the number of
cores (physical CPUs) embedded in each machine.

For programmers, the challenge has been to exploit these multiple CPUs by 
running many processes in parallel (i.e., simultaneously).

This is particularly important in scientific programming, where we need to
handle

* large amounts of data and

* CPU intensive simulations and other calculations.

In this lecture we discuss parallelization for scientific computing, with a
focus on

#. tools for parallelization in Python and 

#. how these tools can be applied to quantitative economic problems.



Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    import matplotlib.pyplot as plt

    %matplotlib inline


Types of Parallelization
========================

Large textbooks have been written on this topic but we don't want to spend too
much time on the details.

Instead, let's keep a tight focus on what's useful to us

Let's briefly review the two main kinds of parallelization and how they
relate to modern scientific computing

Multiprocessing
---------------


Multithreading
--------------


Advantages and Disadvantages
----------------------------

Multithreading is more lightweight because most system and memory resources
are shared by the threads. 

In addition, the fact that multiple threads all access a shared pool of memory
is very convenient in terms of numerical programming.

On the other hand, multiprocessing is more flexible and can be distributed
across clusters.

For the great majority of what we do in these lectures, multithreading will
suffice.


Implicit Multithreading in NumPy
================================

To be added.  Give an example and add a screenshot of htop.



Multithreading with Numba
=========================


.. code-block:: ipython

    from numpy.random import randn

    @jit
    def h(w, r=0.1, s=0.3, v1=0.1, v2=1.0):
        """
        Dynamics of wealth.
        """

        # Draw shocks
        R = np.exp(v1 * randn()) * (1 + r)
        Y = np.exp(v2 * randn())

        # Update wealth
        w = R * s * w + Y
        return w

.. code-block:: ipython

    fig, ax = plt.subplots(figsize=(9, 6))

    T = 100
    w = np.empty(T)
    w[0] = 5
    for t in range(T-1):
        w[t+1] = h(w[t])

    ax.plot(range(T), w)


.. code-block:: ipython

    @jit
    def compute_long_run_median(w0=5, T=1000, num_reps=50_000):

        obs = np.empty(num_reps)
        for i in range(num_reps):
            w = w0
            for t in range(T):
                w = h(w)
            obs[i] = w

        return np.median(obs)


.. code-block:: ipython

    %%time
    compute_long_run_median()

.. code-block:: ipython

    @jit(parallel=True)
    def compute_long_run_median_parallel(w0=5, T=1000, num_reps=50_000):

        obs = np.empty(num_reps)
        for i in prange(num_reps):
            w = w0
            for t in range(T):
                w = h(w)
            obs[i] = w

        return np.median(obs)


.. code-block:: ipython

    %%time
    compute_long_run_median_parallel()


Dask
====

To be added.


GPUs
=====

Just say a few words about them.  How do they relate to the foregoing? Explain that we can't introduce executable
GPU code here.


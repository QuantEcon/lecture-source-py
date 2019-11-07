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

Overview
========

foobar

Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    import matplotlib.pyplot as plt

    %matplotlib inline


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



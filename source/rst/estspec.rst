.. _estspec:

.. include:: /_static/includes/header.raw

.. highlight:: python3

******************************************
Estimation of :index:`Spectra`
******************************************

.. index::
    single: Spectra; Estimation

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
============

In a :ref:`previous lecture <arma>`, we covered some fundamental properties of covariance stationary linear stochastic processes.

One objective for that lecture was to introduce spectral densities --- a standard and very useful technique for analyzing such processes.

In this lecture, we turn to the problem of estimating spectral densities and other related quantities from data.

.. index::
    single: Spectra, Estimation; Fast Fourier Transform

Estimates of the spectral density are computed using what is known as a periodogram --- which in
turn is computed via the famous `fast Fourier transform <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`_.

Once the basic technique has been explained, we will apply it to the analysis of several key macroeconomic time series.

For supplementary reading, see :cite:`Sargent1987` or :cite:`CryerChan2008`.

Let's start with some standard imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    from quantecon import ARMA, periodogram, ar_periodogram


.. _periodograms:

:index:`Periodograms`
=====================

:ref:`Recall that <arma_specd>` the spectral density :math:`f` of a covariance stationary process with
autocorrelation function :math:`\gamma` can be written

.. math::

    f(\omega) = \gamma(0) + 2 \sum_{k \geq 1} \gamma(k) \cos(\omega k),
    \qquad \omega \in \mathbb R


Now consider the problem of estimating the spectral density of a given time series, when :math:`\gamma` is unknown.

In particular, let :math:`X_0, \ldots, X_{n-1}` be :math:`n` consecutive observations of a single time series that is assumed to be covariance stationary.

The most common estimator of the spectral density of this process is the *periodogram* of :math:`X_0, \ldots, X_{n-1}`, which is defined as

.. math::
    :label: estspec_p

    I(\omega)
    := \frac{1}{n} \left| \sum_{t=0}^{n-1} X_t e^{i t \omega} \right|^2,
    \qquad \omega \in \mathbb R


(Recall that :math:`|z|` denotes the modulus of complex number :math:`z`)

Alternatively, :math:`I(\omega)` can be expressed as

.. math::

    I(\omega)
    = \frac{1}{n}
    \left\{
    \left[\sum_{t=0}^{n-1} X_t \cos(\omega t) \right]^2 +
    \left[\sum_{t=0}^{n-1} X_t \sin(\omega t) \right]^2
    \right\}


It is straightforward to show that the function :math:`I` is even and :math:`2
\pi`-periodic (i.e., :math:`I(\omega) = I(-\omega)` and :math:`I(\omega +
2\pi) = I(\omega)` for all :math:`\omega \in \mathbb R`).

From these two results, you will be able to verify that the values of
:math:`I` on :math:`[0, \pi]` determine the values of :math:`I` on all of
:math:`\mathbb R`.

The next section helps to explain the connection between the periodogram and the spectral density.


Interpretation
----------------

.. index::
    single: Periodograms; Interpretation

To interpret the periodogram, it is convenient to focus on its values at the *Fourier frequencies*

.. math::

    \omega_j := \frac{2 \pi j}{n},
    \quad j = 0, \ldots, n - 1


In what sense is :math:`I(\omega_j)` an estimate of :math:`f(\omega_j)`?

The answer is straightforward, although it does involve some algebra.

With a bit of effort, one can show that for any integer :math:`j > 0`,

.. math::

    \sum_{t=0}^{n-1} e^{i t \omega_j }
    = \sum_{t=0}^{n-1} \exp \left\{ i 2 \pi j \frac{t}{n} \right\} = 0


Letting :math:`\bar X` denote the sample mean :math:`n^{-1} \sum_{t=0}^{n-1} X_t`, we then have

.. math::

    n I(\omega_j)
     = \left| \sum_{t=0}^{n-1} (X_t - \bar X) e^{i t \omega_j } \right|^2
     =  \sum_{t=0}^{n-1} (X_t - \bar X) e^{i t \omega_j }
    \sum_{r=0}^{n-1} (X_r - \bar X) e^{-i r \omega_j }


By carefully working through the sums, one can transform this to

.. math::

    n I(\omega_j) = \sum_{t=0}^{n-1} (X_t - \bar X)^2 +
    2 \sum_{k=1}^{n-1} \sum_{t=k}^{n-1} (X_t - \bar X)(X_{t-k} - \bar X)
    \cos(\omega_j k)


Now let

.. math::

    \hat \gamma(k)
    := \frac{1}{n} \sum_{t=k}^{n-1} (X_t - \bar X)(X_{t-k} - \bar X),
    \qquad k = 0,1,\ldots, n-1


This is the sample autocovariance function, the natural "plug-in estimator" of the :ref:`autocovariance function <arma_defs>` :math:`\gamma`.

("Plug-in estimator" is an informal term for an estimator found by replacing expectations with sample means)

With this notation, we can now write

.. math::

    I(\omega_j) = \hat \gamma(0) +
    2 \sum_{k=1}^{n-1} \hat \gamma(k) \cos(\omega_j k)


Recalling our expression for :math:`f` given :ref:`above <periodograms>`,
we see that :math:`I(\omega_j)` is just a sample analog of :math:`f(\omega_j)`.


Calculation
--------------

.. index::
    single: Periodograms; Computation

Let's now consider how to compute the periodogram as defined in :eq:`estspec_p`.

There are already functions available that will do this for us
--- an example is ``statsmodels.tsa.stattools.periodogram`` in the ``statsmodels`` package.

However, it is very simple to replicate their results, and this will give us a platform to make useful extensions.

The most common way to calculate the periodogram is via the discrete Fourier transform,
which in turn is implemented through the `fast Fourier transform <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`_ algorithm.

In general, given a sequence :math:`a_0, \ldots, a_{n-1}`, the discrete
Fourier transform computes the sequence

.. math::

    A_j := \sum_{t=0}^{n-1} a_t \exp \left\{ i 2 \pi \frac{tj}{n} \right\},
    \qquad j = 0, \ldots, n-1


With ``numpy.fft.fft`` imported as ``fft`` and :math:`a_0, \ldots, a_{n-1}` stored in NumPy array ``a``, the function call ``fft(a)`` returns the values :math:`A_0, \ldots, A_{n-1}` as a NumPy array.

It follows that when the data :math:`X_0, \ldots, X_{n-1}` are stored in array ``X``, the values :math:`I(\omega_j)` at the Fourier frequencies, which are given by

.. math::

    \frac{1}{n} \left| \sum_{t=0}^{n-1} X_t \exp \left\{ i 2 \pi \frac{t j}{n} \right\} \right|^2,
    \qquad j = 0, \ldots, n-1


can be computed by ``np.abs(fft(X))**2 / len(X)``.

Note: The NumPy function ``abs`` acts elementwise, and correctly handles complex numbers (by computing their modulus, which is exactly what we need).

A function called ``periodogram`` that puts all this together can be found `here <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/estspec.py>`__.

Let's generate some data for this function using the ``ARMA`` class from `QuantEcon.py <https://github.com/QuantEcon/QuantEcon.py>`__ (see the :ref:`lecture on linear processes <arma>` for more details).

Here's a code snippet that, once the preceding code has been run, generates data from the process

.. math::
    :label: esp_arma

    X_t = 0.5 X_{t-1} + \epsilon_t - 0.8 \epsilon_{t-2}


where :math:`\{ \epsilon_t \}` is white noise with unit variance, and compares the periodogram to the actual spectral density




.. code-block:: ipython

    n = 40                          # Data size
    ϕ, θ = 0.5, (0, -0.8)           # AR and MA parameters
    lp = ARMA(ϕ, θ)
    X = lp.simulation(ts_length=n)

    fig, ax = plt.subplots()
    x, y = periodogram(X)
    ax.plot(x, y, 'b-', lw=2, alpha=0.5, label='periodogram')
    x_sd, y_sd = lp.spectral_density(two_pi=False, res=120)
    ax.plot(x_sd, y_sd, 'r-', lw=2, alpha=0.8, label='spectral density')
    ax.legend()
    plt.show()



This estimate looks rather disappointing, but the data size is only 40, so
perhaps it's not surprising that the estimate is poor.

However, if we try again with ``n = 1200`` the outcome is not much better

.. figure:: /_static/lecture_specific/estspec/periodogram1.png

The periodogram is far too irregular relative to the underlying spectral density.

This brings us to our next topic.


:index:`Smoothing`
==================

.. index::
    single: Spectra, Estimation; Smoothing

There are two related issues here.

One is that, given the way the fast Fourier transform is implemented, the
number of points :math:`\omega` at which :math:`I(\omega)` is estimated
increases in line with the amount of data.

In other words, although we have more data, we are also using it to estimate more values.

A second issue is that densities of all types are fundamentally hard to
estimate without parametric assumptions.

.. index::
    single: Nonparametric Estimation

Typically, nonparametric estimation of densities requires some degree of smoothing.


The standard way that smoothing is applied to periodograms is by taking local averages.

In other words, the value :math:`I(\omega_j)` is replaced with a weighted
average of the adjacent values

.. math::

    I(\omega_{j-p}), I(\omega_{j-p+1}), \ldots, I(\omega_j), \ldots, I(\omega_{j+p})


This weighted average can be written as

.. math::
    :label: estspec_ws

    I_S(\omega_j) := \sum_{\ell = -p}^{p} w(\ell) I(\omega_{j+\ell})


where the weights :math:`w(-p), \ldots, w(p)` are a sequence of :math:`2p + 1` nonnegative
values summing to one.

In general, larger values of :math:`p` indicate more smoothing --- more on
this below.

The next figure shows the kind of sequence typically used.

Note the smaller weights towards the edges and larger weights in the center, so that more distant values from :math:`I(\omega_j)` have less weight than closer ones in the sum :eq:`estspec_ws`.



.. code-block:: python3

    def hanning_window(M):
        w = [0.5 - 0.5 * np.cos(2 * np.pi * n/(M-1)) for n in range(M)]
        return w

    window = hanning_window(25) / np.abs(sum(hanning_window(25)))
    x = np.linspace(-12, 12, 25)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(x, window)
    ax.set_title("Hanning window")
    ax.set_ylabel("Weights")
    ax.set_xlabel("Position in sequence of weights")
    plt.show()




Estimation with Smoothing
------------------------------

.. index::
    single: Spectra, Estimation; Smoothing

Our next step is to provide code that will not only estimate the periodogram but also provide smoothing as required.

Such functions have been written in  `estspec.py <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/estspec.py>`__ and are available once you've installed `QuantEcon.py <http://quantecon.org/python_index.html>`__.

The `GitHub listing <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/estspec.py>`__ displays three functions,  ``smooth()``, ``periodogram()``, ``ar_periodogram()``. We will discuss the first two here and the third one :ref:`below <ar_periodograms>`.

The ``periodogram()`` function returns a periodogram, optionally smoothed via the ``smooth()`` function.

Regarding the ``smooth()`` function, since smoothing adds a nontrivial amount of computation, we have applied a fairly terse array-centric method based around ``np.convolve``.

Readers are left either to  explore or simply to use this code according to their interests.

The next three figures each show smoothed and unsmoothed periodograms, as well as the population or "true" spectral density.

(The model is the same as before --- see equation :eq:`esp_arma` --- and there are 400 observations)

From the top figure to bottom, the window length is varied from small to large.

.. _fig_window_smoothing:

.. figure:: /_static/lecture_specific/estspec/window_smoothing.png


In looking at the figure, we can see that for this model and data size, the
window length chosen in the middle figure provides the best fit.

Relative to this value, the first window length provides insufficient
smoothing, while the third gives too much smoothing.

Of course in real estimation problems, the true spectral density is not visible
and the choice of appropriate smoothing will have to be made based on
judgement/priors or some other theory.


.. _estspec_pfas:

Pre-Filtering and Smoothing
------------------------------

.. index::
    single: Spectra, Estimation; Pre-Filtering

.. index::
    single: Spectra, Estimation; Smoothing

In the `code listing <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/estspec.py>`__, we showed three functions from the file ``estspec.py``.

The third function in the file (``ar_periodogram()``) adds a pre-processing step to periodogram smoothing.

First, we describe the basic idea, and after that we give the code.

The essential idea is to

#. Transform the data in order to make estimation of the spectral density more efficient.
#. Compute the periodogram associated with the transformed data.
#. Reverse the effect of the transformation on the periodogram, so that it now
   estimates the spectral density of the original process.

Step 1 is called *pre-filtering* or *pre-whitening*, while step 3 is called *recoloring*.

The first step is called pre-whitening because the
transformation is usually designed to turn the data into something closer to white noise.

Why would this be desirable in terms of spectral density estimation?

The reason is that we are smoothing our estimated periodogram based on
estimated values at nearby points --- recall :eq:`estspec_ws`.

The underlying assumption that makes this a good idea is that the true
spectral density is relatively regular --- the value of :math:`I(\omega)` is close
to that of :math:`I(\omega')` when :math:`\omega` is close to :math:`\omega'`.

This will not be true in all cases, but it is certainly true for white noise.

For white noise, :math:`I` is as regular as possible --- :ref:`it is a constant function <arma_wnsd>`.

In this case, values of :math:`I(\omega')` at points :math:`\omega'` near to :math:`\omega`
provided the maximum possible amount of information about the value :math:`I(\omega)`.

Another way to put this is that if :math:`I` is relatively constant, then we can use a large amount of smoothing without introducing too much bias.

.. _ar_periodograms:

The AR(1) Setting
-------------------

.. index::
    single: Spectra, Estimation; AR(1) Setting

Let's examine this idea more carefully in a particular setting --- where
the data are assumed to be generated by an  AR(1) process.


(More general ARMA settings can be handled using similar techniques to those described below)

Suppose in particular that :math:`\{X_t\}` is covariance stationary and AR(1),
with

.. math::
    :label: estspec_ar_dgp

    X_{t+1} = \mu + \phi X_t + \epsilon_{t+1}


where :math:`\mu` and :math:`\phi \in (-1, 1)` are unknown parameters and :math:`\{ \epsilon_t \}` is white noise.

It follows that if we regress :math:`X_{t+1}` on :math:`X_t` and an intercept, the residuals
will approximate white noise.

Let

* :math:`g` be the spectral density of :math:`\{ \epsilon_t \}` --- a constant function, as discussed above

* :math:`I_0` be the periodogram estimated from the residuals --- an estimate of :math:`g`

* :math:`f` be the spectral density of :math:`\{ X_t \}` --- the object we are trying to estimate

In view of :ref:`an earlier result <arma_spec_den>` we obtained while discussing ARMA processes, :math:`f` and :math:`g` are related by

.. math::
    :label: ar_sdsc

    f(\omega) = \left| \frac{1}{1 - \phi e^{i\omega}} \right|^2 g(\omega)


This suggests that the recoloring step, which constructs an estimate :math:`I` of :math:`f` from :math:`I_0`, should set

.. math::

    I(\omega) = \left| \frac{1}{1 - \hat \phi e^{i\omega}} \right|^2 I_0(\omega)


where :math:`\hat \phi` is the OLS estimate of :math:`\phi`.

The code for ``ar_periodogram()`` --- the third function in ``estspec.py`` --- does exactly this. (See the code `here <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/estspec.py>`__).

The next figure shows realizations of the two kinds of smoothed periodograms

#. "standard smoothed periodogram", the ordinary smoothed periodogram, and
#. "AR smoothed periodogram", the pre-whitened and recolored one generated by ``ar_periodogram()``

The periodograms are calculated from time series drawn from :eq:`estspec_ar_dgp` with :math:`\mu = 0` and :math:`\phi = -0.9`.

Each time series is of length 150.

The difference between the three subfigures is just randomness --- each one uses a different draw of the time series.

.. _fig_ar_smoothed_periodogram:

.. figure:: /_static/lecture_specific/estspec/ar_smoothed_periodogram.png

In all cases, periodograms are fit with the "hamming" window and window length of 65.

Overall, the fit of the AR smoothed periodogram is much better, in the sense
of being closer to the true spectral density.




Exercises
=============


.. _estspec_ex1:

Exercise 1
------------

Replicate :ref:`this figure <fig_window_smoothing>` (modulo randomness).

The model is as in equation :eq:`esp_arma` and there are 400 observations.

For the smoothed periodogram, the window type is "hamming".



.. _estspec_ex2:

Exercise 2
------------

Replicate :ref:`this figure <fig_ar_smoothed_periodogram>` (modulo randomness).

The model is as in equation :eq:`estspec_ar_dgp`, with :math:`\mu = 0`, :math:`\phi = -0.9`
and 150 observations in each time series.

All periodograms are fit with the "hamming" window and window length of 65.



Solutions
============

Exercise 1
----------

.. code-block:: python3


    ## Data
    n = 400
    ϕ = 0.5
    θ = 0, -0.8
    lp = ARMA(ϕ, θ)
    X = lp.simulation(ts_length=n)

    fig, ax = plt.subplots(3, 1, figsize=(10, 12))

    for i, wl in enumerate((15, 55, 175)):  # Window lengths

        x, y = periodogram(X)
        ax[i].plot(x, y, 'b-', lw=2, alpha=0.5, label='periodogram')

        x_sd, y_sd = lp.spectral_density(two_pi=False, res=120)
        ax[i].plot(x_sd, y_sd, 'r-', lw=2, alpha=0.8, label='spectral density')

        x, y_smoothed = periodogram(X, window='hamming', window_len=wl)
        ax[i].plot(x, y_smoothed, 'k-', lw=2, label='smoothed periodogram')

        ax[i].legend()
        ax[i].set_title(f'window length = {wl}')
    plt.show()

Exercise 2
----------

.. code-block:: python3

    lp = ARMA(-0.9)
    wl = 65


    fig, ax = plt.subplots(3, 1, figsize=(10,12))

    for i in range(3):
        X = lp.simulation(ts_length=150)
        ax[i].set_xlim(0, np.pi)

        x_sd, y_sd = lp.spectral_density(two_pi=False, res=180)
        ax[i].semilogy(x_sd, y_sd, 'r-', lw=2, alpha=0.75,
            label='spectral density')

        x, y_smoothed = periodogram(X, window='hamming', window_len=wl)
        ax[i].semilogy(x, y_smoothed, 'k-', lw=2, alpha=0.75,
            label='standard smoothed periodogram')

        x, y_ar = ar_periodogram(X, window='hamming', window_len=wl)
        ax[i].semilogy(x, y_ar, 'b-', lw=2, alpha=0.75,
            label='AR smoothed periodogram')

        ax[i].legend(loc='upper left')
    plt.show()

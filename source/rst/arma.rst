.. _arma:

.. include:: /_static/includes/header.raw

.. highlight:: python3

******************************************
:index:`Covariance Stationary Processes`
******************************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
============

In this lecture we study covariance stationary linear stochastic processes, a
class of models routinely used to study economic and financial time series.

This class has the advantage of being

#. simple enough to be described by an elegant and comprehensive theory
#. relatively broad in terms of the kinds of dynamics it can represent

We consider these models in both the time and frequency domain.

:index:`ARMA Processes`
------------------------

We will focus much of our attention on linear covariance stationary models with a finite number of parameters.

In particular, we will study stationary ARMA processes, which form a cornerstone of the standard theory of time series analysis.

Every ARMA process can be represented in :doc:`linear state space <linear_models>` form.

However, ARMA processes have some important structure that makes it valuable to study them separately.



:index:`Spectral Analysis`
---------------------------

Analysis in the frequency domain is also called spectral analysis.

In essence, spectral analysis provides an alternative representation of the
autocovariance function of a covariance stationary process.

Having a second representation of this important object

* shines a light on the dynamics of the process in question

* allows for a simpler, more tractable representation in some important cases

The famous *Fourier transform* and its inverse are used to map between the two representations.



Other Reading
---------------

For supplementary reading, see

.. only:: html

    * :cite:`Ljungqvist2012`, chapter 2
    * :cite:`Sargent1987`, chapter 11
    * John Cochrane's :download:`notes on time series analysis </_static/lecture_specific/arma/time_series_book.pdf>`, chapter 8
    * :cite:`Shiryaev1995`, chapter 6
    * :cite:`CryerChan2008`, all

.. only:: latex

    * :cite:`Ljungqvist2012`, chapter 2
    * :cite:`Sargent1987`, chapter 11
    * John Cochrane's `notes on time series analysis <https://lectures.quantecon.org/_downloads/time_series_book.pdf>`__, chapter 8
    * :cite:`Shiryaev1995`, chapter 6
    * :cite:`CryerChan2008`, all


Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    import quantecon as qe


Introduction
=================================

Consider a sequence of random variables :math:`\{ X_t \}` indexed by :math:`t \in \mathbb Z` and taking values in :math:`\mathbb R`.

Thus, :math:`\{ X_t \}` begins in the infinite past and extends to the infinite future --- a convenient and standard assumption.

As in other fields, successful economic modeling typically assumes the existence of features that are constant over time.

If these assumptions are correct, then each new observation :math:`X_t, X_{t+1},\ldots` can provide additional information about the time-invariant features, allowing us to  learn from as data arrive.

For this reason, we will focus in what follows on processes that are *stationary* --- or become so after a transformation
(see for example :doc:`this lecture <additive_functionals>`).




.. _arma_defs:

Definitions
--------------

.. index::
    single: Covariance Stationary

A real-valued stochastic process :math:`\{ X_t \}` is called *covariance stationary* if

#. Its mean :math:`\mu := \mathbb E X_t` does not depend on :math:`t`.
#. For all :math:`k` in :math:`\mathbb Z`, the :math:`k`-th autocovariance :math:`\gamma(k) := \mathbb E (X_t - \mu)(X_{t + k} - \mu)` is finite and depends only on :math:`k`.


The function :math:`\gamma \colon \mathbb Z \to \mathbb R` is called the *autocovariance function* of the process.

Throughout this lecture, we will work exclusively with zero-mean (i.e., :math:`\mu = 0`) covariance stationary processes.

The zero-mean assumption costs nothing in terms of generality since working with non-zero-mean processes involves no more than adding a constant.


Example 1: :index:`White Noise`
--------------------------------

Perhaps the simplest class of covariance stationary processes is the white noise processes.


A process :math:`\{ \epsilon_t \}` is called a *white noise process* if

#. :math:`\mathbb E \epsilon_t = 0`
#. :math:`\gamma(k) = \sigma^2 \mathbf 1\{k = 0\}` for some :math:`\sigma > 0`

(Here :math:`\mathbf 1\{k = 0\}` is defined to be 1 if :math:`k = 0` and zero otherwise)

White noise processes play the role of **building blocks** for processes with more complicated dynamics.


.. _generalized_lps:

Example 2: :index:`General Linear Processes`
--------------------------------------------

From the simple building block provided by white noise, we can construct a very flexible family of covariance stationary processes --- the *general linear processes*

.. math::
    :label: ma_inf

    X_t = \sum_{j=0}^{\infty} \psi_j \epsilon_{t-j},
    \qquad t \in \mathbb Z


where

* :math:`\{\epsilon_t\}` is white noise
* :math:`\{\psi_t\}` is a square summable sequence in :math:`\mathbb R` (that is, :math:`\sum_{t=0}^{\infty} \psi_t^2 < \infty`)

The sequence :math:`\{\psi_t\}` is often called a *linear filter*.

Equation :eq:`ma_inf` is said to present  a **moving average** process or a moving average representation.

With some manipulations, it is possible to confirm that the autocovariance function for :eq:`ma_inf` is

.. math::
    :label: ma_inf_ac

    \gamma(k) = \sigma^2 \sum_{j=0}^{\infty} \psi_j \psi_{j+k}


By the `Cauchy-Schwartz inequality <https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality>`_, one can show that :math:`\gamma(k)` satisfies equation :eq:`ma_inf_ac`.

Evidently, :math:`\gamma(k)` does not depend on :math:`t`.




:index:`Wold Representation`
------------------------------

Remarkably, the class of general linear processes goes a long way towards
describing the entire class of zero-mean covariance stationary processes.

In particular, `Wold's decomposition theorem <https://en.wikipedia.org/wiki/Wold%27s_theorem>`_ states that every
zero-mean covariance stationary process :math:`\{X_t\}` can be written as

.. math::

    X_t = \sum_{j=0}^{\infty} \psi_j \epsilon_{t-j} + \eta_t


where

* :math:`\{\epsilon_t\}` is white noise
* :math:`\{\psi_t\}` is square summable
* :math:`\psi_0 \epsilon_t` is the one-step ahead prediction error in forecasting :math:`X_t` as a linear least-squares function of the infinite history :math:`X_{t-1}, X_{t-2}, \ldots`
* :math:`\eta_t` can be expressed as a linear function of :math:`X_{t-1}, X_{t-2},\ldots` and is perfectly predictable over arbitrarily long horizons

For the method of constructing a Wold representation, intuition, and further discussion, see :cite:`Sargent1987`, p. 286.


AR and MA
--------------------

.. index::
    single: Covariance Stationary Processes; AR

.. index::
    single: Covariance Stationary Processes; MA

General linear processes are a very broad class of processes.

It often pays to specialize to those for which there exists a representation having only finitely many parameters.

(Experience and theory combine to indicate that models with a relatively small number of parameters typically perform better than larger models, especially for forecasting)

One very simple example of such a model is the first-order autoregressive or AR(1) process

.. math::
    :label: ar1_rep

    X_t = \phi X_{t-1} + \epsilon_t
    \quad \text{where} \quad
    | \phi | < 1
    \quad \text{and } \{ \epsilon_t \} \text{ is white noise}


By direct substitution, it is easy to verify that :math:`X_t = \sum_{j=0}^{\infty} \phi^j \epsilon_{t-j}`.

Hence :math:`\{X_t\}` is a general linear process.

Applying :eq:`ma_inf_ac` to the previous expression for :math:`X_t`, we get the AR(1) autocovariance function

.. math::
    :label: ar_acov

    \gamma(k) = \phi^k \frac{\sigma^2}{1 - \phi^2},
    \qquad k = 0, 1, \ldots


The next figure plots an example of this function for :math:`\phi = 0.8` and :math:`\phi = -0.8` with :math:`\sigma = 1`.



.. code-block:: python3

    num_rows, num_cols = 2, 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    for i, ϕ in enumerate((0.8, -0.8)):
        ax = axes[i]
        times = list(range(16))
        acov = [ϕ**k / (1 - ϕ**2) for k in times]
        ax.plot(times, acov, 'bo-', alpha=0.6,
                label=f'autocovariance, $\phi = {ϕ:.2}$')
        ax.legend(loc='upper right')
        ax.set(xlabel='time', xlim=(0, 15))
        ax.hlines(0, 0, 15, linestyle='--', alpha=0.5)
    plt.show()



Another very simple process is the MA(1) process (here MA means "moving average")

.. math::

    X_t = \epsilon_t + \theta \epsilon_{t-1}


You will be able to verify that

.. math::

    \gamma(0) = \sigma^2 (1 + \theta^2),
    \quad
    \gamma(1) = \sigma^2 \theta,
    \quad \text{and} \quad
    \gamma(k) = 0 \quad \forall \, k > 1


The AR(1) can be generalized to an AR(:math:`p`) and likewise for the MA(1).

Putting all of this together, we get the

:index:`ARMA` Processes
------------------------

A stochastic process :math:`\{X_t\}` is called an *autoregressive moving
average process*, or ARMA(:math:`p,q`), if it can be written as

.. math::
    :label: arma

    X_t = \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} +
        \epsilon_t + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q}


where :math:`\{ \epsilon_t \}` is white noise.

An alternative notation for ARMA processes uses the *lag operator* :math:`L`.

**Def.** Given arbitrary variable :math:`Y_t`, let :math:`L^k Y_t := Y_{t-k}`.

It turns out that

* lag operators facilitate  succinct representations for linear stochastic processes
* algebraic manipulations that treat the lag operator as an ordinary scalar  are legitimate

Using :math:`L`, we can rewrite :eq:`arma` as

.. math::
    :label: arma_lag

    L^0 X_t - \phi_1 L^1 X_t - \cdots - \phi_p L^p X_t
    = L^0 \epsilon_t + \theta_1 L^1 \epsilon_t + \cdots + \theta_q L^q \epsilon_t


If we let :math:`\phi(z)` and :math:`\theta(z)` be the polynomials

.. math::
    :label: arma_poly

    \phi(z) := 1 - \phi_1 z - \cdots - \phi_p z^p
    \quad \text{and} \quad
    \theta(z) := 1 + \theta_1 z + \cdots + \theta_q z^q


then :eq:`arma_lag`  becomes

.. math::
    :label: arma_lag1

    \phi(L) X_t = \theta(L) \epsilon_t


In what follows we **always assume** that the roots of the polynomial :math:`\phi(z)` lie outside the unit circle in the complex plane.

This condition is sufficient to guarantee that the ARMA(:math:`p,q`) process is covariance stationary.

In fact, it implies that the process falls within the class of general linear processes :ref:`described above <generalized_lps>`.

That is, given an ARMA(:math:`p,q`) process :math:`\{ X_t \}` satisfying the unit circle condition, there exists a square summable sequence :math:`\{\psi_t\}` with :math:`X_t = \sum_{j=0}^{\infty} \psi_j \epsilon_{t-j}` for all :math:`t`.

The sequence :math:`\{\psi_t\}` can be obtained by a recursive procedure outlined on page 79 of :cite:`CryerChan2008`.

The function :math:`t \mapsto \psi_t` is often called the *impulse response function*.



:index:`Spectral Analysis`
=================================

Autocovariance functions provide a great deal of information about covariance stationary processes.

In fact, for zero-mean Gaussian processes, the autocovariance function characterizes the entire joint distribution.

Even for non-Gaussian processes, it provides a significant amount of information.

It turns out that there is an alternative representation of the autocovariance function of a covariance stationary process, called the *spectral density*.

At times, the spectral density is easier to derive, easier to manipulate, and provides additional intuition.

:index:`Complex Numbers`
-------------------------

Before discussing the spectral density, we invite you to recall the main properties of complex numbers (or :ref:`skip to the next section <arma_specd>`).

It can be helpful to remember that, in a formal sense, complex numbers are just points :math:`(x, y) \in \mathbb R^2` endowed with a specific notion of multiplication.

When :math:`(x, y)` is regarded as a complex number, :math:`x` is called the *real part* and :math:`y` is called the *imaginary part*.

The *modulus* or *absolute value* of a complex number :math:`z = (x, y)` is just its Euclidean norm in :math:`\mathbb R^2`, but is usually written as :math:`|z|` instead of :math:`\|z\|`.

The product of two complex numbers :math:`(x, y)` and :math:`(u, v)` is defined to be :math:`(xu - vy, xv + yu)`, while addition is standard pointwise vector addition.

When endowed with these notions of multiplication and addition, the set of complex numbers forms a `field <https://en.wikipedia.org/wiki/Field_%28mathematics%29>`_ --- addition and multiplication play well together, just as they do in :math:`\mathbb R`.

The complex number :math:`(x, y)` is often written as :math:`x + i y`, where :math:`i` is called the *imaginary unit* and is understood to obey :math:`i^2 = -1`.

The :math:`x + i y` notation provides an easy way to remember the definition of multiplication given above, because, proceeding naively,

.. math::

    (x + i y) (u + i v) = xu - yv + i (xv + yu)


Converted back to our first notation, this becomes :math:`(xu - vy, xv + yu)` as promised.

Complex numbers can be represented in  the polar form :math:`r e^{i \omega}` where

.. math::

    r e^{i \omega} := r (\cos(\omega) + i \sin(\omega)) = x + i y

where :math:`x = r \cos(\omega), y = r \sin(\omega)`, and :math:`\omega = \arctan(y/z)` or :math:`\tan(\omega) = y/x`.


.. _arma_specd:

:index:`Spectral Densities`
---------------------------------

Let :math:`\{ X_t \}` be a covariance stationary process with autocovariance function :math:`\gamma`  satisfying :math:`\sum_{k} \gamma(k)^2 < \infty`.

The *spectral density* :math:`f` of :math:`\{ X_t \}` is defined as the `discrete time Fourier transform <https://en.wikipedia.org/wiki/Discrete-time_Fourier_transform>`_ of its autocovariance function :math:`\gamma`.

.. math::

    f(\omega) := \sum_{k \in \mathbb Z} \gamma(k) e^{-i \omega k},
    \qquad \omega \in \mathbb R


(Some authors normalize the expression on the right by constants such as :math:`1/\pi` --- the convention chosen  makes little difference provided you are consistent).

Using the fact that :math:`\gamma` is *even*, in the sense that :math:`\gamma(t) = \gamma(-t)` for all :math:`t`, we can show that

.. math::
    :label: arma_sd_cos

    f(\omega) = \gamma(0) + 2 \sum_{k \geq 1} \gamma(k) \cos(\omega k)


It is not difficult to confirm that :math:`f` is

* real-valued
* even (:math:`f(\omega) = f(-\omega)`   ),  and
* :math:`2\pi`-periodic, in the sense that :math:`f(2\pi + \omega) = f(\omega)` for all :math:`\omega`

It follows that the values of :math:`f` on :math:`[0, \pi]` determine the values of :math:`f` on
all of :math:`\mathbb R` --- the proof is an exercise.

For this reason, it is standard to plot the spectral density only on the interval :math:`[0, \pi]`.

.. _arma_wnsd:

Example 1: :index:`White Noise`
---------------------------------

Consider a white noise process :math:`\{\epsilon_t\}` with standard deviation :math:`\sigma`.

It is easy to check that in  this case :math:`f(\omega) = \sigma^2`.  So :math:`f` is a constant function.

As we will see, this can be interpreted as meaning that "all frequencies are equally present".

(White light has this property when frequency refers to the visible spectrum, a connection that provides the origins of the term "white noise")


Example 2: :index:`AR` and :index:`MA` and :index:`ARMA`
---------------------------------------------------------

It is an exercise to show that the MA(1) process :math:`X_t = \theta \epsilon_{t-1} + \epsilon_t` has a spectral density

.. math::
    :label: ma1_sd_ed

    f(\omega)
    = \sigma^2 ( 1 + 2 \theta \cos(\omega) + \theta^2 )


With a bit more effort, it's possible to show (see, e.g., p. 261 of :cite:`Sargent1987`) that the spectral density of the AR(1) process :math:`X_t = \phi X_{t-1} + \epsilon_t` is

.. math::
    :label: ar1_sd_ed

    f(\omega)
    = \frac{\sigma^2}{ 1 - 2 \phi \cos(\omega) + \phi^2 }


More generally, it can be shown that the spectral density of the ARMA process :eq:`arma` is

.. _arma_spec_den:

.. math::
    :label: arma_sd

    f(\omega) = \left| \frac{\theta(e^{i\omega})}{\phi(e^{i\omega})} \right|^2 \sigma^2


where

* :math:`\sigma` is the standard deviation of the white noise process :math:`\{\epsilon_t\}`.
* the polynomials :math:`\phi(\cdot)` and :math:`\theta(\cdot)` are as defined in :eq:`arma_poly`.

The derivation of :eq:`arma_sd` uses the fact that convolutions become products under Fourier transformations.

The proof is elegant and can be found in many places --- see, for example, :cite:`Sargent1987`, chapter 11, section 4.

It's a nice exercise to verify that :eq:`ma1_sd_ed` and :eq:`ar1_sd_ed` are indeed special cases of :eq:`arma_sd`.


Interpreting the :index:`Spectral Density`
--------------------------------------------

.. index::
    single: Spectral Density; interpretation

Plotting :eq:`ar1_sd_ed` reveals the shape of the spectral density for the AR(1) model when :math:`\phi` takes the values 0.8 and -0.8 respectively.



.. code-block:: python3

    def ar1_sd(ϕ, ω):
        return 1 / (1 - 2 * ϕ * np.cos(ω) + ϕ**2)

    ωs = np.linspace(0, np.pi, 180)
    num_rows, num_cols = 2, 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    # Autocovariance when phi = 0.8
    for i, ϕ in enumerate((0.8, -0.8)):
        ax = axes[i]
        sd = ar1_sd(ϕ, ωs)
        ax.plot(ωs, sd, 'b-', alpha=0.6, lw=2,
                label='spectral density, $\phi = {ϕ:.2}$')
        ax.legend(loc='upper center')
        ax.set(xlabel='frequency', xlim=(0, np.pi))
    plt.show()




These spectral densities correspond to the autocovariance functions for the
AR(1) process shown above.

Informally, we think of the spectral density as being large at those :math:`\omega \in [0, \pi]` at which
the autocovariance function seems approximately to exhibit big damped cycles.

To see the idea, let's consider why, in the lower panel of the preceding figure, the spectral density for the case :math:`\phi = -0.8` is large at :math:`\omega = \pi`.

Recall that the spectral density can be expressed as

.. math::
    :label: sumpr

    f(\omega)
    = \gamma(0) + 2 \sum_{k \geq 1} \gamma(k) \cos(\omega k)
    = \gamma(0) + 2 \sum_{k \geq 1} (-0.8)^k \cos(\omega k)


When we evaluate this at :math:`\omega = \pi`, we get a large number because
:math:`\cos(\pi k)` is large and positive when :math:`(-0.8)^k` is
positive, and large in absolute value and negative when :math:`(-0.8)^k` is negative.

Hence the product is always large and positive, and hence the sum of the
products on the right-hand side of :eq:`sumpr` is large.

These ideas are illustrated in the next figure, which has :math:`k` on the horizontal axis.



.. code-block:: python3

    ϕ = -0.8
    times = list(range(16))
    y1 = [ϕ**k / (1 - ϕ**2) for k in times]
    y2 = [np.cos(np.pi * k) for k in times]
    y3 = [a * b for a, b in zip(y1, y2)]

    num_rows, num_cols = 3, 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.25)

    # Autocovariance when ϕ = -0.8
    ax = axes[0]
    ax.plot(times, y1, 'bo-', alpha=0.6, label='$\gamma(k)$')
    ax.legend(loc='upper right')
    ax.set(xlim=(0, 15), yticks=(-2, 0, 2))
    ax.hlines(0, 0, 15, linestyle='--', alpha=0.5)

    # Cycles at frequency π
    ax = axes[1]
    ax.plot(times, y2, 'bo-', alpha=0.6, label='$\cos(\pi k)$')
    ax.legend(loc='upper right')
    ax.set(xlim=(0, 15), yticks=(-1, 0, 1))
    ax.hlines(0, 0, 15, linestyle='--', alpha=0.5)

    # Product
    ax = axes[2]
    ax.stem(times, y3, label='$\gamma(k) \cos(\pi k)$')
    ax.legend(loc='upper right')
    ax.set(xlim=(0, 15), ylim=(-3, 3), yticks=(-1, 0, 1, 2, 3))
    ax.hlines(0, 0, 15, linestyle='--', alpha=0.5)
    ax.set_xlabel("k")

    plt.show()



On the other hand, if we evaluate :math:`f(\omega)` at :math:`\omega = \pi / 3`, then the cycles are
not matched, the sequence :math:`\gamma(k) \cos(\omega k)` contains
both positive and negative terms, and hence the sum of these terms is much smaller.



.. code-block:: python3

    ϕ = -0.8
    times = list(range(16))
    y1 = [ϕ**k / (1 - ϕ**2) for k in times]
    y2 = [np.cos(np.pi * k/3) for k in times]
    y3 = [a * b for a, b in zip(y1, y2)]

    num_rows, num_cols = 3, 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.25)

    # Autocovariance when phi = -0.8
    ax = axes[0]
    ax.plot(times, y1, 'bo-', alpha=0.6, label='$\gamma(k)$')
    ax.legend(loc='upper right')
    ax.set(xlim=(0, 15), yticks=(-2, 0, 2))
    ax.hlines(0, 0, 15, linestyle='--', alpha=0.5)

    # Cycles at frequency π
    ax = axes[1]
    ax.plot(times, y2, 'bo-', alpha=0.6, label='$\cos(\pi k/3)$')
    ax.legend(loc='upper right')
    ax.set(xlim=(0, 15), yticks=(-1, 0, 1))
    ax.hlines(0, 0, 15, linestyle='--', alpha=0.5)

    # Product
    ax = axes[2]
    ax.stem(times, y3, label='$\gamma(k) \cos(\pi k/3)$')
    ax.legend(loc='upper right')
    ax.set(xlim=(0, 15), ylim=(-3, 3), yticks=(-1, 0, 1, 2, 3))
    ax.hlines(0, 0, 15, linestyle='--', alpha=0.5)
    ax.set_xlabel("$k$")

    plt.show()



In summary, the spectral density is large at frequencies :math:`\omega` where the autocovariance function exhibits damped cycles.


Inverting the Transformation
-------------------------------

.. index::
    single: Spectral Density; Inverting the Transformation

We have just seen that the spectral density is useful in the sense that it provides a frequency-based perspective on the autocovariance structure of a covariance stationary process.

Another reason that the spectral density is useful is that it can be "inverted" to recover the autocovariance function via the *inverse Fourier transform*.

In particular, for all :math:`k \in \mathbb Z`, we have

.. math::
    :label: ift

    \gamma(k) = \frac{1}{2 \pi} \int_{-\pi}^{\pi} f(\omega) e^{i \omega k} d\omega


This is convenient in situations where the spectral density is easier to calculate and manipulate than the autocovariance function.

(For example, the expression :eq:`arma_sd` for the ARMA spectral density is much easier to work with than the expression for the ARMA autocovariance)


Mathematical Theory
---------------------

.. index::
    single: Spectral Density; Mathematical Theory

This section is loosely based on :cite:`Sargent1987`, p. 249-253, and included for those who

* would like a bit more insight into spectral densities
* and have at least some background in `Hilbert space <https://en.wikipedia.org/wiki/Hilbert_space>`_ theory

Others should feel free to skip to the :ref:`next section <arma_imp>` --- none of this material is necessary to progress to computation.

Recall that every `separable <https://en.wikipedia.org/wiki/Separable_space>`_ Hilbert space :math:`H` has a countable orthonormal basis :math:`\{ h_k \}`.

The nice thing about such a basis is that every :math:`f \in H` satisfies

.. math::
    :label: arma_fc

    f = \sum_k \alpha_k h_k
    \quad \text{where} \quad
    \alpha_k := \langle f, h_k \rangle


where :math:`\langle \cdot, \cdot \rangle` denotes the inner product in :math:`H`.

Thus, :math:`f` can be represented to any degree of precision by linearly combining basis vectors.

The scalar sequence :math:`\alpha = \{\alpha_k\}` is called the *Fourier coefficients* of :math:`f`, and satisfies :math:`\sum_k |\alpha_k|^2 < \infty`.

In other words, :math:`\alpha` is in :math:`\ell_2`, the set of square summable sequences.

Consider an operator :math:`T` that maps :math:`\alpha \in \ell_2` into its expansion :math:`\sum_k \alpha_k h_k \in H`.

The Fourier coefficients of :math:`T\alpha` are just :math:`\alpha = \{ \alpha_k \}`, as you can verify by confirming that :math:`\langle T \alpha, h_k \rangle = \alpha_k`.

Using elementary results from Hilbert space theory, it can be shown that

* :math:`T` is one-to-one --- if :math:`\alpha` and :math:`\beta` are distinct in :math:`\ell_2`, then so are their expansions in :math:`H`.
* :math:`T` is onto --- if :math:`f \in H` then its preimage in :math:`\ell_2` is the sequence :math:`\alpha` given by :math:`\alpha_k = \langle f, h_k \rangle`.
* :math:`T` is a linear isometry --- in particular, :math:`\langle \alpha, \beta \rangle = \langle T\alpha, T\beta \rangle`.

Summarizing these results, we say that any separable Hilbert space is isometrically isomorphic to :math:`\ell_2`.

In essence, this says that each separable Hilbert space we consider is just a different way of looking at the fundamental space :math:`\ell_2`.

With this in mind, let's specialize to a setting where

* :math:`\gamma \in \ell_2` is the autocovariance function of a covariance stationary process, and :math:`f` is the spectral density.
* :math:`H = L_2`, where :math:`L_2` is the set of square summable functions on the interval :math:`[-\pi, \pi]`, with inner product :math:`\langle g, h \rangle = \int_{-\pi}^{\pi} g(\omega) h(\omega) d \omega`.
* :math:`\{h_k\} =` the orthonormal basis for :math:`L_2` given by the set of trigonometric functions.

.. math::

    h_k(\omega) = \frac{e^{i \omega k}}{\sqrt{2 \pi}},
    \quad k \in \mathbb Z,
    \quad \omega \in [-\pi, \pi]


Using the definition of :math:`T` from above and the fact that :math:`f` is even, we now have

.. math::
    :label: arma_it

    T \gamma
    = \sum_{k \in \mathbb Z}
    \gamma(k) \frac{e^{i \omega k}}{\sqrt{2 \pi}} = \frac{1}{\sqrt{2 \pi}} f(\omega)


In other words, apart from a scalar multiple, the spectral density is just a transformation of :math:`\gamma \in \ell_2` under a certain linear isometry --- a different way to view :math:`\gamma`.

In particular, it is an expansion of the autocovariance function with respect to the trigonometric basis functions in :math:`L_2`.

As discussed above, the Fourier coefficients of :math:`T \gamma` are given by the sequence :math:`\gamma`, and,
in particular, :math:`\gamma(k) = \langle T \gamma, h_k \rangle`.

Transforming this inner product into its integral expression and using :eq:`arma_it` gives
:eq:`ift`, justifying our earlier expression for the inverse transform.



.. _arma_imp:

Implementation
=================================

Most code for working with covariance stationary models deals with ARMA models.

Python code for studying ARMA models can be found in the ``tsa`` submodule of `statsmodels <http://statsmodels.sourceforge.net/>`_.

Since this code doesn't quite cover our needs --- particularly vis-a-vis spectral analysis --- we've put together the module `arma.py <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/arma.py>`_, which is part of `QuantEcon.py <http://quantecon.org/quantecon-py>`_ package.

The module provides functions for mapping ARMA(:math:`p,q`) models into their

#. impulse response function

#. simulated time series

#. autocovariance function

#. spectral density


Application
-----------------------------------------

Let's use this code to replicate the plots on pages 68--69 of :cite:`Ljungqvist2012`.



Here are some functions to generate the plots




.. code-block:: python3

    def plot_impulse_response(arma, ax=None):
        if ax is None:
            ax = plt.gca()
        yi = arma.impulse_response()
        ax.stem(list(range(len(yi))), yi)
        ax.set(xlim=(-0.5), ylim=(min(yi)-0.1, max(yi)+0.1),
                     title='Impulse response', xlabel='time', ylabel='response')
        return ax

    def plot_spectral_density(arma, ax=None):
        if ax is None:
            ax = plt.gca()
        w, spect = arma.spectral_density(two_pi=False)
        ax.semilogy(w, spect)
        ax.set(xlim=(0, np.pi), ylim=(0, np.max(spect)),
               title='Spectral density', xlabel='frequency', ylabel='spectrum')
        return ax

    def plot_autocovariance(arma, ax=None):
        if ax is None:
            ax = plt.gca()
        acov = arma.autocovariance()
        ax.stem(list(range(len(acov))), acov)
        ax.set(xlim=(-0.5, len(acov) - 0.5), title='Autocovariance',
               xlabel='time', ylabel='autocovariance')
        return ax

    def plot_simulation(arma, ax=None):
        if ax is None:
            ax = plt.gca()
        x_out = arma.simulation()
        ax.plot(x_out)
        ax.set(title='Sample path', xlabel='time', ylabel='state space')
        return ax

    def quad_plot(arma):
        """
        Plots the impulse response, spectral_density, autocovariance,
        and one realization of the process.

        """
        num_rows, num_cols = 2, 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        plot_functions = [plot_impulse_response,
                          plot_spectral_density,
                          plot_autocovariance,
                          plot_simulation]
        for plot_func, ax in zip(plot_functions, axes.flatten()):
            plot_func(arma, ax)
        plt.tight_layout()
        plt.show()




Now let's call these functions to generate plots.

As a warmup, let's make sure things look right when we for the pure white noise model :math:`X_t = \epsilon_t`.



.. code-block:: python3

    ϕ = 0.0
    θ = 0.0
    arma = qe.ARMA(ϕ, θ)
    quad_plot(arma)


If we look carefully, things look good: the spectrum is the flat line at :math:`10^0` at the very top of the spectrum graphs,
which is at it should be.

Also

   * the variance  equals :math:`1 = \frac{1}{2 \pi} \int_{-\pi}^\pi 1 d \omega` as it should.

   * the covariogram and impulse response look as they should.

   * it is actually challenging to visualize a time series realization of white noise -- a sequence of surprises -- but this too looks pretty good.


To get some more examples, as our laboratory
we'll replicate quartets of graphs that :cite:`Ljungqvist2012` use to teach "how to read spectral densities".


Ljunqvist and Sargent's first  model is  :math:`X_t = 1.3 X_{t-1} - .7 X_{t-2} + \epsilon_t`




.. code-block:: python3

    ϕ = 1.3, -.7
    θ = 0.0
    arma = qe.ARMA(ϕ, θ)
    quad_plot(arma)




Ljungqvist and Sargent's second model is :math:`X_t = .9 X_{t-1} + \epsilon_t`

.. code-block:: python3

    ϕ = 0.9
    θ = -0.0
    arma = qe.ARMA(ϕ, θ)
    quad_plot(arma)


Ljungqvist and Sargent's third  model is  :math:`X_t = .8 X_{t-4} + \epsilon_t`

.. code-block:: python3

    ϕ = 0., 0., 0., .8
    θ = -0.0
    arma = qe.ARMA(ϕ, θ)
    quad_plot(arma)



Ljungqvist and Sargent's fourth  model is  :math:`X_t = .98 X_{t-1}  + \epsilon_t -.7 \epsilon_{t-1}`


.. code-block:: python3

    ϕ = .98
    θ = -0.7
    arma = qe.ARMA(ϕ, θ)
    quad_plot(arma)

Explanation
--------------------

The call



    ``arma = ARMA(ϕ, θ, σ)``



creates an instance ``arma`` that represents the ARMA(:math:`p, q`) model

.. math::

    X_t = \phi_1 X_{t-1} + ... + \phi_p X_{t-p} +
        \epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}


If ``ϕ`` and ``θ`` are arrays or sequences, then the interpretation will
be

* ``ϕ`` holds the vector of parameters :math:`(\phi_1, \phi_2,..., \phi_p)`.

* ``θ`` holds the vector of parameters :math:`(\theta_1, \theta_2,..., \theta_q)`.

The parameter ``σ`` is always a scalar, the standard deviation of the white noise.

We also permit ``ϕ`` and ``θ`` to be scalars, in which case the model will be interpreted as

.. math::

    X_t = \phi X_{t-1} + \epsilon_t + \theta \epsilon_{t-1}


The two numerical packages most useful for working with ARMA models are ``scipy.signal`` and ``numpy.fft``.



The package ``scipy.signal`` expects the parameters to be passed into its functions in a manner consistent with the alternative ARMA notation :eq:`arma_lag1`.

For example, the impulse response sequence :math:`\{\psi_t\}` discussed above can be obtained using ``scipy.signal.dimpulse``, and the function call should be of the form

    ``times, ψ = dimpulse((ma_poly, ar_poly, 1), n=impulse_length)``

where ``ma_poly`` and ``ar_poly`` correspond to the polynomials in :eq:`arma_poly` --- that is,

* ``ma_poly`` is the vector :math:`(1, \theta_1, \theta_2, \ldots, \theta_q)`
* ``ar_poly`` is the vector :math:`(1, -\phi_1, -\phi_2, \ldots, - \phi_p)`


To this end, we also maintain the arrays ``ma_poly`` and ``ar_poly`` as instance data, with their values computed automatically from the values of ``phi`` and ``theta`` supplied by the user.

If the user decides to change the value of either ``theta`` or ``phi`` ex-post by assignments
such as ``arma.phi = (0.5, 0.2)`` or ``arma.theta = (0, -0.1)``.

then ``ma_poly`` and ``ar_poly`` should update automatically to reflect these new parameters.

This is achieved in our implementation by using :ref:`descriptors<descriptors>`.





Computing the Autocovariance Function
-----------------------------------------

As discussed above, for ARMA processes the spectral density has a :ref:`simple representation <arma_spec_den>` that is relatively easy to calculate.

Given this fact, the easiest way to obtain the autocovariance function is to recover it from the spectral
density via the inverse Fourier transform.



Here we use NumPy's Fourier transform package `np.fft`, which wraps a standard Fortran-based package called FFTPACK.





A look at `the np.fft documentation <http://docs.scipy.org/doc/numpy/reference/routines.fft.html>`_ shows that the inverse transform `np.fft.ifft` takes a given sequence :math:`A_0, A_1, \ldots, A_{n-1}` and
returns the sequence :math:`a_0, a_1, \ldots, a_{n-1}` defined by

.. math::

    a_k = \frac{1}{n} \sum_{t=0}^{n-1} A_t e^{ik 2\pi t / n}


Thus, if we set :math:`A_t = f(\omega_t)`, where :math:`f` is the spectral density and
:math:`\omega_t := 2 \pi t / n`, then

.. math::

    a_k
    = \frac{1}{n} \sum_{t=0}^{n-1} f(\omega_t) e^{i \omega_t k}
    = \frac{1}{2\pi} \frac{2 \pi}{n} \sum_{t=0}^{n-1} f(\omega_t) e^{i \omega_t k},
    \qquad
    \omega_t := 2 \pi t / n


For :math:`n` sufficiently large, we then have

.. math::

    a_k
    \approx \frac{1}{2\pi} \int_0^{2 \pi} f(\omega) e^{i \omega k} d \omega
    = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(\omega) e^{i \omega k} d \omega


(You can check the last equality)

In view of :eq:`ift`, we have now shown that, for :math:`n` sufficiently large, :math:`a_k \approx \gamma(k)` --- which is exactly what we want to compute.

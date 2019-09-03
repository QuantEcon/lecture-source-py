.. _kalman:

.. include:: /_static/includes/header.raw

.. highlight:: python3

*********************************
A First Look at the Kalman Filter
*********************************

.. index::
    single: Kalman Filter

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
========

This lecture provides a simple and intuitive introduction to the Kalman filter, for those who either

* have heard of the Kalman filter but don't know how it works, or
* know the Kalman filter equations, but don't know where they come from

For additional (more advanced) reading on the Kalman filter, see

* :cite:`Ljungqvist2012`, section 2.7
* :cite:`AndersonMoore2005`

The second reference presents a  comprehensive treatment of the Kalman filter.

Required knowledge: Familiarity with matrix manipulations, multivariate normal distributions, covariance matrices, etc.

Let's start with some standard imports:

.. code-block:: ipython

    from scipy import linalg
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    %matplotlib inline
    from quantecon import Kalman, LinearStateSpace
    from scipy.stats import norm
    from scipy.integrate import quad
    from numpy.random import multivariate_normal
    from scipy.linalg import eigvals


The Basic Idea
==============

The Kalman filter has many applications in economics, but for now
let's pretend that we are rocket scientists.

A missile has been launched from country Y and our mission is to track it.

Let :math:`x  \in \mathbb{R}^2` denote the current location of the missile---a
pair indicating latitude-longitude coordinates on a map.

At the present moment in time, the precise location :math:`x` is unknown, but
we do have some beliefs about :math:`x`.

One way to summarize our knowledge is a point prediction :math:`\hat x`

* But what if the President wants to know the probability that the missile is currently over the Sea of Japan?
* Then it is better to summarize our initial beliefs with a bivariate probability density :math:`p`

    * :math:`\int_E p(x)dx` indicates the probability that we attach to the missile being in region :math:`E`.

The density :math:`p` is called our *prior* for the random variable :math:`x`.

To keep things tractable in our example,  we  assume that our prior is Gaussian.

In particular, we take

.. math::
    :label: prior

    p = N(\hat x, \Sigma)


where :math:`\hat x` is the mean of the distribution and :math:`\Sigma` is a
:math:`2 \times 2` covariance matrix.  In our simulations, we will suppose that

.. math::
    :label: kalman_dhxs

    \hat x
    = \left(
    \begin{array}{c}
        0.2 \\
        -0.2
    \end{array}
      \right),
    \qquad
    \Sigma
    = \left(
    \begin{array}{cc}
        0.4 & 0.3 \\
        0.3 & 0.45
    \end{array}
      \right)


This density :math:`p(x)` is shown below as a contour map, with the center of the red ellipse being equal to :math:`\hat x`.



.. code-block:: python3
  :class: collapse


  # Set up the Gaussian prior density p
  Σ = [[0.4, 0.3], [0.3, 0.45]]
  Σ = np.matrix(Σ)
  x_hat = np.matrix([0.2, -0.2]).T
  # Define the matrices G and R from the equation y = G x + N(0, R)
  G = [[1, 0], [0, 1]]
  G = np.matrix(G)
  R = 0.5 * Σ
  # The matrices A and Q
  A = [[1.2, 0], [0, -0.2]]
  A = np.matrix(A)
  Q = 0.3 * Σ
  # The observed value of y
  y = np.matrix([2.3, -1.9]).T

  # Set up grid for plotting
  x_grid = np.linspace(-1.5, 2.9, 100)
  y_grid = np.linspace(-3.1, 1.7, 100)
  X, Y = np.meshgrid(x_grid, y_grid)

  def bivariate_normal(x, y, σ_x=1.0, σ_y=1.0, μ_x=0.0, μ_y=0.0, σ_xy=0.0):
      """
      Compute and return the probability density function of bivariate normal
      distribution of normal random variables x and y

      Parameters
      ----------
      x : array_like(float)
          Random variable

      y : array_like(float)
          Random variable

      σ_x : array_like(float)
            Standard deviation of random variable x

      σ_y : array_like(float)
            Standard deviation of random variable y

      μ_x : scalar(float)
            Mean value of random variable x

      μ_y : scalar(float)
            Mean value of random variable y

      σ_xy : array_like(float)
             Covariance of random variables x and y

      """

      x_μ = x - μ_x
      y_μ = y - μ_y

      ρ = σ_xy / (σ_x * σ_y)
      z = x_μ**2 / σ_x**2 + y_μ**2 / σ_y**2 - 2 * ρ * x_μ * y_μ / (σ_x * σ_y)
      denom = 2 * np.pi * σ_x * σ_y * np.sqrt(1 - ρ**2)
      return np.exp(-z / (2 * (1 - ρ**2))) / denom

  def gen_gaussian_plot_vals(μ, C):
      "Z values for plotting the bivariate Gaussian N(μ, C)"
      m_x, m_y = float(μ[0]), float(μ[1])
      s_x, s_y = np.sqrt(C[0, 0]), np.sqrt(C[1, 1])
      s_xy = C[0, 1]
      return bivariate_normal(X, Y, s_x, s_y, m_x, m_y, s_xy)

  # Plot the figure

  fig, ax = plt.subplots(figsize=(10, 8))
  ax.grid()

  Z = gen_gaussian_plot_vals(x_hat, Σ)
  ax.contourf(X, Y, Z, 6, alpha=0.6, cmap=cm.jet)
  cs = ax.contour(X, Y, Z, 6, colors="black")
  ax.clabel(cs, inline=1, fontsize=10)

  plt.show()




The Filtering Step
------------------

We are now presented with some good news and some bad news.

The good news is that the missile has been located by our sensors, which report that the current location is :math:`y = (2.3, -1.9)`.

The next figure shows the original prior :math:`p(x)` and the new reported
location :math:`y`



.. code-block:: python3

  fig, ax = plt.subplots(figsize=(10, 8))
  ax.grid()

  Z = gen_gaussian_plot_vals(x_hat, Σ)
  ax.contourf(X, Y, Z, 6, alpha=0.6, cmap=cm.jet)
  cs = ax.contour(X, Y, Z, 6, colors="black")
  ax.clabel(cs, inline=1, fontsize=10)
  ax.text(float(y[0]), float(y[1]), "$y$", fontsize=20, color="black")

  plt.show()



The bad news is that our sensors are imprecise.

In particular, we should interpret the output of our sensor not as
:math:`y=x`, but rather as

.. math::
    :label: kl_measurement_model

    y = G x + v, \quad \text{where} \quad v \sim N(0, R)


Here :math:`G` and :math:`R` are :math:`2 \times 2` matrices with :math:`R`
positive definite.  Both are assumed known, and the noise term :math:`v` is assumed
to be independent of :math:`x`.

How then should we combine our prior :math:`p(x) = N(\hat x, \Sigma)` and this
new information :math:`y` to improve our understanding of the location of the
missile?

As you may have guessed, the answer is to use Bayes' theorem, which tells
us to  update our prior :math:`p(x)` to :math:`p(x \,|\, y)` via

.. math::

    p(x \,|\, y) = \frac{p(y \,|\, x) \, p(x)} {p(y)}


where :math:`p(y) = \int p(y \,|\, x) \, p(x) dx`.

In solving for :math:`p(x \,|\, y)`, we observe that

* :math:`p(x) = N(\hat x, \Sigma)`.
* In view of :eq:`kl_measurement_model`, the conditional density :math:`p(y \,|\, x)` is :math:`N(Gx, R)`.
* :math:`p(y)` does not depend on :math:`x`, and enters into the calculations only as a normalizing constant.

Because we are in a linear and Gaussian framework, the updated density can be computed by calculating population linear regressions.

In particular, the solution is known [#f1]_ to be

.. math::

    p(x \,|\, y) = N(\hat x^F, \Sigma^F)


where

.. math::
    :label: kl_filter_exp

    \hat x^F := \hat x + \Sigma G' (G \Sigma G' + R)^{-1}(y - G \hat x)
    \quad \text{and} \quad
    \Sigma^F := \Sigma - \Sigma G' (G \Sigma G' + R)^{-1} G \Sigma


Here  :math:`\Sigma G' (G \Sigma G' + R)^{-1}` is the matrix of population regression coefficients of the hidden object :math:`x - \hat x` on the surprise :math:`y - G \hat x`.

This new density :math:`p(x \,|\, y) = N(\hat x^F, \Sigma^F)` is shown in the next figure via contour lines and the color map.

The original density is left in as contour lines for comparison



.. code-block:: python3

  fig, ax = plt.subplots(figsize=(10, 8))
  ax.grid()

  Z = gen_gaussian_plot_vals(x_hat, Σ)
  cs1 = ax.contour(X, Y, Z, 6, colors="black")
  ax.clabel(cs1, inline=1, fontsize=10)
  M = Σ * G.T * linalg.inv(G * Σ * G.T + R)
  x_hat_F = x_hat + M * (y - G * x_hat)
  Σ_F = Σ - M * G * Σ
  new_Z = gen_gaussian_plot_vals(x_hat_F, Σ_F)
  cs2 = ax.contour(X, Y, new_Z, 6, colors="black")
  ax.clabel(cs2, inline=1, fontsize=10)
  ax.contourf(X, Y, new_Z, 6, alpha=0.6, cmap=cm.jet)
  ax.text(float(y[0]), float(y[1]), "$y$", fontsize=20, color="black")

  plt.show()



Our new density twists the prior :math:`p(x)` in a direction determined by  the new
information :math:`y - G \hat x`.

In generating the figure, we set :math:`G` to the identity matrix and :math:`R = 0.5 \Sigma` for :math:`\Sigma` defined in :eq:`kalman_dhxs`.


.. _kl_forecase_step:

The Forecast Step
-----------------


What have we achieved so far?

We have obtained probabilities for the current location of the state (missile) given prior and current information.

This is called "filtering" rather than forecasting because we are filtering
out noise rather than looking into the future.

* :math:`p(x \,|\, y) = N(\hat x^F, \Sigma^F)` is called the *filtering distribution*

But now let's suppose that we are given another task: to predict the location of the missile after one unit of time (whatever that may be) has elapsed.

To do this we need a model of how the state evolves.

Let's suppose that we have one, and that it's linear and Gaussian. In particular,

.. math::
    :label: kl_xdynam

    x_{t+1} = A x_t + w_{t+1}, \quad \text{where} \quad w_t \sim N(0, Q)


Our aim is to combine this law of motion and our current distribution :math:`p(x \,|\, y) = N(\hat x^F, \Sigma^F)` to come up with a new *predictive* distribution for the location in one unit of time.

In view of :eq:`kl_xdynam`, all we have to do is introduce a random vector :math:`x^F \sim N(\hat x^F, \Sigma^F)` and work out the distribution of :math:`A x^F + w` where :math:`w` is independent of :math:`x^F` and has distribution :math:`N(0, Q)`.

Since linear combinations of Gaussians are Gaussian, :math:`A x^F + w` is Gaussian.

Elementary calculations and the expressions in :eq:`kl_filter_exp` tell us that

.. math::

    \mathbb{E} [A x^F + w]
    = A \mathbb{E} x^F + \mathbb{E} w
    = A \hat x^F
    = A \hat x + A \Sigma G' (G \Sigma G' + R)^{-1}(y - G \hat x)


and

.. math::

    \operatorname{Var} [A x^F + w]
    = A \operatorname{Var}[x^F] A' + Q
    = A \Sigma^F A' + Q
    = A \Sigma A' - A \Sigma G' (G \Sigma G' + R)^{-1} G \Sigma A' + Q


The matrix :math:`A \Sigma G' (G \Sigma G' + R)^{-1}` is often written as
:math:`K_{\Sigma}` and called the *Kalman gain*.

* The subscript :math:`\Sigma` has been added to remind us that  :math:`K_{\Sigma}` depends on :math:`\Sigma`, but not :math:`y` or :math:`\hat x`.

Using this notation, we can summarize our results as follows.

Our updated prediction is the density :math:`N(\hat x_{new}, \Sigma_{new})` where

.. math::
    :label: kl_mlom0

    \begin{aligned}
        \hat x_{new} &:= A \hat x + K_{\Sigma} (y - G \hat x) \\
        \Sigma_{new} &:= A \Sigma A' - K_{\Sigma} G \Sigma A' + Q \nonumber
    \end{aligned}


* The density :math:`p_{new}(x) = N(\hat x_{new}, \Sigma_{new})` is called the *predictive distribution*

The predictive distribution is the new density shown in the following figure, where
the update has used parameters.

.. math::

    A
    = \left(
    \begin{array}{cc}
        1.2 & 0.0 \\
        0.0 & -0.2
    \end{array}
      \right),
      \qquad
    Q = 0.3 * \Sigma




.. code-block:: python3

  fig, ax = plt.subplots(figsize=(10, 8))
  ax.grid()

  # Density 1
  Z = gen_gaussian_plot_vals(x_hat, Σ)
  cs1 = ax.contour(X, Y, Z, 6, colors="black")
  ax.clabel(cs1, inline=1, fontsize=10)

  # Density 2
  M = Σ * G.T * linalg.inv(G * Σ * G.T + R)
  x_hat_F = x_hat + M * (y - G * x_hat)
  Σ_F = Σ - M * G * Σ
  Z_F = gen_gaussian_plot_vals(x_hat_F, Σ_F)
  cs2 = ax.contour(X, Y, Z_F, 6, colors="black")
  ax.clabel(cs2, inline=1, fontsize=10)

  # Density 3
  new_x_hat = A * x_hat_F
  new_Σ = A * Σ_F * A.T + Q
  new_Z = gen_gaussian_plot_vals(new_x_hat, new_Σ)
  cs3 = ax.contour(X, Y, new_Z, 6, colors="black")
  ax.clabel(cs3, inline=1, fontsize=10)
  ax.contourf(X, Y, new_Z, 6, alpha=0.6, cmap=cm.jet)
  ax.text(float(y[0]), float(y[1]), "$y$", fontsize=20, color="black")

  plt.show()




The Recursive Procedure
-----------------------

.. index::
    single: Kalman Filter; Recursive Procedure

Let's look back at what we've done.

We started the current period with a prior :math:`p(x)` for the location :math:`x` of the missile.

We then used the current measurement :math:`y` to update to :math:`p(x \,|\, y)`.

Finally, we used the law of motion :eq:`kl_xdynam` for :math:`\{x_t\}` to update to :math:`p_{new}(x)`.

If we now step into the next period, we are ready to go round again, taking :math:`p_{new}(x)`
as the current prior.

Swapping notation :math:`p_t(x)` for :math:`p(x)` and :math:`p_{t+1}(x)` for :math:`p_{new}(x)`, the full recursive procedure is:

1. Start the current period with prior :math:`p_t(x) = N(\hat x_t, \Sigma_t)`.
2. Observe current measurement :math:`y_t`.
3. Compute the filtering distribution :math:`p_t(x \,|\, y) = N(\hat x_t^F, \Sigma_t^F)` from :math:`p_t(x)` and :math:`y_t`, applying Bayes rule and the conditional distribution :eq:`kl_measurement_model`.
4. Compute the predictive distribution :math:`p_{t+1}(x) = N(\hat x_{t+1}, \Sigma_{t+1})` from the filtering distribution and :eq:`kl_xdynam`.
5. Increment :math:`t` by one and go to step 1.

Repeating :eq:`kl_mlom0`, the dynamics for :math:`\hat x_t` and :math:`\Sigma_t` are as follows

.. math::
    :label: kalman_lom

    \begin{aligned}
        \hat x_{t+1} &= A \hat x_t + K_{\Sigma_t} (y_t - G \hat x_t) \\
        \Sigma_{t+1} &= A \Sigma_t A' - K_{\Sigma_t} G \Sigma_t A' + Q \nonumber
    \end{aligned}


These are the standard dynamic equations for the Kalman filter (see, for example, :cite:`Ljungqvist2012`, page 58).

.. _kalman_convergence:

Convergence
===========

The matrix :math:`\Sigma_t` is a measure of the uncertainty of our prediction :math:`\hat x_t` of :math:`x_t`.

Apart from special cases, this uncertainty will never be fully resolved, regardless of how much time elapses.

One reason is that our prediction :math:`\hat x_t` is made based on information available at :math:`t-1`, not :math:`t`.

Even if we know the precise value of :math:`x_{t-1}` (which we don't), the transition equation :eq:`kl_xdynam` implies that :math:`x_t = A x_{t-1} + w_t`.

Since the shock :math:`w_t` is not observable at :math:`t-1`, any time :math:`t-1` prediction of :math:`x_t` will incur some error (unless :math:`w_t` is degenerate).

However, it is certainly possible that :math:`\Sigma_t` converges to a constant matrix as :math:`t \to \infty`.

To study this topic, let's expand the second equation in :eq:`kalman_lom`:

.. math::
    :label: kalman_sdy

    \Sigma_{t+1} = A \Sigma_t A' -  A \Sigma_t G' (G \Sigma_t G' + R)^{-1} G \Sigma_t A' + Q


This is a nonlinear difference equation in :math:`\Sigma_t`.

A fixed point of :eq:`kalman_sdy` is a constant matrix :math:`\Sigma` such that

.. math::
    :label: kalman_dare

    \Sigma = A \Sigma A' -  A \Sigma G' (G \Sigma G' + R)^{-1} G \Sigma A' + Q


Equation :eq:`kalman_sdy` is known as a discrete-time Riccati difference equation.

Equation :eq:`kalman_dare` is known as a `discrete-time algebraic Riccati equation <https://en.wikipedia.org/wiki/Algebraic_Riccati_equation>`_.

Conditions under which a fixed point exists and the sequence :math:`\{\Sigma_t\}` converges to it are discussed in :cite:`AHMS1996` and :cite:`AndersonMoore2005`, chapter 4.

A sufficient (but not necessary) condition is that all the eigenvalues :math:`\lambda_i` of :math:`A` satisfy :math:`|\lambda_i| < 1` (cf. e.g., :cite:`AndersonMoore2005`, p. 77).

(This strong condition assures that the unconditional  distribution of :math:`x_t`  converges as :math:`t \rightarrow + \infty`)

In this case, for any initial choice of :math:`\Sigma_0` that is both non-negative and symmetric, the sequence :math:`\{\Sigma_t\}` in :eq:`kalman_sdy` converges to a non-negative symmetric matrix :math:`\Sigma` that solves :eq:`kalman_dare`.



Implementation
==============

.. index::
    single: Kalman Filter; Programming Implementation



The class ``Kalman`` from the `QuantEcon.py`_ package implements the Kalman filter





* Instance data consists of:

    * the moments :math:`(\hat x_t, \Sigma_t)` of the current prior.

    * An instance of the `LinearStateSpace <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lss.py>`_ class from `QuantEcon.py <http://quantecon.org/python_index.html>`_.


The latter represents a linear state space model of the form

.. math::

    \begin{aligned}
        x_{t+1} & = A x_t + C w_{t+1}
        \\
        y_t & = G x_t + H v_t
    \end{aligned}


where the shocks :math:`w_t` and :math:`v_t` are IID standard normals.

To connect this with the notation of this lecture we set

.. math::

    Q := CC' \quad \text{and} \quad R := HH'




* The class ``Kalman`` from the `QuantEcon.py <http://quantecon.org/python_index.html>`_ package has a number of methods, some that we will wait to use until we study more advanced applications in subsequent lectures.

* Methods pertinent for this lecture  are:

    * ``prior_to_filtered``, which updates :math:`(\hat x_t, \Sigma_t)` to :math:`(\hat x_t^F, \Sigma_t^F)`

    * ``filtered_to_forecast``, which updates the filtering distribution to the predictive distribution -- which becomes the new prior :math:`(\hat x_{t+1}, \Sigma_{t+1})`

    * ``update``, which combines the last two methods

    * a ``stationary_values``, which computes the solution to :eq:`kalman_dare` and the corresponding (stationary) Kalman gain


You can view the program `on GitHub <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/kalman.py>`__.



Exercises
=========


.. _kalman_ex1:

Exercise 1
----------

Consider the following simple application of the Kalman filter, loosely based
on :cite:`Ljungqvist2012`, section 2.9.2.

Suppose that

* all variables are scalars
* the hidden state :math:`\{x_t\}` is in fact constant, equal to some :math:`\theta \in \mathbb{R}` unknown to the modeler

State dynamics are therefore given by :eq:`kl_xdynam` with :math:`A=1`, :math:`Q=0` and :math:`x_0 = \theta`.

The measurement equation is :math:`y_t = \theta + v_t` where :math:`v_t` is :math:`N(0,1)` and IID.

The task of this exercise to simulate the model and, using the code from ``kalman.py``, plot the first five predictive densities :math:`p_t(x) = N(\hat x_t, \Sigma_t)`.

As shown in :cite:`Ljungqvist2012`, sections 2.9.1--2.9.2, these distributions asymptotically put all mass on the unknown value :math:`\theta`.

In the simulation, take :math:`\theta = 10`, :math:`\hat x_0 = 8` and :math:`\Sigma_0 = 1`.

Your figure should -- modulo randomness -- look something like this

.. figure:: /_static/lecture_specific/kalman/kl_ex1_fig.png


.. _kalman_ex2:

Exercise 2
----------

The preceding figure gives some support to the idea that probability mass
converges to :math:`\theta`.

To get a better idea, choose a small :math:`\epsilon > 0` and calculate

.. math::

    z_t := 1 - \int_{\theta - \epsilon}^{\theta + \epsilon} p_t(x) dx


for :math:`t = 0, 1, 2, \ldots, T`.

Plot :math:`z_t` against :math:`T`, setting :math:`\epsilon = 0.1` and :math:`T = 600`.

Your figure should show error erratically declining something like this

.. figure:: /_static/lecture_specific/kalman/kl_ex2_fig.png


.. _kalman_ex3:

Exercise 3
----------

As discussed :ref:`above <kalman_convergence>`, if the shock sequence :math:`\{w_t\}` is not degenerate, then it is not in general possible to predict :math:`x_t` without error at time :math:`t-1` (and this would be the case even if we could observe :math:`x_{t-1}`).

Let's now compare the prediction :math:`\hat x_t` made by the Kalman filter
against a competitor who **is** allowed to observe :math:`x_{t-1}`.

This competitor will use the conditional expectation :math:`\mathbb E[ x_t
\,|\, x_{t-1}]`, which in this case is :math:`A x_{t-1}`.

The conditional expectation is known to be the optimal prediction method in terms of minimizing mean squared error.

(More precisely, the minimizer of :math:`\mathbb E \, \| x_t - g(x_{t-1}) \|^2` with respect to :math:`g` is :math:`g^*(x_{t-1}) := \mathbb E[ x_t \,|\, x_{t-1}]`)

Thus we are comparing the Kalman filter against a competitor who has more
information (in the sense of being able to observe the latent state) and
behaves optimally in terms of minimizing squared error.

Our horse race will be assessed in terms of squared error.

In particular, your task is to generate a graph plotting observations of both :math:`\| x_t - A x_{t-1} \|^2` and :math:`\| x_t - \hat x_t \|^2` against :math:`t` for :math:`t = 1, \ldots, 50`.

For the parameters, set :math:`G = I, R = 0.5 I` and :math:`Q = 0.3 I`, where :math:`I` is
the :math:`2 \times 2` identity.

Set

.. math::

    A
    = \left(
    \begin{array}{cc}
        0.5 & 0.4 \\
        0.6 & 0.3
    \end{array}
      \right)


To initialize the prior density, set

.. math::

    \Sigma_0
    = \left(
    \begin{array}{cc}
        0.9 & 0.3 \\
        0.3 & 0.9
    \end{array}
      \right)


and :math:`\hat x_0 = (8, 8)`.

Finally, set :math:`x_0 = (0, 0)`.

You should end up with a figure similar to the following (modulo randomness)

.. figure:: /_static/lecture_specific/kalman/kalman_ex3.png

Observe how, after an initial learning period, the Kalman filter performs quite well, even relative to the competitor who predicts optimally with knowledge of the latent state.




.. _kalman_ex4:

Exercise 4
----------------

Try varying the coefficient :math:`0.3` in :math:`Q = 0.3 I` up and down.

Observe how the diagonal values in the stationary solution :math:`\Sigma` (see :eq:`kalman_dare`) increase and decrease in line with this coefficient.

The interpretation is that more randomness in the law of motion for :math:`x_t` causes more (permanent) uncertainty in prediction.




Solutions
==========

Exercise 1
----------

.. code-block:: python3

    # Parameters
    θ = 10  # Constant value of state x_t
    A, C, G, H = 1, 0, 1, 1
    ss = LinearStateSpace(A, C, G, H, mu_0=θ)

    # Set prior, initialize kalman filter
    x_hat_0, Σ_0 = 8, 1
    kalman = Kalman(ss, x_hat_0, Σ_0)

    # Draw observations of y from state space model
    N = 5
    x, y = ss.simulate(N)
    y = y.flatten()

    # Set up plot
    fig, ax = plt.subplots(figsize=(10,8))
    xgrid = np.linspace(θ - 5, θ + 2, 200)

    for i in range(N):
        # Record the current predicted mean and variance
        m, v = [float(z) for z in (kalman.x_hat, kalman.Sigma)]
        # Plot, update filter
        ax.plot(xgrid, norm.pdf(xgrid, loc=m, scale=np.sqrt(v)), label=f'$t={i}$')
        kalman.update(y[i])

    ax.set_title(f'First {N} densities when $\\theta = {θ:.1f}$')
    ax.legend(loc='upper left')
    plt.show()


Exercise 2
----------

.. code-block:: python3

    ϵ = 0.1
    θ = 10  # Constant value of state x_t
    A, C, G, H = 1, 0, 1, 1
    ss = LinearStateSpace(A, C, G, H, mu_0=θ)

    x_hat_0, Σ_0 = 8, 1
    kalman = Kalman(ss, x_hat_0, Σ_0)

    T = 600
    z = np.empty(T)
    x, y = ss.simulate(T)
    y = y.flatten()

    for t in range(T):
        # Record the current predicted mean and variance and plot their densities
        m, v = [float(temp) for temp in (kalman.x_hat, kalman.Sigma)]

        f = lambda x: norm.pdf(x, loc=m, scale=np.sqrt(v))
        integral, error = quad(f, θ - ϵ, θ + ϵ)
        z[t] = 1 - integral

        kalman.update(y[t])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, T)
    ax.plot(range(T), z)
    ax.fill_between(range(T), np.zeros(T), z, color="blue", alpha=0.2)
    plt.show()


Exercise 3
----------

.. code-block:: python3

    # Define A, C, G, H
    G = np.identity(2)
    H = np.sqrt(0.5) * np.identity(2)

    A = [[0.5, 0.4],
         [0.6, 0.3]]
    C = np.sqrt(0.3) * np.identity(2)

    # Set up state space mode, initial value x_0 set to zero
    ss = LinearStateSpace(A, C, G, H, mu_0 = np.zeros(2))

    # Define the prior density
    Σ = [[0.9, 0.3],
         [0.3, 0.9]]
    Σ = np.array(Σ)
    x_hat = np.array([8, 8])

    # Initialize the Kalman filter
    kn = Kalman(ss, x_hat, Σ)

    # Print eigenvalues of A
    print("Eigenvalues of A:")
    print(eigvals(A))

    # Print stationary Σ
    S, K = kn.stationary_values()
    print("Stationary prediction error variance:")
    print(S)

    # Generate the plot
    T = 50
    x, y = ss.simulate(T)

    e1 = np.empty(T-1)
    e2 = np.empty(T-1)

    for t in range(1, T):
        kn.update(y[:,t])
        e1[t-1] = np.sum((x[:, t] - kn.x_hat.flatten())**2)
        e2[t-1] = np.sum((x[:, t] - A @ x[:, t-1])**2)

    fig, ax = plt.subplots(figsize=(9,6))
    ax.plot(range(1, T), e1, 'k-', lw=2, alpha=0.6,
            label='Kalman filter error')
    ax.plot(range(1, T), e2, 'g-', lw=2, alpha=0.6, 
            label='Conditional expectation error')
    ax.legend()
    plt.show()




.. rubric:: Footnotes

.. [#f1] See, for example, page 93 of :cite:`Bishop2006`. To get from his expressions to the ones used above, you will also need to apply the `Woodbury matrix identity <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_.

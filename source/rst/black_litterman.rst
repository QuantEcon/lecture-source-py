.. _black_litterman:

.. include:: /_static/includes/lecture_howto_py.raw

.. index::
    single: python

****************************************************
Two modifications of mean-variance portfolio theory
****************************************************

.. contents:: :depth: 2

**Authors:** Daniel Csaba, Thomas J. Sargent and Balint Szoke

Overview
=========

Remarks about estimating means and variances
----------------------------------------------

The famous **Black-Litterman** (1992) :cite:`black1992global` portfolio choice model that we
describe in this notebook is motivated by the finding that with high or
moderate frequency data, means are more difficult to estimate than
variances

A model of **robust portfolio choice** that we'll describe also begins
from the same starting point

To begin, we'll take for granted that means are more difficult to
estimate that covariances and will focus on how Black and Litterman, on
the one hand, an robust control theorists, on the other, would recommend
modifying the **mean-variance portfolio choice model** to take that into
account

At the end of this notebook, we shall use some rates of convergence
results and some simulations to verify how means are more difficult to
estimate than variances

Among the ideas in play in this notebook will be

-  Mean-variance portfolio theory

-  Bayesian approaches to estimating linear regressions

-  A risk-sensitivity operator and its connection to robust control
   theory

.. code-block:: ipython

    import numpy as np
    import scipy as sp
    import scipy.stats as stat
    import matplotlib.pyplot as plt
    %matplotlib inline
    from ipywidgets import interact, FloatSlider


Adjusting mean-variance portfolio choice theory for distrust of mean excess returns
-----------------------------------------------------------------------------------

This lecture describes two lines of thought that modify the classic
mean-variance portfolio choice model in ways designed to make its
recommendations more plausible

As we mentioned above, the two approaches build on a common hunch --
that because it is much easier statistically to estimate covariances of
excess returns than it is to estimate their means, it makes sense to
contemplated the consequences of adjusting investors' subjective beliefs
about mean returns in order to render more sensible decisions

Both of the adjustments that we describe are designed to confront a
widely recognized embarrassment to mean-variance portfolio theory,
namely, that it usually implies taking very extreme long-short portfolio
positions

Mean-variance portfolio choice
-------------------------------

A risk free security earns one-period net return :math:`r_f`

An :math:`n \times 1` vector of risky securities earns
an :math:`n \times 1` vector :math:`\vec r - r_f {\bf 1}` of *excess
returns*, where :math:`{\bf 1}` is an :math:`n \times 1` vector of
ones

The excess return vector is multivariate normal with mean :math:`\mu`
and covariance matrix :math:`\Sigma`, which we express either as

.. math:: \vec r - r_f {\bf 1} \sim {\mathcal N}(\mu, \Sigma)

or

.. math:: \vec r - r_f {\bf 1} = \mu + C \epsilon 

where :math:`\epsilon \sim {\mathcal N}(0, I)` is an :math:`n \times 1`
random vector.

Let :math:`w` be an :math:`n \times 1`  vector of portfolio weights

A portfolio consisting :math:`w` earns returns

.. math:: w' (\vec r - r_f {\bf 1}) \sim {\mathcal N}(w' \mu, w' \Sigma w )

The **mean-variance portfolio choice problem** is to choose :math:`w` to
maximize

.. math::
  :label: choice-problem
  
  U(\mu,\Sigma;w) = w'\mu - \frac{\delta}{2} w' \Sigma w

where :math:`\delta > 0` is a risk-aversion parameter. The first-order
condition for maximizing :eq:`choice-problem` with respect to the vector :math:`w` is

.. math:: \mu = \delta \Sigma w

which implies the following design of a risky portfolio:

.. math:: 
  :label: risky-portfolio
  
  w = (\delta \Sigma)^{-1} \mu

Estimating :math:`\mu` and :math:`\Sigma`
--------------------------------------------

The key inputs into the portfolio choice model :eq:`risky-portfolio` are

-  estimates of the parameters :math:`\mu, \Sigma` of the random excess
   return vector\ :math:`(\vec r - r_f {\bf 1})`

-  the risk-aversion parameter :math:`\delta`

A standard way of estimating :math:`\mu` is maximum-likelihood or least
squares; that amounts to estimating :math:`\mu` by a sample mean of
excess returns and estimating :math:`\Sigma` by a sample covariance
matrix

The Black-Litterman starting point
--------------------------------------------
                                  

When estimates of :math:`\mu` and :math:`\Sigma` from historical
sample means and covariances have been combined with **reasonable** values
of the risk-aversion parameter :math:`\delta` to compute an
optimal portfolio from formula :eq:`risky-portfolio`, a typical outcome has been
:math:`w`'s with **extreme long and short positions**

A common reaction to these outcomes is that they are so unreasonable that a portfolio
manager cannot recommend them to a customer

.. code-block:: python3

    np.random.seed(12)

    N = 10                                           # Number of assets
    T = 200                                          # Sample size

    # random market portfolio (sum is normalized to 1)
    w_m = np.random.rand(N) 
    w_m = w_m / (w_m.sum())                      

    # True risk premia and variance of excess return (constructed so that the Sharpe ratio is 1)
    μ = (np.random.randn(N) + 5)  /100                   # Mean excess return (risk premium)   
    S = np.random.randn(N, N)                            # Random matrix for the covariance matrix
    V = S @ S.T                                          # Turn the random matrix into symmetric psd   
    Σ = V * (w_m @ μ)**2 / (w_m @ V @ w_m)               # Make sure that the Sharpe ratio is one

    # Risk aversion of market portfolio holder
    δ = 1 / np.sqrt(w_m @ Σ @ w_m)

    # Generate a sample of excess returns
    excess_return = stat.multivariate_normal(μ, Σ)
    sample = excess_return.rvs(T)

    # Estimate μ and Σ
    μ_est = sample.mean(0).reshape(N, 1)
    Σ_est = np.cov(sample.T)

    w = np.linalg.solve(δ * Σ_est, μ_est)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('Mean-variance portfolio weights recommendation and the market portfolio')
    ax.plot(np.arange(N)+1, w, 'o', c='k', label='$w$ (mean-variance)')
    ax.plot(np.arange(N)+1, w_m, 'o', c='r', label='$w_m$ (market portfolio)')
    ax.vlines(np.arange(N)+1, 0, w, lw=1)
    ax.vlines(np.arange(N)+1, 0, w_m, lw=1)
    ax.axhline(0, c='k')
    ax.axhline(-1, c='k', ls='--')
    ax.axhline(1, c='k', ls='--')
    ax.set_xlabel('Assets')
    ax.xaxis.set_ticks(np.arange(1, N+1, 1))
    plt.legend(numpoints=1, fontsize=11)
    plt.show()

Black and Litterman's responded to this situation in the following way:

-  They continue to accept :eq:`risky-portfolio` as a good model for choosing an optimal
   portfolio :math:`w`

-  They want to continue to allow the customer to express his or her
   risk tolerance by setting :math:`\delta`

-  Leaving :math:`\Sigma` at its maximum-likelihood value, they push
   :math:`\mu` away from its maximum value in a way designed to make
   portfolio choices that are more plausible in terms of conforming to
   what most people actually do

In particular, given :math:`\Sigma` and a reasonable value
of :math:`\delta`, Black and Litterman reverse engineered a vector
:math:`\mu_{BL}` of mean excess returns that makes the :math:`w`
implied by formula :eq:`risky-portfolio` equal the **actual** market portfolio
:math:`w_m`, so that

.. math:: w_m = (\delta \Sigma)^{-1} \mu_{BL}

Details
--------

Let's define

.. math:: w_m' \mu \equiv ( r_m - r_f) 

as the (scalar) excess return on the market portfolio :math:`w_m`

Define

.. math:: \sigma^2 = w_m' \Sigma w_m 

as the variance of the excess return on the market portfolio
:math:`w_m`

Define

.. math:: {\bf SR}_m = \frac{ r_m - r_f}{\sigma} 

as the **Sharpe-ratio** on the market portfolio :math:`w_m`

Let :math:`\delta_m` be the value of the risk aversion parameter that
induces an investor to hold the market portfolio in light of the optimal
portfolio choice rule :eq:`risky-portfolio`

Evidently, portfolio rule :eq:`risky-portfolio` then implies that
:math:`r_m - r_f = \delta_m \sigma^2` or

.. math:: \delta_m = \frac{r_m - r_f}{\sigma^2} 

or

.. math:: \delta_m = \frac{\bf SR}{\sigma}

Following the Black-Litterman philosophy, our first step will be to back
a value of :math:`\delta_m` from

-  an estimate of the Sharpe-ratio, and

-  our maximum likelihood estimate of :math:`\sigma` drawn from our
   estimates or :math:`w_m` and :math:`\Sigma`

The second key Black-Litterman step is then to use this value of
:math:`\delta` together with the maximum likelihood estimate of
:math:`\Sigma` to deduce a :math:`\mu_{\bf BL}` that verifies
portfolio rule :eq:`risky-portfolio` at the market portfolio :math:`w = w_m`

.. math:: \mu_m = \delta_m \Sigma w_m  

The starting point of the Black-Litterman portfolio choice model is thus
a pair :math:`(\delta_m, \mu_m)` that tells the customer to hold the
market portfolio

.. code-block:: python3

    # Observed mean excess market return
    r_m = w_m @ μ_est

    # Estimated variance of market portfolio
    σ_m = w_m @ Σ_est @ w_m

    # Sharpe-ratio
    SR_m = r_m / np.sqrt(σ_m)

    # Risk aversion of market portfolio holder
    d_m = r_m / σ_m

    # Derive "view" which would induce market portfolio
    μ_m = (d_m * Σ_est @ w_m).reshape(N, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(r'Difference between $\hat{\mu}$ (estimate) and $\mu_{BL}$ (market implied)')
    ax.plot(np.arange(N)+1, μ_est, 'o', c='k', label='$\hat{\mu}$')
    ax.plot(np.arange(N)+1, μ_m, 'o', c='r', label='$\mu_{BL}$')
    ax.vlines(np.arange(N) + 1, μ_m, μ_est, lw=1)
    ax.axhline(0, c='k', ls='--')
    ax.set_xlabel('Assets')
    ax.xaxis.set_ticks(np.arange(1, N+1, 1))
    plt.legend(numpoints=1)
    plt.show()


Adding *views*
-------------------

Black and Litterman start with a baseline customer who asserts that he
or she shares the **market's views**, which means that his or her
believes that excess returns are governed by

.. math::
  :label: excess-returns
  
  \vec r - r_f {\bf 1} \sim {\mathcal N}( \mu_{BL}, \Sigma)

Black and Litterman would advise that customer to hold the market
portfolio of risky securities

Black and Litterman then imagine a consumer who would like to express a
view that differs from the market's

The consumer wants appropriately to
mix his view with the market's before using :eq:`risky-portfolio` to choose a portfolio

Suppose that the customer's view is expressed by a hunch that rather
than :eq:`excess-returns`, excess returns are governed by

.. math:: \vec r - r_f {\bf 1} \sim {\mathcal N}( \hat \mu, \tau \Sigma)

where :math:`\tau > 0` is a scalar parameter that determines how the
decision maker wants to mix his view :math:`\hat \mu` with the market's
view :math:`\mu_{\bf BL}`

Black and Litterman would then use a formula like the following one to
mix the views :math:`\hat \mu` and :math:`\mu_{\bf BL}`

.. math::
  :label: mix-views
  
  \tilde \mu = (\Sigma^{-1} + (\tau \Sigma)^{-1})^{-1} (\Sigma^{-1} \mu_{BL}  + (\tau \Sigma)^{-1} \hat \mu)

Black and Litterman would then advice the customer to hold the portfolio
associated with these views implied by rule :eq:`risky-portfolio`:

.. math:: \tilde w = (\delta \Sigma)^{-1} \tilde \mu

This portfolio :math:`\tilde w` will deviate from the
portfolio :math:`w_{BL}` in amounts that depend on the mixing parameter
:math:`\tau`.

If :math:`\hat \mu` is the maximum likelihood estimator
and :math:`\tau` is chosen heavily to weight this view, then the
customer's portfolio will involve big short-long positions

.. code-block:: python3

    def black_litterman(λ, μ1, μ2, Σ1, Σ2):
        """
        This function calculates the Black-Litterman mixture
        mean excess return and covariance matrix
        """
        Σ1_inv = np.linalg.inv(Σ1)
        Σ2_inv = np.linalg.inv(Σ2)

        μ_tilde = np.linalg.solve(Σ1_inv + λ * Σ2_inv,
                                  Σ1_inv @ μ1 + λ * Σ2_inv @ μ2)
        return μ_tilde

    τ = 1
    μ_tilde = black_litterman(1, μ_m, μ_est, Σ_est, τ * Σ_est)

    # The Black-Litterman recommendation for the portfolio weights
    w_tilde = np.linalg.solve(δ * Σ_est, μ_tilde)

    τ_slider = FloatSlider(min=0.05, max=10, step=0.5, value=τ)

    @interact(τ=τ_slider)
    def BL_plot(τ):
        μ_tilde = black_litterman(1, μ_m, μ_est, Σ_est, τ * Σ_est)
        w_tilde = np.linalg.solve(δ * Σ_est, μ_tilde)

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].plot(np.arange(N)+1, μ_est, 'o', c='k', label=r'$\hat{\mu}$ (subj view)')
        ax[0].plot(np.arange(N)+1, μ_m, 'o', c='r', label=r'$\mu_{BL}$ (market)')
        ax[0].plot(np.arange(N)+1, μ_tilde, 'o', c='y', label=r'$\tilde{\mu}$ (mixture)')
        ax[0].vlines(np.arange(N)+1, μ_m, μ_est, lw=1)
        ax[0].axhline(0, c='k', ls='--')
        ax[0].set(xlim=(0, N+1), xlabel='Assets',
                  title=r'Relationship between $\hat{\mu}$, $\mu_{BL}$and$\tilde{\mu}$')
        ax[0].xaxis.set_ticks(np.arange(1, N+1, 1))
        ax[0].legend(numpoints=1)

        ax[1].set_title('Black-Litterman portfolio weight recommendation')
        ax[1].plot(np.arange(N)+1, w, 'o', c='k', label=r'$w$ (mean-variance)')
        ax[1].plot(np.arange(N)+1, w_m, 'o', c='r', label=r'$w_{m}$ (market, BL)')
        ax[1].plot(np.arange(N)+1, w_tilde, 'o', c='y', label=r'$\tilde{w}$ (mixture)')
        ax[1].vlines(np.arange(N)+1, 0, w, lw=1)
        ax[1].vlines(np.arange(N)+1, 0, w_m, lw=1)
        ax[1].axhline(0, c='k')
        ax[1].axhline(-1, c='k', ls='--')
        ax[1].axhline(1, c='k', ls='--')
        ax[1].set(xlim=(0, N+1), xlabel='Assets', 
                  title='Black-Litterman portfolio weight recommendation')
        ax[1].xaxis.set_ticks(np.arange(1, N+1, 1))
        ax[1].legend(numpoints=1)
        plt.show()


Bayes interpretation of the Black-Litterman recommendation
-----------------------------------------------------------

Consider the following Bayesian interpretation of the Black-Litterman
recommendation

The prior belief over the mean excess returns is consistent with the
market porfolio and is given by

.. math:: \mu \sim \mathcal{N}(\mu_{BL}, \Sigma)

Given a particular realization of the mean excess returns
:math:`\mu` one observes the average excess returns :math:`\hat \mu`
on the market according to the distribution

.. math:: \hat \mu \mid \mu, \Sigma \sim \mathcal{N}(\mu, \tau\Sigma)

where :math:`\tau` is typically small capturing the idea that the
variation in the mean is smaller than the variation of the individual
random variable

Given the realized excess returns one should then update the prior over
the mean excess returns according to Bayes rule

The corresponding
posterior over mean excess returns is normally distributed with mean

.. math:: (\Sigma^{-1} + (\tau \Sigma)^{-1})^{-1} (\Sigma^{-1}\mu_{BL}   + (\tau \Sigma)^{-1} \hat \mu)

The covariance matrix is

.. math:: (\Sigma^{-1} + (\tau \Sigma)^{-1})^{-1}

Hence, the Black-Litterman recommendation is consistent with the Bayes
update of the prior over the mean excess returns in light of the
realized average excess returns on the market

Curve Decolletage
------------------

Consider two independent "competing" views on the excess market returns

.. math:: \vec r_e  \sim {\mathcal N}( \mu_{BL}, \Sigma)

and

.. math:: \vec r_e \sim {\mathcal N}( \hat{\mu}, \tau\Sigma)

A special feature of the multivariate normal random variable
:math:`Z` is that its density function depends only on the (Euclidiean)
length of its realization :math:`z`

Formally, let the
:math:`k`-dimensional random vector be

.. math:: Z\sim \mathcal{N}(\mu, \Sigma)

then 

.. math:: \bar{Z} \equiv \Sigma(Z-\mu)\sim \mathcal{N}(\mathbf{0}, I)

and so the points where the density takes the same value can be
described by the ellipse

.. math::
  :label: ellipse
  
  \bar z \cdot \bar z =  (z - \mu)'\Sigma^{-1}(z - \mu) = \bar d

where :math:`\bar d\in\mathbb{R}_+` denotes the (transformation) of a
particular density value

The curves defined by equation :eq:`ellipse` can be
labelled as iso-likelihood ellipses

    **Remark:** More generally there is a class of density functions
    that possesses this feature, i.e.

    .. math:: 
      
      \exists g: \mathbb{R}_+ \mapsto \mathbb{R}_+ \ \ \text{ and } \ \ c \geq 0,
      \ \ \text{s.t.  the density } \ \ f \ \ \text{of} \ \ Z  \ \ 
      \text{ has the form } \quad f(z) = c g(z\cdot z)

    This property is called **spherical symmetry** (see p 81. in Leamer
    (1978) :cite:`leamer1978specification`)

In our specific example, we can use the pair
:math:`(\bar d_1, \bar d_2)` as being two "likelihood" values for which
the corresponding isolikelihood ellipses in the excess return space are
given by

.. math::

   \begin{align}
   (\vec r_e - \mu_{BL})'\Sigma^{-1}(\vec r_e - \mu_{BL}) &= \bar d_1 \\
   (\vec r_e - \hat \mu)'\left(\tau \Sigma\right)^{-1}(\vec r_e - \hat \mu) &= \bar d_2
   \end{align}

Notice that for particular :math:`\bar d_1` and :math:`\bar d_2` values
the two ellipses have a tangency point

These tangency points, indexed
by the pairs :math:`(\bar d_1, \bar d_2)`, characterize points
:math:`\vec r_e` from which there exists no deviation where one can
increase the likelihood of one view without decreasing the likelihood of
the other view

The pairs :math:`(\bar d_1, \bar d_2)` for which there
is such a point outlines a curve in the excess return space. This curve
is reminiscent of the Pareto curve in an Edgeworth-box setting

Leamer (1978) :cite:`leamer1978specification` calls this curve *information contract curve* and
describes it by the following program: maximize the likelihood of one
view, say the Black-Litterman recommendation, while keeping the
likelihood of the other view at least at a prespecified constant
:math:`\bar d_2`

.. math::

   \begin{align*}
    \bar d_1(\bar d_2) &\equiv \max_{\vec r_e} \ \ (\vec r_e - \mu_{BL})'\Sigma^{-1}(\vec r_e - \mu_{BL}) \\
   \text{subject to }  \quad  &(\vec r_e - \hat\mu)'(\tau\Sigma)^{-1}(\vec r_e - \hat \mu) \geq \bar d_2
   \end{align*}

Denoting the multiplier on the constraint by :math:`\lambda`, the
first-order condition is

.. math:: 2(\vec r_e - \mu_{BL} )'\Sigma^{-1} + \lambda 2(\vec r_e - \hat\mu)'(\tau\Sigma)^{-1} = \mathbf{0}

which defines the *information contract curve* between
:math:`\mu_{BL}` and :math:`\hat \mu`

.. math:: 
  :label: info-curve
  
  \vec r_e = (\Sigma^{-1} + \lambda (\tau \Sigma)^{-1})^{-1} (\Sigma^{-1} \mu_{BL}  
  + \lambda (\tau \Sigma)^{-1}\hat \mu )

Note that if :math:`\lambda = 1`, :eq:`info-curve` is equivalent with :eq:`mix-views` and it
identifies one point on the information contract curve. Furthermore,
because :math:`\lambda` is a function of the minimum likelihood
:math:`\bar d_2` on the RHS of the constraint, by varying
:math:`\bar d_2` (or :math:`\lambda` ), we can trace out the whole curve
as the figure below illustrates

.. code-block:: python3

    np.random.seed(1987102)
    
    N = 2                                           # Number of assets
    T = 200                                         # Sample size
    τ = 0.8
    
    # Random market portfolio (sum is normalized to 1)
    w_m = np.random.rand(N) 
    w_m = w_m / (w_m.sum())                      
    
    μ = (np.random.randn(N) + 5) / 100                   
    S = np.random.randn(N, N)                       
    V = S @ S.T                                     
    Σ = V * (w_m @ μ)**2 / (w_m @ V @ w_m)
    
    excess_return = stat.multivariate_normal(μ, Σ)
    sample = excess_return.rvs(T)
    
    μ_est = sample.mean(0).reshape(N, 1)
    Σ_est = np.cov(sample.T)
    
    σ_m = w_m @ Σ_est @ w_m
    d_m = (w_m @ μ_est) / σ_m
    μ_m = (d_m * Σ_est @ w_m).reshape(N, 1)

    N_r1, N_r2 = 100, 100
    r1 = np.linspace(-0.04, .1, N_r1)
    r2 = np.linspace(-0.02, .15, N_r2)
    
    λ_grid = np.linspace(.001, 20, 100)
    curve = np.asarray([black_litterman(λ, μ_m, μ_est, Σ_est, 
                                        τ * Σ_est).flatten() for λ in λ_grid]) 
    
    λ_slider = FloatSlider(min=.1, max=7, step=.5, value=1)
    
    @interact(λ=λ_slider)
    def decolletage(λ):
        dist_r_BL = stat.multivariate_normal(μ_m.squeeze(), Σ_est)
        dist_r_hat = stat.multivariate_normal(μ_est.squeeze(), τ * Σ_est)
        
        X, Y = np.meshgrid(r1, r2)
        Z_BL = np.zeros((N_r1, N_r2))
        Z_hat = np.zeros((N_r1, N_r2))
    
        for i in range(N_r1):
            for j in range(N_r2):
                Z_BL[i, j] = dist_r_BL.pdf(np.hstack([X[i, j], Y[i, j]]))
                Z_hat[i, j] = dist_r_hat.pdf(np.hstack([X[i, j], Y[i, j]]))
        
        μ_tilde = black_litterman(λ, μ_m, μ_est, Σ_est, τ * Σ_est).flatten()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contourf(X, Y, Z_hat, cmap='viridis', alpha =.4)
        ax.contourf(X, Y, Z_BL, cmap='viridis', alpha =.4)
        ax.contour(X, Y, Z_BL, [dist_r_BL.pdf(μ_tilde)], cmap='viridis', alpha=.9)
        ax.contour(X, Y, Z_hat, [dist_r_hat.pdf(μ_tilde)], cmap='viridis', alpha=.9)
        ax.scatter(μ_est[0], μ_est[1])
        ax.scatter(μ_m[0], μ_m[1])
        ax.scatter(μ_tilde[0], μ_tilde[1], c='k', s=20*3)
        
        ax.plot(curve[:, 0], curve[:, 1], c='k')
        ax.axhline(0, c='k', alpha=.8)
        ax.axvline(0, c='k', alpha=.8)
        ax.set_xlabel(r'Excess return on the first asset, $r_{e, 1}$')
        ax.set_ylabel(r'Excess return on the second asset, $r_{e, 2}$')
        ax.text(μ_est[0] + 0.003, μ_est[1], r'$\hat{\mu}$')
        ax.text(μ_m[0] + 0.003, μ_m[1] + 0.005, r'$\mu_{BL}$')
        plt.show()


Note that the line that connects the two points
:math:`\hat \mu` and :math:`\mu_{BL}` is linear, which comes from the
fact that the covariance matrices of the two competing distributions
(views) are proportional to each other

To illustrate the fact that this is not necessarily the case, consider
another example using the same parameter values, except that the "second
view" constituting the constraint has covariance matrix
:math:`\tau I` instead of :math:`\tau \Sigma`

This leads to the
following figure, on which the curve connecting :math:`\hat \mu`
and :math:`\mu_{BL}` are bending

.. code-block:: python3

    λ_grid = np.linspace(.001, 20000, 1000)
    curve = np.asarray([black_litterman(λ, μ_m, μ_est, Σ_est, 
                                        τ * np.eye(N)).flatten() for λ in λ_grid]) 
    
    λ_slider = FloatSlider(min=5, max=1500, step=100, value=200)
    
    @interact(λ=λ_slider)
    def decolletage(λ):
        dist_r_BL = stat.multivariate_normal(μ_m.squeeze(), Σ_est)
        dist_r_hat = stat.multivariate_normal(μ_est.squeeze(), τ * np.eye(N))
        
        X, Y = np.meshgrid(r1, r2)
        Z_BL = np.zeros((N_r1, N_r2))
        Z_hat = np.zeros((N_r1, N_r2))
    
        for i in range(N_r1):
            for j in range(N_r2):
                Z_BL[i, j] = dist_r_BL.pdf(np.hstack([X[i, j], Y[i, j]]))
                Z_hat[i, j] = dist_r_hat.pdf(np.hstack([X[i, j], Y[i, j]]))
        
        μ_tilde = black_litterman(λ, μ_m, μ_est, Σ_est, τ * np.eye(N)).flatten()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contourf(X, Y, Z_hat, cmap='viridis', alpha=.4)
        ax.contourf(X, Y, Z_BL, cmap='viridis', alpha=.4)
        ax.contour(X, Y, Z_BL, [dist_r_BL.pdf(μ_tilde)], cmap='viridis', alpha=.9)
        ax.contour(X, Y, Z_hat, [dist_r_hat.pdf(μ_tilde)], cmap='viridis', alpha=.9)
        ax.scatter(μ_est[0], μ_est[1])
        ax.scatter(μ_m[0], μ_m[1])
        
        ax.scatter(μ_tilde[0], μ_tilde[1], c='k', s=20*3)
        
        ax.plot(curve[:, 0], curve[:, 1], c='k')
        ax.axhline(0, c='k', alpha=.8)
        ax.axvline(0, c='k', alpha=.8)
        ax.set_xlabel(r'Excess return on the first asset, $r_{e, 1}$')
        ax.set_ylabel(r'Excess return on the second asset, $r_{e, 2}$')
        ax.text(μ_est[0] + 0.003, μ_est[1], r'$\hat{\mu}$')
        ax.text(μ_m[0] + 0.003, μ_m[1] + 0.005, r'$\mu_{BL}$')
        plt.show()
    

Black-Litterman recommendation as regularization
--------------------------------------------------

First, consider the OLS regression

.. math:: \min_{\beta} \Vert X\beta - y \Vert^2

which yields the solution

.. math:: \hat{\beta}_{OLS} = (X'X)^{-1}X'y

A common performance measure of estimators is the *mean squared error
(MSE)*

An estimator is "good" if its MSE is realtively small. Suppose
that :math:`\beta_0` is the "true" value of the coefficient, then the MSE
of the OLS estimator is

.. math:: 
  
  \text{mse}(\hat \beta_{OLS}, \beta_0) := \mathbb E \Vert \hat \beta_{OLS} - \beta_0\Vert^2 = 
  \underbrace{\mathbb E \Vert \hat \beta_{OLS} - \mathbb E  
  \beta_{OLS}\Vert^2}_{\text{variance}} + 
  \underbrace{\Vert \mathbb E \hat\beta_{OLS} - \beta_0\Vert^2}_{\text{bias}}

From this decomposition one can see that in order for the MSE to be
small, both the bias and the variance terms must be small

For example,
consider the case when :math:`X` is a :math:`T`-vector of ones (where
:math:`T` is the sample size), so :math:`\hat\beta_{OLS}` is simply the
sample average, while :math:`\beta_0\in \mathbb{R}` is defined by the
true mean of :math:`y`

In this example the MSE is

.. math:: 
  
  \text{mse}(\hat \beta_{OLS}, \beta_0) = \underbrace{\frac{1}{T^2} 
  \mathbb E \left(\sum_{t=1}^{T} (y_{t}- \beta_0)\right)^2 }_{\text{variance}} + 
  \underbrace{0}_{\text{bias}}

However, because there is a trade-off between the estimator's bias and
variance, there are cases when by permitting a small bias we can
substantially reduce the variance so overall the MSE gets smaller

A typical scenario when this proves to be useful is when the number of
coefficients to be estimated is large relative to the sample size

In these cases one approach to handle the bias-variance trade-off is the
so called *Tikhonov regularization*

A general form with regularization matrix :math:`\Gamma` can be written as

.. math:: \min_{\beta} \Big\{ \Vert X\beta - y \Vert^2 + \Vert \Gamma (\beta - \tilde \beta) \Vert^2 \Big\}

which yields the solution

.. math:: \hat{\beta}_{Reg} = (X'X + \Gamma'\Gamma)^{-1}(X'y + \Gamma'\Gamma\tilde \beta)

Substituting the value of :math:`\hat{\beta}_{OLS}` yields

.. math:: \hat{\beta}_{Reg} = (X'X + \Gamma'\Gamma)^{-1}(X'X\hat{\beta}_{OLS} + \Gamma'\Gamma\tilde \beta)

Often, the regularization matrix takes the form
:math:`\Gamma = \lambda I` with :math:`\lambda>0`
and :math:`\tilde \beta = \mathbf{0}`

Then the Tikhonov regularization is equivalent to what is called *ridge regression* in statistics

To illustrate how this estimator addresses the bias-variance trade-off,
we compute the MSE of the ridge estimator

.. math:: 
  
  \text{mse}(\hat \beta_{\text{ridge}}, \beta_0) = \underbrace{\frac{1}{(T+\lambda)^2} 
  \mathbb E \left(\sum_{t=1}^{T} (y_{t}- \beta_0)\right)^2 }_{\text{variance}} + 
  \underbrace{\left(\frac{\lambda}{T+\lambda}\right)^2 \beta_0^2}_{\text{bias}}

The ridge regression shrinks the coefficients of the estimated vector
towards zero relative to the OLS estimates thus reducing the variance
term at the cost of introducing a "small" bias

However, there is nothing special about the zero vector

When :math:`\tilde \beta \neq \mathbf{0}` shrinkage occurs in the direction
of :math:`\tilde \beta`

Now, we can give a regularization interpretation of the Black-Litterman
portfolio recommendation

To this end, simplify first the equation :eq:`mix-views` characterizing the Black-Litterman recommendation

.. math::

   \begin{align*}
   \tilde \mu &= (\Sigma^{-1} + (\tau \Sigma)^{-1})^{-1} (\Sigma^{-1}\mu_{BL}  + (\tau \Sigma)^{-1}\hat \mu) \\
   &= (1 + \tau^{-1})^{-1}\Sigma \Sigma^{-1} (\mu_{BL}  + \tau ^{-1}\hat \mu) \\
   &= (1 + \tau^{-1})^{-1} ( \mu_{BL}  + \tau ^{-1}\hat \mu)
   \end{align*}

In our case, :math:`\hat \mu` is the estimated mean excess returns of
securities. This could be written as a vector autoregression where

-  :math:`y` is the stacked vector of observed excess returns of size
   :math:`(N T\times 1)` -- :math:`N` securities and :math:`T`
   observations

-  :math:`X = \sqrt{T^{-1}}(I_{N} \otimes \iota_T)` where :math:`I_N`
   is the identity matrix and :math:`\iota_T` is a column vector of
   ones.

Correspondingly, the OLS regression of :math:`y` on :math:`X` would
yield the mean excess returns as coefficients

With :math:`\Gamma = \sqrt{\tau T^{-1}}(I_{N} \otimes \iota_T)` we can
write the regularized version of the mean excess return estimation

.. math::

   \begin{align*}
   \hat{\beta}_{Reg} &= (X'X + \Gamma'\Gamma)^{-1}(X'X\hat{\beta}_{OLS} + \Gamma'\Gamma\tilde \beta) \\
   &= (1 + \tau)^{-1}X'X (X'X)^{-1} (\hat \beta_{OLS}  + \tau \tilde \beta) \\
   &= (1 + \tau)^{-1} (\hat \beta_{OLS}  + \tau \tilde \beta) \\
   &= (1 + \tau^{-1})^{-1} ( \tau^{-1}\hat \beta_{OLS}  +  \tilde \beta)
   \end{align*}

Given that
:math:`\hat \beta_{OLS} = \hat \mu` and :math:`\tilde \beta = \mu_{BL}`
in the Black-Litterman model we have the following interpretation of the
model's recommendation

The estimated (personal) view of the mean excess returns,
:math:`\hat{\mu}` that would lead to extreme short-long positions are
"shrunk" towards the conservative market view, :math:`\mu_{BL}`, that
leads to the more conservative market portfolio

So the Black-Litterman procedure results in a recommendation that is a
compromise between the conservative market portfolio and the more
extreme portfolio that is implied by estimated "personal" views

Digression on :math:`{\sf T}` operator
---------------------------------------

The Black-Litterman approach is partly inspired by the econometric
insight that it is easier to estimate covariances of excess returns than
the means

That is what gave Black and Litterman license to adjust
investors' perception of mean excess returns while not tampering with
the covariance matrix of excess returns

The robust control theory is another approach that also hinges on
adjusting mean excess returns but not covariances

Associated with a robust control problem is what Hansen and Sargent call
a :math:`{\sf T}` operator

Let's define the :math:`{\sf T}` operator as it applies to the problem
at hand

Let :math:`x` be an :math:`n \times 1` Gaussian random vector with mean
vector :math:`\mu` and covariance matrix :math:`\Sigma = C C'`. This
means that :math:`x` can be represented as

.. math:: x = \mu + C \epsilon 

where :math:`\epsilon \sim {\mathcal N}(0,I)`

Let :math:`\phi(\epsilon)` denote the associated standardized Gaussian
density

Let :math:`m(\epsilon,\mu)` be a **likelihood ratio**, meaning that it
satisfies

-  :math:`m(\epsilon, \mu) > 0`

-  :math:`\int m(\epsilon,\mu) \phi(\epsilon) d \epsilon =1`

That is, :math:`m(\epsilon, \mu)` is a nonnegative random variable with
mean 1

Multiplying :math:`\phi(\epsilon)`
by the likelihood ratio :math:`m(\epsilon, \mu)` produces a distorted distribution for
:math:`\epsilon`, namely

.. math:: \tilde \phi(\epsilon) = m(\epsilon,\mu) \phi(\epsilon) 

The next concept that we need is the **entropy** of the distorted
distribution :math:`\tilde \phi` with respect to :math:`\phi`

**Entropy** is defined as

.. math:: {\rm ent} = \int \log m(\epsilon,\mu) m(\epsilon,\mu) \phi(\epsilon) d \epsilon

or

.. math:: {\rm ent} = \int \log m(\epsilon,\mu) \tilde \phi(\epsilon) d \epsilon

That is, relative entropy is the expected value of the likelihood ratio
:math:`m` where the expectation is taken with respect to the twisted
density :math:`\tilde \phi`

Relative entropy is nonnegative. It is a measure of the discrepancy
between two probability distributions

As such, it plays an important
role in governing the behavior of statistical tests designed to
discriminate one probability distribution from another

We are ready to define the :math:`{\sf T}` operator

Let :math:`V(x)` be a value function

Define

.. math::

    \eqalign{ {\sf T}\left(V(x)\right) & = \min_{m(\epsilon,\mu)} \int m(\epsilon,\mu)[V(\mu + C \epsilon) + \theta \log m(\epsilon,\mu) ] \phi(\epsilon) d \epsilon \cr
                            & = - \log \theta \int \exp \left( \frac{- V(\mu + C \epsilon)}{\theta} \right) \phi(\epsilon) d \epsilon } 

This asserts that :math:`{\sf T}` is an indirect utility function for a
minimization problem in which an **evil agent** chooses a distorted
probability distribution :math:`\tilde \phi` to lower expected utility,
subject to a penalty term that gets bigger the larger is relative
entropy

Here the penalty parameter

.. math:: \theta \in [\underline \theta, +\infty]

is a robustness parameter when it is :math:`+\infty`, there is no scope for the minimizing agent to distort the distribution,
so no robustness to alternative distributions is acquired 
As :math:`\theta` is lowered, more robustness is achieved

**Note:** The :math:`{\sf T}` operator is sometimes called a
*risk-sensitivity* operator

We shall apply :math:`{\sf T}`\ to the special case of a linear value
function :math:`w'(\vec r - r_f 1)` 
where :math:`\vec r - r_f 1 \sim {\mathcal N}(\mu,\Sigma)` or
:math:`\vec r - r_f {\bf 1} = \mu + C \epsilon`\ and
:math:`\epsilon \sim {\mathcal N}(0,I)`

The associated worst-case distribution of :math:`\epsilon` is Gaussian
with mean :math:`v =-\theta^{-1} C' w` and covariance matrix :math:`I`
(When the value function is affine, the worst-case distribution distorts
the mean vector of :math:`\epsilon` but not the covariance matrix
of :math:`\epsilon`)

For utility function argument :math:`w'(\vec r - r_f 1)`

.. math:: {\sf T} ( \vec r - r_f {\bf 1}) = w' \mu + \zeta - \frac{1}{2 \theta} w' \Sigma w

and entropy is

.. math:: \frac{v'v}{2} = \frac{1}{2\theta^2}  w' C C' w

A robust mean-variance portfolio model
-------------------------------------------

According to criterion (1), the mean-variance portfolio choice problem
chooses :math:`w` to maximize

.. math:: E [w ( \vec r - r_f {\bf 1})]] - {\rm var} [ w ( \vec r - r_f {\bf 1}) ]

which equals

.. math:: w'\mu - \frac{\delta}{2} w' \Sigma w    

A robust decision maker can be modelled as replacing the mean return
:math:`E [w ( \vec r - r_f {\bf 1})]` with the risk-sensitive

.. math:: {\sf T} [w ( \vec r - r_f {\bf 1})] = w' \mu - \frac{1}{2 \theta} w' \Sigma w

that comes from replacing the mean :math:`\mu` of :math:`\vec r - r\_f {\bf 1}` with the worst-case mean

.. math:: \mu - \theta^{-1} \Sigma w  

Notice how the worst-case mean vector depends on the portfolio
:math:`w`

The operator :math:`{\sf T}` is the indirect utility function that
emerges from solving a problem in which an agent who chooses
probabilities does so in order to minimize the expected utility of a
maximizing agent (in our case, the maximizing agent chooses portfolio
weights :math:`w`)

The robust version of the mean-variance portfolio choice problem is then
to choose a portfolio :math:`w` that maximizes

.. math:: {\sf T} [w ( \vec r - r_f {\bf 1})] - \frac{\delta}{2} w' \Sigma w 

or

.. math::
  :label: robust-mean-variance
  
  w' (\mu - \theta^{-1} \Sigma w ) - \frac{\delta}{2} w' \Sigma w

The minimizer of :eq:`robust-mean-variance` is

.. math:: w_{\rm rob} = \frac{1}{\delta + \gamma } \Sigma^{-1} \mu

where :math:`\gamma \equiv \theta^{-1}` is sometimes called the
risk-sensitivity parameter

An increase in the risk-sensitivity parameter :math:`\gamma` shrinks the
portfolio weights toward zero in the same way that an increase in risk
aversion does

--------------

Appendix
========

We want to illustrate the "folk theorem" that with high or moderate
frequency data, it is more difficult to estimate means than variances

In order to operationalize this statement, we take two analog
estimators:

-  sample average: :math:`\bar X_N = \frac{1}{N}\sum_{i=1}^{N} X_i`
-  sample variance:
   :math:`S_N = \frac{1}{N-1}\sum_{t=1}^{N} (X_i - \bar X_N)^2`

to estimate the unconditional mean and unconditional variance of the
random variable :math:`X`, respectively

To measure the "difficulty of estimation", we use *mean squared error*
(MSE), that is the average squared difference between the estimator and
the true value

Assuming that the process :math:`\{X_i\}`\ is ergodic,
both analog estimators are known to converge to their true values as the
sample size :math:`N` goes to infinity

More precisely
for all :math:`\varepsilon > 0`

.. math:: \lim_{N\to \infty} \ \ P\left\{ \left |\bar X_N - \mathbb E X \right| > \varepsilon \right\} = 0 \quad \quad

and

.. math:: \lim_{N\to \infty} \ \ P \left\{ \left| S_N - \mathbb V X \right| > \varepsilon \right\} = 0

A necessary condition for these convergence results is that the
associated MSEs vanish as :math:`N` goes to infintiy, or in other words,

.. math:: \text{MSE}(\bar X_N, \mathbb E X) = o(1) \quad \quad  \text{and} \quad \quad \text{MSE}(S_N, \mathbb V X) = o(1)

Even if the MSEs converge to zero, the associated rates might be
different. Looking at the limit of the *relative MSE* (as the sample
size grows to infinity)

.. math:: \frac{\text{MSE}(S_N, \mathbb V X)}{\text{MSE}(\bar X_N, \mathbb E X)} = \frac{o(1)}{o(1)} \underset{N \to \infty}{\to} B

can inform us about the relative (asymptotic) rates

We will show that in general, with dependent data, the limit
:math:`B` depends on the sampling frequency. In particular, we find
that the rate of convergence of the variance estimator is less sensitive
to increased sampling frequency than the rate of convergence of the mean
estimator. Hence, we can expect the relative asymptotic
rate, :math:`B`, to get smaller with higher frequency data,
illustrating that "it is more difficult to estimate means than
variances". That is, we need significantly more data to obtain a given
precision of the mean estimate than for our variance estimate

A special case -- i.i.d. sample
---------------------------------

We start our analysis with the benchmark case of iid data. Consider a
sample of size :math:`N` generated by the following iid process,

.. math:: X_i \sim \mathcal{N}(\mu, \sigma^2)

Taking :math:`\bar X_N` to estimate the mean, the MSE is

.. math:: \text{MSE}(\bar X_N, \mu) = \frac{\sigma^2}{N}

Taking :math:`S_N` to estimate the variance, the MSE is

.. math:: \text{MSE}(S_N, \sigma^2) = \frac{2\sigma^4}{N-1}

Both estimators are unbiased and hence the MSEs reflect the
corresponding variances of the estimators

Furthermore, both MSEs are
:math:`o(1)` with a (multiplicative) factor of difference in their rates
of convergence:

.. math:: \frac{\text{MSE}(S_N, \sigma^2)}{\text{MSE}(\bar X_N, \mu)} = \frac{N2\sigma^2}{N-1} \quad \underset{N \to \infty}{\to} \quad 2\sigma^2

We are interested in how this (asymptotic) relative rate of convergence
changes as increasing sampling frequency puts dependence into the data

Dependence and sampling frequency
----------------------------------

To investigate how sampling frequency affects relative rates of
convergence, we assume that the data are generated by a mean-reverting
continuous time process of the form

.. math:: dX_t = -\kappa (X_t -\mu)dt + \sigma dW_t\quad\quad

where :math:`\mu`\ is the unconditional mean, :math:`\kappa > 0` is a
persistence parameter, and :math:`\{W_t\}` is a standardized Brownian
motion

Observations arising from this system in particular discrete periods
:math:`\mathcal T(h) \equiv \{nh : n \in \mathbb Z \}`\ with\ :math:`h>0`
can be described by the following process

.. math:: X_{t+1} = (1 - \exp(-\kappa h))\mu + \exp(-\kappa h)X_t + \epsilon_{t, h}

where

.. math:: \epsilon_{t, h} \sim \mathcal{N}(0, \Sigma_h) \quad \text{with}\quad \Sigma_h = \frac{\sigma^2(1-\exp(-2\kappa h))}{2\kappa}

We call :math:`h` the *frequency* parameter, whereas :math:`n`
represents the number of *lags* between observations

Hence, the effective distance between two observations :math:`X_t` and
:math:`X_{t+n}` in the discrete time notation is equal
to :math:`h\cdot n` in terms of the underlying continuous time process

Straightforward calculations show that the autocorrelation function for
the stochastic process :math:`\{X_{t}\}_{t\in \mathcal T(h)}` is

.. math:: \Gamma_h(n) \equiv \text{corr}(X_{t + h n}, X_t) = \exp(-\kappa h n)

and the auto-covariance function is

.. math:: \gamma_h(n) \equiv \text{cov}(X_{t + h n}, X_t) = \frac{\exp(-\kappa h n)\sigma^2}{2\kappa} .

It follows that if :math:`n=0`, the unconditional variance is given
by :math:`\gamma_h(0) = \frac{\sigma^2}{2\kappa}` irrespective of the
sampling frequency

The following figure illustrates how the dependence between the
observations is related to sampling frequency

-  For any given :math:`h`, the autocorrelation converges to zero as we increase the distance -- :math:`n`-- between the observations. This represents the "weak dependence" of the :math:`X` process

-  Moreover, for a fixed lag length, :math:`n`, the dependence vanishes as the sampling frequency goes to infinity. In fact, letting :math:`h` go to :math:`\infty` gives back the case of i.i.d. data

.. code-block:: python3

    μ = .0
    κ = .1 
    σ = .5
    var_uncond = σ**2 / (2 * κ) 
    
    n_grid = np.linspace(0, 40, 100)
    autocorr_h1 = np.exp(-κ * n_grid * 1)
    autocorr_h2 = np.exp(-κ * n_grid * 2)
    autocorr_h5 = np.exp(-κ * n_grid * 5)
    autocorr_h1000 = np.exp(-κ * n_grid * 1e8)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_grid, autocorr_h1, label=r'$h=1$', c='darkblue', lw=2)
    ax.plot(n_grid, autocorr_h2, label=r'$h=2$', c='darkred', lw=2)
    ax.plot(n_grid, autocorr_h5, label=r'$h=5$', c='orange', lw=2)
    ax.plot(n_grid, autocorr_h1000, label=r'"$h=\infty$"', c='darkgreen', lw=2)
    ax.legend()
    ax.grid()
    ax.set(title=r'Autocorrelation functions, $\Gamma_h(n)$', 
           xlabel=r'Lags between observations, $n$')
    plt.show()


Frequency and the mean estimator
---------------------------------

Consider again the AR(1) process generated by discrete sampling with
frequency :math:`h`. Assume that we have a sample of size :math:`N` and
we would like to estimate the unconditional mean -- in our case the true
mean is :math:`\mu`

Again, the sample average is an unbiased estimator of the unconditional
mean

.. math:: \mathbb{E}[\bar X_N] = \frac{1}{N}\sum_{i = 1}^N \mathbb{E}[X_i] = \mathbb{E}[X_0] = \mu

The variance of the sample mean is given by

.. math::

   \begin{align}
   \mathbb{V}\left(\bar X_N\right) &= \mathbb{V}\left(\frac{1}{N}\sum_{i = 1}^N X_i\right) \\
   &= \frac{1}{N^2} \left(\sum_{i = 1}^N \mathbb{V}(X_i) + 2 \sum_{i = 1}^{N-1} \sum_{s = i+1}^N \text{cov}(X_i, X_s) \right) \\
   &= \frac{1}{N^2} \left( N \gamma(0) + 2 \sum_{i=1}^{N-1} i \cdot \gamma\left(h\cdot (N - i)\right) \right) \\
   &= \frac{1}{N^2} \left( N \frac{\sigma^2}{2\kappa} + 2 \sum_{i=1}^{N-1} i \cdot \exp(-\kappa h (N - i)) \frac{\sigma^2}{2\kappa} \right)
   \end{align}

It is explicit in the above equation that time dependence in the data
inflates the variance of the mean estimator through the covariance
terms. Moreover, as we can see, a higher sampling frequency---smaller
:math:`h`---makes all the covariance terms larger everything else being
fixed. This implies a relatively slower rate of convergence of the
sample average for high frequency data 

Intuitively, the stronger dependence across observations for high frequency data reduces the
"information content" of each observation relative to the iid case

We can upper bound the variance term in the following way

.. math::

   \begin{align}
   \mathbb{V}(\bar X_N) &= \frac{1}{N^2} \left( N \sigma^2 + 2 \sum_{i=1}^{N-1} i \cdot \exp(-\kappa h (N - i)) \sigma^2 \right) \\
   &\leq \frac{\sigma^2}{2\kappa N} \left(1 + 2 \sum_{i=1}^{N-1} \cdot \exp(-\kappa h (i)) \right) \\
   &= \underbrace{\frac{\sigma^2}{2\kappa N}}_{\text{i.i.d.  case}} \left(1 + 2 \frac{1 - \exp(-\kappa h)^{N-1}}{1 - \exp(-\kappa h)} \right)
   \end{align}

Asymptotically the :math:`\exp(-\kappa h)^{N-1}` vanishes and the
dependence in the data inflates the benchmark iid variance by a factor
of 

.. math:: \left(1 + 2 \frac{1}{1 - \exp(-\kappa h)} \right)

This long run factor is larger the higher is the frequency (the smaller
is :math:`h`)

Therefore, we expect the asymptotic relative MSEs, :math:`B`, to change
with time dependent data. We just saw that the mean estimator's rate is
roughly changing by a factor of

.. math:: \left(1 + 2 \frac{1}{1 - \exp(-\kappa h)} \right)

Unfortunately, the variance estimator's MSE is harder to derive

Nonetheless, we can approximate it by using (large sample) simulations,
thus getting an idea about how the asymptotic relative MSEs changes in
the sampling frequency :math:`h` relative to the iid case that we
compute in closed form

.. code-block:: python3

    def sample_generator(h, N, M):
        ϕ = (1 - np.exp(-κ * h)) * μ
        ρ = np.exp(-κ * h)
        s = σ**2 * (1 - np.exp(-2 * κ * h)) / (2 * κ)
    
        mean_uncond = μ
        std_uncond = np.sqrt(σ**2 / (2 * κ))
    
        ε_path = stat.norm(0, np.sqrt(s)).rvs((M, N)) 
        
        y_path = np.zeros((M, N + 1))
        y_path[:, 0] = stat.norm(mean_uncond, std_uncond).rvs(M)
    
        for i in range(N):
            y_path[:, i + 1] = ϕ + ρ * y_path[:, i] + ε_path[:, i]
        
        return y_path


.. code-block:: python3

    # Generate large sample for different frequencies
    N_app, M_app = 1000, 30000                      # Sample size, number of simulations
    h_grid = np.linspace(.1, 80, 30)
    
    var_est_store = []
    mean_est_store = []
    labels = []
    
    for h in h_grid:
        labels.append(h)
        sample = sample_generator(h, N_app, M_app)
        mean_est_store.append(np.mean(sample, 1))
        var_est_store.append(np.var(sample, 1))
        
    var_est_store = np.array(var_est_store)
    mean_est_store = np.array(mean_est_store)

    # Save mse of estimators
    mse_mean = np.var(mean_est_store, 1) + (np.mean(mean_est_store, 1) - μ)**2
    mse_var = np.var(var_est_store, 1) + (np.mean(var_est_store, 1) - var_uncond)**2
    
    benchmark_rate = 2 * var_uncond       # iid case
    
    # Relative MSE for large samples
    rate_h = mse_var / mse_mean

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h_grid, rate_h, c='darkblue', lw=2, 
            label=r'large sample relative MSE, $B(h)$')
    ax.axhline(benchmark_rate, c='k', ls='--', label=r'iid benchmark')
    ax.set_title('Relative MSE for large samples as a function of sampling frequency \n MSE($S_N$) relative to MSE($\\bar X_N$)')
    ax.set_xlabel('Sampling frequency, $h$')
    ax.legend()
    plt.show()


The above figure illustrates the relationship between the asymptotic
relative MSEs and the sampling frequency

-  We can see that with low frequency data -- large values of :math:`h`
   -- the ratio of asymptotic rates approaches the iid case

-  As :math:`h` gets smaller -- the higher the frequency -- the relative
   performance of the variance estimator is better in the sense that the
   ratio of asymptotic rates gets smaller. That is, as the time
   dependence gets more pronounced, the rate of convergence of the mean
   estimator's MSE deteriorates more than that of the variance
   estimator

References
------------

Black, F. and Litterman, R., 1992. "Global portfolio optimization".
Financial analysts journal, 48(5), pp.28-43

Dickey, J. 1975. "Bayesian alternatives to the F-test and least-squares
estimate in the normal linear model", in: S.E. Fienberg and A. Zellner,
eds., "Studies in Bayesian econometrics and statistics" (North-Holland,
Amsterdam) 515-554

Hansen, Lars Peter and Thomas J. Sargent. 2001. "Robust Control and
Model Uncertainty." American Economic Review, 91(2): 60-66

Leamer, E.E., 1978. **Specification searches: Ad hoc inference with
nonexperimental data**, (Vol. 53). John Wiley & Sons Incorporated

